"""Native 外层优化层单元测试（unittest，无需 pytest；需 torch）。

运行：python -m unittest discover -s Nooht_Native/tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from native import backend as backend_module
from native.backend import Backend, benchmark, check_correctness, detect_backend
from native.config import NativeConfig, lumenoo_config
from native.copy_mechanism import (
    CopyDecoder,
    CopySpan,
    auto_construct_copy_labels,
    find_longest_overlap,
)
from native.generation import NativeGenerator
from noild import NoildConfig, NoildLayer
from native.memory_compression import (
    Segment,
    SelectiveMemoryCompressor,
    split_segments,
)
from native.protected_zone import (
    InstructionAnchor,
    PersonaAdapter,
    ProtectedZone,
    ThinkingMemoryBlock,
)


class TestBackend(unittest.TestCase):
    def test_detect(self):
        self.assertIsInstance(detect_backend(), Backend)

    def test_rocm_detection_uses_valid_enum(self):
        original_rocm = backend_module._has_rocm
        original_cuda = backend_module._has_cuda
        try:
            backend_module._has_rocm = lambda: True
            backend_module._has_cuda = lambda: True
            self.assertIs(detect_backend(), Backend.ROCM)
        finally:
            backend_module._has_rocm = original_rocm
            backend_module._has_cuda = original_cuda

    def test_correctness_and_benchmark(self):
        report = check_correctness(Backend.CPU)
        self.assertTrue(report["forward_ok"])
        self.assertTrue(report["backward_ok"])
        gated = check_correctness(Backend.CPU, op="gated_delta")
        self.assertTrue(gated["forward_ok"])
        self.assertTrue(gated["backward_ok"])
        bench = benchmark(Backend.CPU, steps=2)
        self.assertGreater(bench["avg_seconds"], 0)


class TestMemoryCompression(unittest.TestCase):
    def test_split_and_protected_excluded(self):
        text = "第一段内容。\n第二段内容！\n第三段。"
        segs = split_segments(text)
        self.assertGreaterEqual(len(segs), 2)
        segs[0].protected = True
        comp = SelectiveMemoryCompressor(enabled=True, low_score_threshold=10.0)
        out = comp.compress(segs)
        # 受保护片段不压缩（原文保留），非保护低分片段被替换
        self.assertEqual(out[0].text, segs[0].text)

    def test_threshold_and_ablation(self):
        comp = SelectiveMemoryCompressor(threshold_ratio=0.5, max_context_len=100)
        self.assertFalse(comp.should_compress(10))
        self.assertTrue(comp.should_compress(60))
        comp.enabled = False
        self.assertFalse(comp.should_compress(60))
        comp.enabled = True
        segs = split_segments("这是一段很长的上下文内容。\n" * 5)
        report = comp.ablation(segs)
        self.assertIn("saved_ratio", report)

    def test_extractive_summary_uses_complete_information_units(self):
        from native.memory_compression import extractive_summary
        text = "旧背景描述很长很长很长很长很长很长很长很长。错误码 E42 必须修复 /tmp/data.json。"
        summary = extractive_summary(text, max_chars=36)
        self.assertIn("E42", summary)
        self.assertIn("摘要", summary)
        self.assertLess(len(summary), len(text))


class TestProtectedZone(unittest.TestCase):
    def test_persona_and_instruction(self):
        zone = ProtectedZone(
            persona_text="冷静的助手",
            instruction=InstructionAnchor(
                system_prompt="你是助手", hard_constraints=["只输出 JSON"], reinject_every=2
            ),
        )
        text = zone.assemble("用户问题", step=0)
        self.assertIn("人设锚点", text)
        self.assertIn("硬约束", text)
        self.assertTrue(zone.instruction.should_reinject(2))
        self.assertTrue(zone.is_protected_text(text))

    def test_thinking_memory_incremental_only(self):
        block = ThinkingMemoryBlock(capacity=2, min_interval=2)
        block.mark_new_info()
        self.assertTrue(block.edit_entry("variables", "x", "1", step=0))
        # 间隔未到：拒绝
        block.mark_new_info()
        self.assertFalse(block.edit_entry("variables", "y", "2", step=1))
        # 间隔已到：允许；且无整块重写接口
        self.assertTrue(block.edit_entry("variables", "y", "2", step=2))
        self.assertFalse(hasattr(block, "rewrite_all"))
        with self.assertRaises(ValueError):
            block.edit_entry("unknown", "k", "v", step=10)

    def test_persona_adapter_is_hooked_into_model(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
        adapter = PersonaAdapter(target_modules=["0"], rank=2, alpha=2)
        NativeGenerator(model, persona_adapter=adapter)
        x = torch.randn(2, 4)
        baseline = model(x)
        key = adapter._name_to_key["0"]
        with torch.no_grad():
            adapter.lora_B[key].fill_(0.5)
        adapter.load()
        adapted = model(x)
        self.assertFalse(torch.allclose(baseline, adapted))
        adapter.unload()
        self.assertTrue(torch.allclose(baseline, model(x)))


class TestCopyMechanism(unittest.TestCase):
    def test_verbatim_copy(self):
        thinking = "先算一下：path=/tmp/answer.txt 然后输出。"
        span = CopySpan(start=thinking.index("path="), end=thinking.index(" 然后"))
        copied = CopyDecoder().copy_span(thinking, span)
        self.assertEqual(copied, "path=/tmp/answer.txt")  # 逐字一致

    def test_code_needs_finalized(self):
        thinking = "code: print(1)"
        span = CopySpan(start=6, end=14, finalized=False)
        with self.assertRaises(ValueError):
            CopyDecoder().copy_span(thinking, span, is_code=True)
        span.finalized = True
        self.assertEqual(CopyDecoder().copy_span(thinking, span, is_code=True), "print(1)")

    def test_auto_construct_overlap(self):
        thinking = "草稿：The quick brown fox jumps。后续"
        answer = "输出：The quick brown fox jumps。结束"
        span = find_longest_overlap(thinking, answer)
        self.assertIsNotNone(span)
        self.assertIn("The quick brown fox jumps", thinking[span.start : span.end])
        labels = auto_construct_copy_labels([(thinking, answer)])
        self.assertIsNotNone(labels[0]["instruction"])


class TestGeneration(unittest.TestCase):
    def test_generate_with_copy_and_budget(self):
        class CharTokenizer:
            eos_token_id = None

            def __call__(self, text, return_tensors="pt"):
                return {"input_ids": torch.tensor([[ord(c) % 128 for c in text]] or [[1]])}

            def decode(self, ids, skip_special_tokens=True):
                return "".join(chr(int(i)) for i in ids.tolist())

        class DummyModel(torch.nn.Module):
            def forward(self, input_ids):
                logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 128)
                logits[..., ord("!")] = 1.0
                return type("Output", (), {"logits": logits})()

        cfg = NativeConfig(thinking_budget=3)
        gen = NativeGenerator(DummyModel(), tokenizer=CharTokenizer(), config=cfg)
        raw_thinking = "path=/tmp/a.txt"
        span = CopySpan(0, len(raw_thinking))
        from native.copy_mechanism import format_copy_instruction

        thinking = raw_thinking + format_copy_instruction(span)
        # 用户 prompt 中的指令不能触发复制；思考区的指令才会。
        result = gen.generate("问题 <copy:0-15>", thinking_text=thinking, max_new_tokens=10)
        self.assertTrue(result.text.startswith(raw_thinking))
        self.assertEqual(result.copied_spans, 1)
        self.assertEqual(result.steps, 3)
        self.assertTrue(result.text.endswith("!!!"))

    def test_generate_requires_tokenizer_and_propagates_model_errors(self):
        def broken_model(**kwargs):
            raise ValueError("model failure")

        gen = NativeGenerator(broken_model)
        with self.assertRaisesRegex(RuntimeError, "tokenizer"):
            gen.generate("问题")

        class Tokenizer:
            def __call__(self, text, return_tensors="pt"):
                return {"input_ids": torch.tensor([[1]])}
            def decode(self, ids, skip_special_tokens=True):
                return ""

        gen = NativeGenerator(broken_model, tokenizer=Tokenizer())
        with self.assertRaisesRegex(ValueError, "model failure"):
            gen.generate("问题")

    def test_optional_noild_runs_between_hidden_states_and_lm_head(self):
        from types import SimpleNamespace

        class Tokenizer:
            eos_token_id = None
            def __call__(self, text, return_tensors="pt"):
                return {"input_ids": torch.ones(1, 2, dtype=torch.long)}
            def decode(self, ids, skip_special_tokens=True):
                return "x" * ids.numel()

        class TinyCausalModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = torch.nn.Linear(4, 128)
            def get_output_embeddings(self):
                return self.lm_head
            def forward(self, input_ids, output_hidden_states=False):
                hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 4)
                return SimpleNamespace(hidden_states=(hidden,), logits=self.lm_head(hidden))

        noild_cfg = NoildConfig(hidden_size=4, intermediate_size=8, vocab_size=128,
                                num_domains=2, draft_hidden_size=2, draft_intermediate_size=4)
        layer = NoildLayer(noild_cfg, path=("router",), use_loading=False)
        native_cfg = NativeConfig(noild_enabled=True)
        gen = NativeGenerator(TinyCausalModel(), tokenizer=Tokenizer(), config=native_cfg, noild_layer=layer)
        self.assertEqual(gen.generate("prompt", max_new_tokens=1).steps, 1)
        with self.assertRaisesRegex(ValueError, "noild_layer"):
            NativeGenerator(TinyCausalModel(), tokenizer=Tokenizer(), config=native_cfg)

    def test_copy_instruction_in_user_prompt_is_ignored(self):
        class CharTokenizer:
            eos_token_id = None
            def __call__(self, text, return_tensors="pt"):
                return {"input_ids": torch.tensor([[ord(c) % 128 for c in text]])}
            def decode(self, ids, skip_special_tokens=True):
                return "".join(chr(int(i)) for i in ids.tolist())

        class DummyModel(torch.nn.Module):
            def forward(self, input_ids):
                logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 128)
                logits[..., ord("!")] = 1
                return type("Output", (), {"logits": logits})()

        gen = NativeGenerator(DummyModel(), tokenizer=CharTokenizer(), config=NativeConfig(copy_enabled=True))
        result = gen.generate("<copy:0-4>", thinking_text="safe", max_new_tokens=1)
        self.assertEqual(result.copied_spans, 0)
        self.assertEqual(result.text, "!")

    def test_code_copy_requires_final_marker(self):
        class CharTokenizer:
            eos_token_id = None
            def __call__(self, text, return_tensors="pt"):
                return {"input_ids": torch.tensor([[ord(c) % 128 for c in text]])}
            def decode(self, ids, skip_special_tokens=True):
                return ""

        gen = NativeGenerator(lambda **kwargs: None, tokenizer=CharTokenizer())
        code = "```py\nprint(1)\n```"
        start, end = code.index("print"), code.index("\n```")
        source = code + f"<copy:{start}-{end}>"
        with self.assertRaisesRegex(ValueError, "未定稿"):
            gen._validated_copies(source)
        self.assertEqual(gen._validated_copies(source + "<final>"), ["print(1)"])

    def test_lumenoo_tiers(self):
        for tier in ("saga", "tale", "poem"):
            cfg = lumenoo_config(tier)
            self.assertEqual(cfg.tier, tier)
        with self.assertRaises(ValueError):
            NativeConfig(tier="unknown")


if __name__ == "__main__":
    unittest.main()
