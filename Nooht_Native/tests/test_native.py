"""Native 外层优化层单元测试（unittest，无需 pytest；需 torch）。

运行：python -m unittest discover -s Nooht_Native/tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from native.backend import Backend, benchmark, check_correctness, detect_backend
from native.config import NativeConfig, lumenoo_config
from native.copy_mechanism import (
    CopyDecoder,
    CopySpan,
    auto_construct_copy_labels,
    find_longest_overlap,
)
from native.generation import NativeGenerator
from native.memory_compression import (
    Segment,
    SelectiveMemoryCompressor,
    split_segments,
)
from native.protected_zone import (
    InstructionAnchor,
    ProtectedZone,
    ThinkingMemoryBlock,
)


class TestBackend(unittest.TestCase):
    def test_detect(self):
        self.assertIsInstance(detect_backend(), Backend)

    def test_correctness_and_benchmark(self):
        report = check_correctness(Backend.CPU)
        self.assertTrue(report["forward_ok"])
        self.assertTrue(report["backward_ok"])
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
        def dummy_model(text):
            return torch.zeros(1, 10)

        cfg = NativeConfig(thinking_budget=3)
        gen = NativeGenerator(dummy_model, config=cfg)
        thinking = "path=/tmp/a.txt"
        span = CopySpan(0, len(thinking))
        from native.copy_mechanism import format_copy_instruction

        prompt = f"请输出 {format_copy_instruction(span)}"
        result = gen.generate(prompt, thinking_text=thinking, max_new_tokens=10)
        self.assertEqual(result.text, thinking)  # 复制逐字
        self.assertEqual(result.copied_spans, 1)
        self.assertLessEqual(result.steps, 3)  # 思考预算截断

    def test_lumenoo_tiers(self):
        for tier in ("saga", "tale", "poem"):
            cfg = lumenoo_config(tier)
            self.assertEqual(cfg.tier, tier)
        with self.assertRaises(ValueError):
            NativeConfig(tier="unknown")


if __name__ == "__main__":
    unittest.main()
