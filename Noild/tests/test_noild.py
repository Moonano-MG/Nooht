"""Noild 单元测试（unittest，无需 pytest；需 torch）。

运行：python -m unittest discover -s Noild/tests -v
（在仓库根目录执行；Noild/ 需在 PYTHONPATH 上，测试内已处理）
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from noild import (
    NoildConfig,
    NoildLayer,
    get_cascade_path,
    draft_loss,
    load_balance_loss,
    mtp_loss,
    set_trainable,
    speculative_greedy_decode,
    train_noild,
    upcycle_init,
    verify_draft,
)
from noild.experts import ExpertMLP, MergeExpert
from noild.loading import ExpertHeatManager
from noild.router import DomainRouter, RejudgeController


def small_config(**kw):
    base = dict(
        hidden_size=64,
        intermediate_size=128,
        vocab_size=256,
        num_domains=4,
        num_general_experts=2,
        draft_hidden_size=32,
        draft_intermediate_size=64,
    )
    base.update(kw)
    return NoildConfig(**base)


class TestConfig(unittest.TestCase):
    def test_paths(self):
        self.assertEqual(
            get_cascade_path("veronoo", "tutti"),
            ("router", "draft", "vertical", "merge"),
        )
        self.assertEqual(
            get_cascade_path("soleinoo", "tutti"),
            ("router", "draft", "general", "vertical", "merge", "general"),
        )
        self.assertEqual(get_cascade_path("lumenoo", "saga"), ())

    def test_no_early_exit(self):
        with self.assertRaises(ValueError):
            NoildConfig(use_early_exit=True)

    def test_merge_window_growth(self):
        cfg = small_config(merge_window_start=2, merge_window_max=10,
                           merge_window_growth_steps=10)
        self.assertEqual(cfg.merge_window_at_step(0), 2)
        self.assertEqual(cfg.merge_window_at_step(1000), 10)


class TestRouter(unittest.TestCase):
    def test_only_domain_no_tier(self):
        router = DomainRouter(small_config())
        out = router(torch.randn(3, 64))
        self.assertFalse(hasattr(out, "tier"))
        self.assertEqual(out.domain_ids.shape, (3, 2))  # top_vertical=2

    def test_rejudge_interval_and_early(self):
        ctrl = RejudgeController(small_config(router_rejudge_interval=4))
        ctrl.mark_rejudged(0)
        should, _ = ctrl.should_rejudge(2)
        self.assertFalse(should)
        should, reason = ctrl.should_rejudge(4)
        self.assertTrue(should)
        self.assertEqual(reason, "interval")
        # 不确定度突升提前重判
        should, reason = ctrl.should_rejudge(1, entropy=99.0)
        self.assertTrue(should)
        self.assertEqual(reason, "uncertainty")
        # 负载大时扩大 N
        ctrl.update_load(10.0)
        self.assertGreater(ctrl.current_interval, 4)


class TestLayer(unittest.TestCase):
    def test_veronoo_path_forward(self):
        layer = NoildLayer(small_config(), path=get_cascade_path("veronoo", "tutti"))
        x = torch.randn(2, 8, 64)
        out = layer(x, step=0)
        self.assertEqual(out.hidden.shape, x.shape)
        self.assertIsNotNone(out.draft_logits)
        self.assertEqual(out.draft_logits.shape[-2:], (4, 256))

    def test_soleinoo_path_forward(self):
        layer = NoildLayer(small_config(), path=get_cascade_path("soleinoo", "tutti"))
        x = torch.randn(2, 8, 64)
        out = layer(x, step=0)
        self.assertEqual(out.hidden.shape, x.shape)
        self.assertIsNotNone(out.mtp_logits)

    def test_flash_path_no_merge(self):
        layer = NoildLayer(small_config(), path=get_cascade_path("veronoo", "duet"))
        out = layer(torch.randn(2, 8, 64))
        self.assertEqual(out.path_taken, ("router", "draft", "general"))

    def test_inference_skips_unused_auxiliary_logits(self):
        cfg = small_config()
        layer = NoildLayer(cfg, path=("router", "draft", "general"), use_loading=False).eval()
        out = layer(torch.randn(1, 4, cfg.hidden_size))
        self.assertIsNone(out.draft_logits)
        self.assertIsNone(out.mtp_logits)
        train_out = layer(torch.randn(1, 4, cfg.hidden_size), return_aux_logits=True)
        self.assertIsNotNone(train_out.draft_logits)
        self.assertIsNotNone(train_out.mtp_logits)

    def test_router_rejudge_cache_and_general_determinism(self):
        cfg = small_config(router_rejudge_interval=2)
        layer = NoildLayer(cfg, path=("router", "general"), use_loading=False).eval()
        with torch.no_grad():
            layer.router.weight.weight.zero_()
            layer.router.weight.weight[0].fill_(2.0)
        calls = 0
        original = layer.router.judge_user_input

        def counted(hidden):
            nonlocal calls
            calls += 1
            return original(hidden)

        layer.router.judge_user_input = counted
        x = torch.ones(1, 4, cfg.hidden_size)
        first = layer(x, step=0)
        second = layer(x, step=1)
        self.assertEqual(calls, 1)
        self.assertEqual(second.rejudge_reason, "cached")
        self.assertTrue(torch.allclose(first.hidden, second.hidden))
        third = layer(x, step=2)
        self.assertEqual(calls, 2)
        self.assertEqual(third.rejudge_reason, "interval")
        new_request = layer(x, step=0)
        self.assertEqual(calls, 3)
        self.assertEqual(new_request.rejudge_reason, "first")

    def test_uncertain_untrained_router_is_throttled(self):
        cfg = small_config(router_rejudge_interval=4)
        layer = NoildLayer(cfg, path=("router",), use_loading=False).eval()
        with torch.no_grad():
            layer.router.weight.weight.zero_()  # exactly uniform, max normalized entropy
        calls = 0
        original = layer.router.judge_user_input
        def counted(hidden):
            nonlocal calls
            calls += 1
            return original(hidden)
        layer.router.judge_user_input = counted
        x = torch.ones(1, 1, cfg.hidden_size)
        for step in range(50):
            layer(x, step=step)
        self.assertLess(calls, 20)  # 不会因高熵每个 token 重判
        self.assertGreater(calls, 1)

    def test_repeated_general_stages_select_distinct_experts(self):
        layer = NoildLayer(small_config(num_general_experts=2), path=("router", "general", "general"), use_loading=False)
        self.assertIsNot(layer._general_for_domain(0, 0), layer._general_for_domain(0, 1))
        out = layer(torch.randn(1, 3, 64))
        self.assertIsNotNone(out.mtp_logits)

    def test_merge_window_tail_is_primary_passthrough(self):
        cfg = small_config(merge_window_start=2, merge_window_max=2)
        merge = MergeExpert(cfg)
        primary = torch.randn(1, 6, cfg.hidden_size)
        vertical_a = torch.randn_like(primary)
        vertical_b = vertical_a.clone()
        vertical_b[:, 2:, :] += 20
        out_a = merge(primary, vertical_a, step=0)
        out_b = merge(primary, vertical_b, step=0)
        self.assertTrue(torch.equal(out_a[:, 2:, :], primary[:, 2:, :]))
        self.assertTrue(torch.equal(out_a[:, 2:, :], out_b[:, 2:, :]))

    def test_cascade_signal_and_auxiliary_head_gradients(self):
        cfg = small_config()
        layer = NoildLayer(cfg, path=("router", "draft", "general", "vertical", "merge"), use_loading=False)
        x = torch.randn(2, 8, cfg.hidden_size, requires_grad=True)
        out = layer(x)
        ratio = float(out.hidden.detach().square().mean().sqrt() / x.detach().square().mean().sqrt())
        self.assertGreater(ratio, 1e-3)
        draft_targets = torch.randint(cfg.vocab_size, (2, 8, cfg.draft_num_tokens))
        mtp_targets = torch.randint(cfg.vocab_size, (2, 8, cfg.mtp_num_tokens))
        (draft_loss(out.draft_logits, draft_targets) + mtp_loss(out.mtp_logits, mtp_targets)).backward()
        self.assertGreater(float(layer.draft.draft_head.weight.grad.abs().sum()), 0)
        mtp_grad = sum(
            float(expert.mtp_head.proj.weight.grad.abs().sum())
            for expert in layer.generals
            if expert.mtp_head.proj.weight.grad is not None
        )
        self.assertGreater(mtp_grad, 0)

    def test_speculative_prefix_verification(self):
        accepted, rate = verify_draft(torch.tensor([4, 7, 3, 8]), torch.tensor([4, 7, 5, 8]))
        self.assertEqual(accepted, 2)
        self.assertEqual(rate, 0.5)

    def test_speculative_decode_accepts_verified_tokens(self):
        def logits_for(ids):
            return torch.zeros(*ids.shape, 16)

        def draft_fn(ids):
            positions = torch.arange(ids.shape[1] + 1, ids.shape[1] + 4).view(1, 3)
            logits = logits_for(positions)
            logits.scatter_(2, positions.unsqueeze(-1), 1.0)
            return logits

        def target_fn(ids):
            positions = torch.arange(2, ids.shape[1] + 2).view(1, -1)
            logits = logits_for(ids)
            logits.scatter_(2, positions.unsqueeze(-1), 1.0)
            return logits

        output, metrics = speculative_greedy_decode(
            torch.tensor([[1]]), draft_fn, target_fn, 5,
            draft_num_tokens=3, draft_output_mode="future",
        )
        self.assertEqual(output.tolist(), [[1, 2, 3, 4, 5, 6]])
        self.assertEqual(metrics["acceptance_rate"], 1.0)

    def test_speculative_causal_output_uses_last_position_only(self):
        def causal_fn(ids):
            logits = torch.zeros(1, ids.shape[1], 16)
            for position in range(ids.shape[1]):
                logits[:, position, position + 2] = 10
            return logits

        output, metrics = speculative_greedy_decode(
            torch.tensor([[1]]), causal_fn, causal_fn, 4,
            draft_num_tokens=3, draft_output_mode="causal",
        )
        self.assertEqual(output.tolist(), [[1, 2, 3, 4, 5]])
        self.assertEqual(metrics["acceptance_rate"], 1.0)

    def test_speculative_accepts_noild_four_dimensional_draft_logits(self):
        def noild_style_draft(ids):
            seq = ids.shape[1]
            # draft logits: (batch, sequence, K, vocab), future tokens are seq+1 .. seq+2
            logits = torch.zeros(1, seq, 2, 32)
            logits[:, -1, 0, seq + 1] = 1
            logits[:, -1, 1, seq + 2] = 1
            return logits

        def target(ids):
            logits = torch.zeros(1, ids.shape[1], 32)
            for pos in range(ids.shape[1]):
                logits[:, pos, pos + 2] = 1
            return logits

        output, metrics = speculative_greedy_decode(
            torch.tensor([[1]]), noild_style_draft, target, 3,
            draft_num_tokens=2, draft_output_mode="future",
        )
        self.assertEqual(output.tolist(), [[1, 2, 3, 4]])
        self.assertEqual(metrics["acceptance_rate"], 1.0)

    def test_noild_layer_constructs_on_meta_device(self):
        with torch.device("meta"):
            layer = NoildLayer(small_config())
        self.assertEqual(layer.router.weight.weight.device.type, "meta")

    def test_train_noild_requires_and_uses_auxiliary_targets(self):
        cfg = small_config()
        layer = NoildLayer(cfg, path=("router", "draft", "general", "vertical", "merge"), use_loading=False)
        head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size)
        optimizer = torch.optim.AdamW(list(layer.parameters()) + list(head.parameters()), lr=1e-3)
        batch_size, seq = 2, 4
        batch = {
            "hidden": torch.randn(batch_size, seq, cfg.hidden_size),
            "target_ids": torch.randint(cfg.vocab_size, (batch_size, seq)),
            "draft_target_ids": torch.randint(cfg.vocab_size, (batch_size, seq, cfg.draft_num_tokens)),
            "mtp_target_ids": torch.randint(cfg.vocab_size, (batch_size, seq, cfg.mtp_num_tokens)),
        }
        history = train_noild(layer, head, [batch], optimizer)
        self.assertIn("draft_loss", history[0])
        self.assertIn("mtp_loss", history[0])
        with self.assertRaisesRegex(KeyError, "draft_target_ids"):
            train_noild(layer, head, [{k: v for k, v in batch.items() if k != "draft_target_ids"}], optimizer)

    def test_merge_both_sources(self):
        merge = MergeExpert(small_config())
        x = torch.randn(2, 8, 64)
        for src in ("draft", "general"):
            y = merge(x, x, source=src, step=0)
            self.assertEqual(y.shape, x.shape)
        with self.assertRaises(ValueError):
            merge(x, x, source="unknown")


class TestLoadingAndTraining(unittest.TestCase):
    def test_cold_start_and_unload(self):
        mgr = ExpertHeatManager(num_vertical=4, num_general=2)
        self.assertEqual(len(mgr.loaded), 6)  # 冷启动全量
        mgr.register_route([1], step=0)
        self.assertIn("vertical:1", mgr.loaded)
        # 长期不用的专家会被卸载（热度衰减 + 冷却期过后）
        for s in range(1, 200):
            mgr.step(s)
        self.assertIn("vertical:1", mgr.loaded)  # 热专家保留

    def test_load_balance_loss(self):
        logits = torch.randn(32, 4)
        loss = load_balance_loss(logits, 4)
        self.assertTrue(torch.isfinite(loss))

    def test_staged_trainable(self):
        layer = NoildLayer(small_config())
        names = set_trainable(layer, 1)
        self.assertIn("router", names)
        router_trainable = all(p.requires_grad for p in layer.router.parameters())
        self.assertTrue(router_trainable)

    def test_upcycle_init(self):
        layer = NoildLayer(small_config())
        src = ExpertMLP(64, 128)
        upcycle_init(layer.verticals, src)
        for v in layer.verticals:
            self.assertTrue(torch.allclose(v.mlp.gate_proj.weight, src.gate_proj.weight))


if __name__ == "__main__":
    unittest.main()
