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
    load_balance_loss,
    set_trainable,
    upcycle_init,
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
