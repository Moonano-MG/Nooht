"""Noild 单元测试（unittest，无需 pytest；需 torch）。

运行：python -m unittest discover -s Noild/tests -v
（在仓库根目录执行；Noild/ 需在 PYTHONPATH 上，测试内已处理）
"""

import math
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from noild import (
    NoildConfig,
    NoildLayer,
    get_cascade_path,
    draft_loss,
    future_token_targets,
    load_balance_loss,
    mtp_loss,
    set_trainable,
    speculative_greedy_decode,
    train_noild,
    upcycle_init,
    verify_draft,
)
from noild.experts import DraftExpert, ExpertMLP, MergeExpert
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


class TestIntegrationAdapters(unittest.TestCase):
    def test_jamba_replaces_only_sparse_moe_feed_forwards(self):
        from examples.jamba_integration import (
            install_noild_on_jamba_moe_layers,
            reset_noild_generation,
        )

        class JambaSparseMoeBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.router = torch.nn.Linear(64, 4)
                self.experts = torch.nn.ModuleList()

        class Decoder(torch.nn.Module):
            def __init__(self, feed_forward):
                super().__init__()
                self.feed_forward = feed_forward

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([
                    Decoder(JambaSparseMoeBlock()),
                    Decoder(torch.nn.Linear(64, 64)),
                ])

        class CausalLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Backbone()

        model = CausalLM()
        replaced = install_noild_on_jamba_moe_layers(model, small_config())
        self.assertEqual(replaced, [0])
        self.assertEqual(model.model.layers[0].feed_forward.__class__.__name__, "NoildJambaFeedForward")
        self.assertIsInstance(model.model.layers[1].feed_forward, torch.nn.Linear)
        model.model.layers[0].feed_forward.step = 7
        reset_noild_generation(model)
        self.assertEqual(model.model.layers[0].feed_forward.step, 0)

    def test_qwen_adapter_requires_and_replaces_explicit_paths(self):
        from examples.qwen_hybrid_integration import install_noild_on_qwen_mlp_modules

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = torch.nn.Linear(64, 64)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Layer()])

        model = Model()
        installed = install_noild_on_qwen_mlp_modules(
            model, small_config(), ["layers.0.mlp"]
        )
        self.assertEqual(installed, ["layers.0.mlp"])
        self.assertEqual(model.layers[0].mlp.__class__.__name__, "NoildQwenFeedForward")
        with self.assertRaisesRegex(ValueError, "module_paths"):
            install_noild_on_qwen_mlp_modules(model, small_config(), [])


class TestDraftInitialization(unittest.TestCase):
    def test_initial_draft_logits_are_uniform_and_head_trains(self):
        cfg = small_config()
        expert = DraftExpert(cfg)
        _, logits = expert(torch.randn(2, 3, cfg.hidden_size))
        self.assertTrue(torch.equal(logits, torch.zeros_like(logits)))
        labels = torch.randint(cfg.vocab_size, logits.shape[:-1])
        loss = draft_loss(logits, labels)
        self.assertAlmostEqual(float(loss), math.log(cfg.vocab_size), places=5)
        loss.backward()
        grad = expert.draft_head.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.0)


class TestRouter(unittest.TestCase):
    def test_nonconsecutive_steps_clear_cache_without_request_id(self):
        cfg = small_config(router_rejudge_interval=100)
        layer = NoildLayer(cfg, path=("router",), use_loading=False).eval()
        calls = 0
        original = layer.router.judge_user_input

        def counted(hidden):
            nonlocal calls
            calls += 1
            return original(hidden)

        layer.router.judge_user_input = counted
        hidden = torch.randn(2, 3, cfg.hidden_size)
        layer(hidden, step=5)
        next_request = layer(hidden, step=7)
        self.assertEqual(next_request.rejudge_reason, "first")
        self.assertEqual(calls, 2)
        continuation = layer(hidden, step=8)
        self.assertEqual(continuation.rejudge_reason, "cached")
        self.assertEqual(calls, 2)

    def test_rejudge_controller_constructs_under_meta_context(self):
        with torch.device("meta"):
            controller = RejudgeController(small_config())
        self.assertAlmostEqual(controller.max_entropy, torch.log(torch.tensor(4.0)).item(), places=6)

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
        # Tutti does not contain a general stage: do not retain dead experts.
        self.assertEqual(len(layer.generals), 0)
        self.assertFalse(any(name.startswith("generals.") for name, _ in layer.named_parameters()))
        x = torch.randn(2, 8, 64)
        out = layer(x, step=0)
        self.assertEqual(out.hidden.shape, x.shape)
        self.assertIsNotNone(out.draft_logits)
        self.assertEqual(out.draft_logits.shape[-2:], (4, 256))
        with self.assertRaisesRegex(RuntimeError, "未配置通用专家"):
            layer._run_generals(x, out.router, return_logits=False)

    def test_general_experts_receive_task_loss_gradients_when_in_path(self):
        cfg = small_config()
        layer = NoildLayer(cfg, path=("router", "general"), use_loading=False)
        self.assertEqual(len(layer.generals), cfg.num_general_experts)
        out = layer(
            torch.randn(2, 8, cfg.hidden_size),
            domain_hint=torch.tensor([0, 1]),
        )
        (out.hidden.square().mean() + out.mtp_logits.square().mean()).backward()
        gradients = [p.grad for p in layer.generals.parameters()]
        self.assertTrue(gradients)
        self.assertTrue(all(g is not None for g in gradients))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.0)

    def test_offload_device_is_inference_only(self):
        layer = NoildLayer(small_config(), path=("router",), offload_device="cpu")
        with self.assertRaisesRegex(RuntimeError, "仅支持 no_grad"):
            layer(torch.randn(1, 2, 64))

    def test_soleinoo_path_forward(self):
        layer = NoildLayer(small_config(), path=get_cascade_path("soleinoo", "tutti"))
        x = torch.randn(2, 8, 64)
        out = layer(x, step=0)
        self.assertEqual(out.hidden.shape, x.shape)
        self.assertIsNotNone(out.mtp_logits)

    def test_flash_path_no_merge(self):
        layer = NoildLayer(small_config(), path=get_cascade_path("veronoo", "duet"))
        self.assertEqual(len(layer.verticals), 0)
        self.assertIsNone(layer.merge)
        self.assertFalse(any(eid.startswith("vertical:") for eid in layer.loading.loaded))
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

    def test_router_cache_rejects_changed_batch_and_request(self):
        cfg = small_config(router_rejudge_interval=100)
        layer = NoildLayer(cfg, path=("router",), use_loading=False).eval()
        calls = 0
        original = layer.router.judge_user_input

        def counted(hidden):
            nonlocal calls
            calls += 1
            return original(hidden)

        layer.router.judge_user_input = counted
        first = layer(torch.randn(2, 3, cfg.hidden_size), step=4, request_id="request-a")
        self.assertEqual(first.rejudge_reason, "first")
        changed_batch = layer(
            torch.randn(3, 3, cfg.hidden_size), step=5, request_id="request-a"
        )
        self.assertEqual(changed_batch.rejudge_reason, "cache_mismatch")
        self.assertEqual(changed_batch.hidden.shape, (3, 3, cfg.hidden_size))
        self.assertEqual(calls, 2)
        changed_request = layer(
            torch.randn(3, 3, cfg.hidden_size), step=6, request_id="request-b"
        )
        self.assertEqual(changed_request.rejudge_reason, "first")
        self.assertEqual(calls, 3)

    def test_training_route_is_not_reused_by_eval(self):
        cfg = small_config()
        layer = NoildLayer(cfg, path=("router",), use_loading=False)
        layer(torch.randn(2, 3, cfg.hidden_size), step=1)
        self.assertIsNone(layer._cached_router)
        layer.eval()
        out = layer(torch.randn(2, 3, cfg.hidden_size), step=10)
        self.assertEqual(out.rejudge_reason, "first")

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
        self.assertEqual(calls, 13)  # 未校准时仅按 interval 重判，不由随机熵触发
        layer.mark_router_uncertainty_calibrated()
        out = layer(x, step=50)
        self.assertEqual(out.rejudge_reason, "uncertainty")
        self.assertEqual(calls, 14)  # 计数为累计值：此前 13 次 interval 重判 + 校准后 1 次不确定性重判

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

    def test_future_token_targets_align_and_mask_sequence_tail(self):
        labels = torch.tensor([[10, 11, 12, 13]])
        targets = future_token_targets(labels, num_tokens=3)
        self.assertEqual(targets.tolist(), [[[10, 11, 12], [11, 12, 13], [12, 13, -100], [13, -100, -100]]])

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
        # The extra target bonus token is not a draft and must not inflate acceptance.
        self.assertEqual(metrics["accepted"], metrics["proposed"])
        self.assertEqual(metrics["proposed"], 4)

    def test_speculative_probe_reports_partial_acceptance_and_correction(self):
        def future_draft(ids):
            logits = torch.zeros(1, 2, 16)
            logits[0, 0, 2] = 1.0
            logits[0, 1, 9] = 1.0  # deliberately diverges at the second proposal
            return logits

        def target(ids):
            logits = torch.zeros(1, ids.shape[1], 16)
            for position in range(ids.shape[1]):
                logits[0, position, position + 2] = 1.0
            return logits

        output, metrics = speculative_greedy_decode(
            torch.tensor([[1]]), future_draft, target, 2,
            draft_num_tokens=2, draft_output_mode="future",
        )
        self.assertEqual(output.tolist(), [[1, 2, 3]])
        self.assertEqual(metrics["proposed"], 2)
        self.assertEqual(metrics["accepted"], 1)
        self.assertEqual(metrics["acceptance_rate"], 0.5)

    def test_speculative_decode_stops_at_accepted_eos_inside_draft(self):
        def future_draft(ids):
            logits = torch.zeros(1, 3, 16)
            for index, token in enumerate((2, 3, 4)):
                logits[0, index, token] = 1.0
            return logits

        def target(ids):
            logits = torch.zeros(1, ids.shape[1], 16)
            for position in range(ids.shape[1]):
                logits[0, position, position + 2] = 1.0
            return logits

        output, metrics = speculative_greedy_decode(
            torch.tensor([[1]]), future_draft, target, 3,
            draft_num_tokens=3, draft_output_mode="future", eos_token_id=3,
        )
        self.assertEqual(output.tolist(), [[1, 2, 3]])
        self.assertEqual(metrics["accepted"], 2)
        self.assertEqual(metrics["proposed"], 3)

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
        # Targets can be derived from next-token labels; no manually generated random targets needed.
        auto_history = train_noild(
            layer, head,
            [{k: v for k, v in batch.items() if k not in ("draft_target_ids", "mtp_target_ids")}],
            optimizer,
        )
        self.assertIn("draft_loss", auto_history[0])
        self.assertIn("mtp_loss", auto_history[0])

    def test_train_noild_errors_when_enabled_auxiliary_logits_are_missing(self):
        cfg = small_config()
        batch = {
            "hidden": torch.randn(2, 3, cfg.hidden_size),
            "target_ids": torch.randint(cfg.vocab_size, (2, 3)),
        }
        for path, field, message in (
            (("router", "draft"), "draft_logits", "draft_logits"),
            (("router", "general"), "mtp_logits", "mtp_logits"),
        ):
            layer = NoildLayer(cfg, path=path, use_loading=False)
            original_forward = layer.forward

            def without_aux(*args, _forward=original_forward, _field=field, **kwargs):
                output = _forward(*args, **kwargs)
                setattr(output, _field, None)
                return output

            layer.forward = without_aux
            head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size)
            optimizer = torch.optim.AdamW(list(layer.parameters()) + list(head.parameters()), lr=1e-3)
            with self.assertRaisesRegex(RuntimeError, message):
                train_noild(layer, head, [batch], optimizer)

    def test_tutti_records_mtp_not_applicable_and_rejects_stray_mtp_targets(self):
        cfg = small_config()
        layer = NoildLayer(cfg, path=get_cascade_path("veronoo", "tutti"), use_loading=False)
        head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size)
        optimizer = torch.optim.AdamW(list(layer.parameters()) + list(head.parameters()), lr=1e-3)
        batch = {
            "hidden": torch.randn(2, 4, cfg.hidden_size),
            "target_ids": torch.randint(cfg.vocab_size, (2, 4)),
        }
        history = train_noild(layer, head, [batch], optimizer)
        self.assertEqual(history[0]["mtp_status"], "not_applicable:no_general_stage")
        self.assertNotIn("mtp_loss", history[0])
        batch["mtp_target_ids"] = torch.randint(cfg.vocab_size, (2, 4, cfg.mtp_num_tokens))
        with self.assertRaisesRegex(ValueError, "没有 general/MTP"):
            train_noild(layer, head, [batch], optimizer)

    def test_merge_both_sources(self):
        merge = MergeExpert(small_config())
        x = torch.randn(2, 8, 64)
        for src in ("draft", "general"):
            y = merge(x, x, source=src, step=0)
            self.assertEqual(y.shape, x.shape)
        with self.assertRaises(ValueError):
            merge(x, x, source="unknown")


class TestLoadingAndTraining(unittest.TestCase):
    def test_async_expert_callbacks_run_off_thread_and_reload_before_use(self):
        main_thread = threading.get_ident()
        calls = []
        def preload(expert_id):
            calls.append(("load", expert_id, threading.get_ident()))
        def unload(expert_id):
            calls.append(("unload", expert_id, threading.get_ident()))

        mgr = ExpertHeatManager(
            num_vertical=2, num_general=0, min_loaded=1, preload_delay=0,
            preload_fn=preload, unload_fn=unload,
        )
        cold_expert = "vertical:1"
        mgr.heat[cold_expert] = 0.0
        mgr.last_used_step[cold_expert] = -100
        mgr.step(100)
        self.assertNotIn(cold_expert, mgr.loaded)
        mgr.register_route([1], 101)
        mgr.ensure_loaded(cold_expert)
        self.assertIn(cold_expert, mgr.loaded)
        mgr.close(wait=True)
        self.assertEqual([kind for kind, eid, _ in calls if eid == cold_expert], ["unload", "load"])
        self.assertTrue(all(thread_id != main_thread for _, _, thread_id in calls))

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
