"""外层优化层 1：后端检测与可插拔算子注册。

当前文件提供硬件检测、纯 PyTorch 参考实现和 kernel 注册接口；仓库默认没有
CUDA/ROCm/XLA/oneAPI 优化 kernel。未注册实现时所有后端均显式回退参考实现，
因此此模块本身不构成多后端性能或算子覆盖验收。

目标规格：
- 运行时检测 CUDA、ROCm、XLA、oneAPI，替换底层算子实现；
- 需同时覆盖 Mamba selective scan（Jamba 底座）与 Gated DeltaNet 算子
  （Evolution 底座）；后者依赖第三方线性注意力算子库，
  非 CUDA 后端可能只有慢速回退实现，需确认；
- 优先级 CUDA > ROCm > XLA、oneAPI；
- 每个后端通过前向/反向正确性测试并记录基础性能；
- 深度 kernel 优化只留接口，不在本阶段。
"""

from __future__ import annotations

import enum
import time
from typing import Callable

import torch


class Backend(enum.Enum):
    CUDA = "cuda"
    ROCM = "rocm"
    XLA = "xla"
    ONEAPI = "oneapi"
    CPU = "cpu"  # 回退：纯 PyTorch 参考实现（慢速但可跑通）


_PRIORITY = [Backend.CUDA, Backend.ROCM, Backend.XLA, Backend.ONEAPI, Backend.CPU]


def _has_cuda() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _has_rocm() -> bool:
    # ROCm 版 PyTorch：torch.version.hip 非空
    try:
        return getattr(torch.version, "hip", None) is not None and _has_cuda()
    except Exception:
        return False


def _has_xla() -> bool:
    """检测 PyTorch/XLA（安装 JAX 本身不代表 PyTorch 能用 XLA）。"""
    try:
        import torch_xla  # noqa: F401
        return True
    except Exception:
        return False


def _has_oneapi() -> bool:
    try:
        import intel_extension_for_pytorch  # noqa: F401
        return True
    except Exception:
        return False


def detect_backend() -> Backend:
    """按优先级 CUDA > ROCm > XLA > oneAPI 检测，缺省 CPU 回退。"""
    if _has_rocm():
        return Backend.ROCM
    if _has_cuda():
        return Backend.CUDA
    if _has_xla():
        return Backend.XLA
    if _has_oneapi():
        return Backend.ONEAPI
    return Backend.CPU


def resolve_backend(name: str | Backend) -> Backend:
    if isinstance(name, Backend):
        return name
    key = str(name).lower()
    if key == "auto":
        return detect_backend()
    for b in _PRIORITY:
        if b.value == key:
            return b
    raise ValueError(f"未知 backend: {name!r}，可选: auto/cuda/rocm/xla/oneapi/cpu")


# -- 算子注册表：深度 kernel 优化只留接口（任务包第三节 1） --
_KERNELS: dict[tuple[str, Backend], Callable] = {}


def register_kernel(op: str, backend: Backend, fn: Callable) -> None:
    """注册某后端某算子的实现。注册前先验证实现的前向/反向正确性。"""
    if op not in ("selective_scan", "gated_delta"):
        raise KeyError(f"未知算子: {op!r}")
    _KERNELS[(op, backend)] = fn


def backend_status(backend: str | Backend) -> dict:
    """报告哪些算子确有注册实现，避免把后端检测误报为 kernel 支持。"""
    selected = resolve_backend(backend)
    ops = sorted(_OP_REFERENCES)
    implementations = {
        op: "registered" if (op, selected) in _KERNELS else "reference_fallback"
        for op in ops
    }
    return {
        "backend": selected.value,
        "detected": selected is Backend.CPU or selected is detect_backend(),
        "operators": implementations,
        "all_optimized": all(value == "registered" for value in implementations.values()),
    }


# -- 纯 PyTorch 参考实现（回退路径，保证跑通） --

def selective_scan_reference(x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor,
                             B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Mamba selective scan 的朴素顺序实现（仅正确性参考，不优化）。

    x: (..., L, D); dt/A/B/C 形状按广播兼容。返回与 x 同形状。
    """
    L = x.shape[-2]
    h = torch.zeros_like(x[..., 0, :])
    outs = []
    for t in range(L):
        # 离散化：h = exp(dt*A) * h + dt * B * x_t（简化标量形式）
        decay = torch.exp(dt[..., t, :] * A)
        h = decay * h + dt[..., t, :] * B[..., t, :] * x[..., t, :]
        outs.append(h * C[..., t, :])
    return torch.stack(outs, dim=-2)


def gated_delta_reference(x: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Gated DeltaNet 的朴素顺序实现（仅正确性参考）。"""
    L = x.shape[-2]
    state = torch.zeros_like(x[..., 0, :])
    outs = []
    for t in range(L):
        # delta 规则简化：state = state + beta * (x_t - state)
        state = state + beta[..., t, :] * (x[..., t, :] - state)
        outs.append(state)
    return torch.stack(outs, dim=-2)


_OP_REFERENCES = {
    "selective_scan": selective_scan_reference,
    "gated_delta": gated_delta_reference,
}


def run_op(op: str, backend: Backend, *args, **kwargs) -> torch.Tensor:
    """按后端分发算子：有注册 kernel 用之，否则回退参考实现。"""
    fn = _KERNELS.get((op, backend))
    if fn is not None:
        return fn(*args, **kwargs)
    if op not in _OP_REFERENCES:
        raise KeyError(f"未知算子: {op!r}，可选: {sorted(_OP_REFERENCES)}")
    return _OP_REFERENCES[op](*args, **kwargs)


def _sample_inputs(op: str, device: str, sequence: int = 4):
    x = torch.randn(2, sequence, 8, device=device)
    if op == "selective_scan":
        return (x, torch.rand_like(x) * 0.1, -torch.rand(8, device=device),
                torch.randn_like(x), torch.randn_like(x))
    if op == "gated_delta":
        return (x, torch.rand(2, sequence, 1, device=device))
    raise KeyError(f"未知算子: {op!r}")


def check_correctness(backend: Backend, op: str = "selective_scan",
                      device: str = "cpu") -> dict:
    """比较注册后端 kernel 与纯 PyTorch 参考实现的前向/反向结果。"""
    torch.manual_seed(0)
    inputs = _sample_inputs(op, device)
    ref_inputs = tuple(t.detach().clone().requires_grad_(i == 0) for i, t in enumerate(inputs))
    test_inputs = tuple(t.detach().clone().requires_grad_(i == 0) for i, t in enumerate(inputs))
    ref = _OP_REFERENCES[op](*ref_inputs)
    actual = run_op(op, backend, *test_inputs)
    max_abs_diff = float((actual.detach() - ref.detach()).abs().max())
    ref.sum().backward()
    actual.sum().backward()
    ref_grad, test_grad = ref_inputs[0].grad, test_inputs[0].grad
    backward_ok = (
        ref_grad is not None and test_grad is not None
        and bool(torch.isfinite(ref_grad).all()) and bool(torch.isfinite(test_grad).all())
        and bool(torch.allclose(ref_grad, test_grad, atol=1e-5, rtol=1e-4))
    )
    return {
        "backend": backend.value,
        "op": op,
        "forward_ok": bool(torch.isfinite(actual).all()) and bool(torch.allclose(actual, ref, atol=1e-5, rtol=1e-4)),
        "backward_ok": backward_ok,
        "max_abs_diff": max_abs_diff,
    }


def benchmark(backend: Backend, op: str = "selective_scan", steps: int = 5) -> dict:
    """记录指定后端算子的基础耗时。"""
    if steps <= 0:
        raise ValueError("steps 必须为正整数")
    torch.manual_seed(0)
    device = "cuda" if backend in (Backend.CUDA, Backend.ROCM) and torch.cuda.is_available() else "cpu"
    args = _sample_inputs(op, device, sequence=16)
    run_op(op, backend, *args)  # warmup
    t0 = time.perf_counter()
    for _ in range(steps):
        run_op(op, backend, *args)
    elapsed = (time.perf_counter() - t0) / steps
    return {"backend": backend.value, "op": op, "avg_seconds": elapsed, "steps": steps}
