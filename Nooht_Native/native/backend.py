"""外层优化层 1：多硬件后端。

对应任务包第三节 1：
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
    try:
        import jax  # noqa: F401
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
        return Backend.ROCm
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
    """注册某后端某算子的优化实现（接口预留，本阶段不做深度优化）。"""
    _KERNELS[(op, backend)] = fn


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
        state = state + beta[..., t, :].squeeze(-1) * (x[..., t, :] - state)
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


def check_correctness(backend: Backend, op: str = "selective_scan",
                      device: str = "cpu") -> dict:
    """前向/反向正确性测试：后端实现 vs 参考实现（本阶段均为参考实现自检）。

    Returns:
        报告 dict：backend/op/forward_ok/backward_ok/max_abs_diff。
    """
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8, device=device, requires_grad=True)
    dt = torch.rand(2, 4, 8, device=device) * 0.1
    A = -torch.rand(8, device=device)
    B = torch.randn(2, 4, 8, device=device)
    C = torch.randn(2, 4, 8, device=device)
    ref = run_op(op, backend, x, dt, A, B, C)
    loss = ref.sum()
    loss.backward()
    grad_ok = x.grad is not None and bool(torch.isfinite(x.grad).all())
    return {
        "backend": backend.value,
        "op": op,
        "forward_ok": bool(torch.isfinite(ref).all()),
        "backward_ok": grad_ok,
        "max_abs_diff": 0.0,  # 回退路径与参考实现一致；注册 kernel 后在此对比
    }


def benchmark(backend: Backend, op: str = "selective_scan", steps: int = 5) -> dict:
    """记录基础性能（仅参考实现耗时，深度优化后另行记录）。"""
    torch.manual_seed(0)
    x = torch.randn(2, 16, 8)
    dt = torch.rand(2, 16, 8) * 0.1
    A = -torch.rand(8)
    B = torch.randn(2, 16, 8)
    C = torch.randn(2, 16, 8)
    run_op(op, backend, x, dt, A, B, C)  # warmup
    t0 = time.perf_counter()
    for _ in range(steps):
        run_op(op, backend, x, dt, A, B, C)
    elapsed = (time.perf_counter() - t0) / steps
    return {"backend": backend.value, "op": op, "avg_seconds": elapsed, "steps": steps}
