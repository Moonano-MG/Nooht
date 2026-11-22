"""Optional PEFT LoRA attachment and a small causal-LM fine-tuning loop.

The caller must choose and verify Jamba target module names for its checkpoint;
this module deliberately does not guess architecture-specific targets or load
weights automatically.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

from .config import NativeConfig


def attach_lora(
    model: nn.Module,
    config: NativeConfig,
    target_modules: Sequence[str],
) -> nn.Module:
    """Wrap ``model`` with a fresh trainable PEFT LoRA adapter.

    ``target_modules`` is required because exact Jamba module names vary by
    Transformers/checkpoint version. Only the adapter is trainable after PEFT
    wrapping. This function never downloads a checkpoint or starts training.
    """
    targets = [name.strip() for name in target_modules if name and name.strip()]
    if not targets:
        raise ValueError("必须显式提供并核实 LoRA target_modules。")
    if config.lora_rank <= 0 or config.lora_alpha <= 0:
        raise ValueError("lora_rank 和 lora_alpha 必须为正数。")
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError("LoRA 训练需额外安装 peft。") from exc

    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=targets,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    return get_peft_model(model, peft_config)


def train_causal_lm_steps(
    model: nn.Module,
    batches: Iterable[Mapping[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    max_steps: int | None = None,
    max_grad_norm: float | None = 1.0,
    gradient_accumulation_steps: int = 1,
    autocast_dtype: torch.dtype | None = None,
) -> list[dict[str, float]]:
    """Run caller-supplied causal-LM batches; suitable for LoRA-wrapped models.

    Each batch must contain ``input_ids`` and may include ``attention_mask``
    and ``labels``. When labels are omitted, input tokens are the targets and
    padding positions in ``attention_mask`` are ignored. ``max_steps`` counts
    input micro-batches; optimizer updates follow ``gradient_accumulation_steps``.
    The model remains caller-owned: this function does not load data, save
    checkpoints, or build the model.
    """
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps 必须为正整数。")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("模型没有可训练参数；请先通过 attach_lora 或显式解冻参数。")
    model.train()
    history: list[dict[str, float]] = []
    optimizer.zero_grad(set_to_none=True)
    pending = 0
    optimizer_steps = 0
    for step, batch in enumerate(batches):
        if max_steps is not None and step >= max_steps:
            break
        if "input_ids" not in batch:
            raise KeyError("每个 batch 必须包含 input_ids。")
        input_ids = batch["input_ids"]
        labels = batch.get("labels")
        if labels is None:
            labels = input_ids.clone()
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                labels = labels.masked_fill(attention_mask == 0, -100)
        if labels.shape != input_ids.shape:
            raise ValueError("labels 的形状必须与 input_ids 一致。")
        kwargs = {key: value for key, value in batch.items() if key not in ("labels",)}
        device_type = input_ids.device.type
        autocast_enabled = autocast_dtype is not None and device_type in ("cpu", "cuda")
        autocast_context = (
            torch.autocast(device_type=device_type, dtype=autocast_dtype)
            if autocast_enabled else nullcontext()
        )
        with autocast_context:
            output = model(**kwargs)
            logits = output.logits if hasattr(output, "logits") else output["logits"]
            if logits.ndim != 3 or logits.shape[:2] != labels.shape:
                raise ValueError("causal LM logits 必须为 (batch, sequence, vocab)，且与 labels 对齐。")
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"非有限训练 loss，step={step}。")
        (loss / gradient_accumulation_steps).backward()
        pending += 1
        if pending == gradient_accumulation_steps:
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            optimizer_steps += 1
        history.append({
            "step": float(step),
            "loss": float(loss.detach()),
            "optimizer_steps": float(optimizer_steps),
        })
    if pending:
        # Average a short final accumulation group rather than underweighting it.
        correction = gradient_accumulation_steps / pending
        for parameter in trainable:
            if parameter.grad is not None:
                parameter.grad.mul_(correction)
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if history:
            history[-1]["optimizer_steps"] = float(optimizer_steps + 1)
        optimizer_steps += 1
    return history
