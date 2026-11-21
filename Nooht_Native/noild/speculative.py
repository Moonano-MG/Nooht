"""草稿校验（猜后验）：预生成专家草稿 token 的接受判定。

提供两个层次：
- ``verify_draft``：检查一批草稿 token 的贪心匹配前缀；
- ``speculative_greedy_decode``：通用单序列草稿/目标模型循环，验证前缀、
  接受已匹配 token，并在首次不匹配处采用目标模型 token。

此处不实现 KV 缓存与概率式推测采样；调用方可提供带缓存的 draft_fn/target_fn，
但本函数本身使用完整 input_ids 保证语义清楚。
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def verify_draft(draft_ids: torch.Tensor, target_ids: torch.Tensor) -> tuple[int, float]:
    """贪心前缀校验。输入为形状 ``(K,)`` 的草稿与目标 token id。"""
    if draft_ids.shape != target_ids.shape:
        raise ValueError(
            f"draft 与 target 形状必须一致，得到 {tuple(draft_ids.shape)} vs {tuple(target_ids.shape)}"
        )
    match = (draft_ids == target_ids).tolist()
    accepted = 0
    for ok in match:
        if not ok:
            break
        accepted += 1
    return accepted, accepted / max(len(match), 1)


def draft_acceptance_rate(
    draft_ids_batch: torch.Tensor, target_ids_batch: torch.Tensor
) -> float:
    """批量接受率：(B, K) 两批 token id 的平均连续前缀接受率。"""
    if draft_ids_batch.shape != target_ids_batch.shape:
        raise ValueError("draft 与 target 批量形状必须一致。")
    rates = [verify_draft(d, t)[1] for d, t in zip(draft_ids_batch, target_ids_batch)]
    return sum(rates) / max(len(rates), 1)


def _extract_logits(output) -> torch.Tensor:
    if hasattr(output, "logits"):
        output = output.logits
    elif isinstance(output, dict) and "logits" in output:
        output = output["logits"]
    elif isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor) or output.ndim != 3:
        raise TypeError("draft_fn/target_fn 必须返回 (batch, sequence, vocab) logits")
    return output


def speculative_greedy_decode(
    input_ids: torch.Tensor,
    draft_fn: Callable[[torch.Tensor], torch.Tensor],
    target_fn: Callable[[torch.Tensor], torch.Tensor],
    max_new_tokens: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """单序列推测贪心解码（无 KV cache）。

    两个函数都接收完整 ``input_ids``，返回模型 logits。``draft_fn`` 每次从
    当前前缀预测 K 个 token；``target_fn`` 对拼接后的前缀与草稿做一次前向，
    目标 logits 逐 token 校验草稿。首次不匹配时接受此前前缀并追加目标 token。

    返回扩展后的 input_ids 及 proposed/accepted/acceptance_rate 指标。
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("当前基础实现只支持 batch size = 1。")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens 不得为负。")
    result = input_ids
    proposed = accepted_total = 0
    while result.shape[1] - input_ids.shape[1] < max_new_tokens:
        draft_logits = _extract_logits(draft_fn(result))
        draft_ids = draft_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        # draft_fn 可返回一次预测 K 个位置；若仅返回一个位置也合法。
        if draft_logits.shape[1] > 1:
            draft_ids = draft_logits.argmax(dim=-1)
        remaining = max_new_tokens - (result.shape[1] - input_ids.shape[1])
        draft_ids = draft_ids[:, :remaining]
        if draft_ids.shape[1] == 0:
            break
        target_input = torch.cat((result, draft_ids), dim=1)
        target_logits = _extract_logits(target_fn(target_input))
        start = result.shape[1] - 1
        target_ids = target_logits[:, start : start + draft_ids.shape[1], :].argmax(dim=-1)
        count, _ = verify_draft(draft_ids[0], target_ids[0])
        proposed += draft_ids.shape[1]
        accepted_total += count
        if count == draft_ids.shape[1]:
            result = torch.cat((result, draft_ids), dim=1)
        else:
            correction = target_ids[:, count : count + 1]
            result = torch.cat((result, draft_ids[:, :count], correction), dim=1)
    rate = accepted_total / max(proposed, 1)
    return result, {"proposed": proposed, "accepted": accepted_total, "acceptance_rate": rate}
