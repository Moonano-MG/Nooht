"""贪心推测解码与草稿校验。

草稿回调支持两种明确定义的输出：
- ``draft_output_mode='causal'``：普通因果 LM 输出 ``(B,L,V)``，每次只取最后
  位置的 next-token logits，并逐 token 自回归构造 K 个草稿；
- ``draft_output_mode='future'``：回调直接输出未来位置 logits ``(B,K,V)``，或
  Noild DraftHead 的 ``(B,L,K,V)``（取最后一个序列位置的 K 个未来预测）。

目标模型对 ``prefix + draft`` 做一次前向，逐位置校验草稿。该基础实现不使用
KV cache，也不支持概率式采样；未验证其加速收益。
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


def _extract(output) -> torch.Tensor:
    if hasattr(output, "logits"):
        output = output.logits
    elif isinstance(output, dict) and "logits" in output:
        output = output["logits"]
    elif isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor) or output.ndim not in (2, 3, 4):
        raise TypeError("回调必须返回 token ids 或 (B,L,V)/(B,L,K,V) logits")
    return output


def _future_logits(output, mode: str) -> torch.Tensor:
    """统一为 (B,K,V) 未来 logits；2D 输出解释为直接 token ids。"""
    value = _extract(output)
    if value.ndim == 2:
        return value.unsqueeze(-1)  # ids sentinel, last dim 1
    if value.ndim == 4:
        # DraftExpert: (B, sequence, draft_num_tokens, vocab)
        return value[:, -1, :, :]
    if mode == "future":
        return value
    # 普通 causal LM：只允许使用最后位置来预测紧邻的下一个 token。
    return value[:, -1:, :]


def speculative_greedy_decode(
    input_ids: torch.Tensor,
    draft_fn: Callable[[torch.Tensor], torch.Tensor],
    target_fn: Callable[[torch.Tensor], torch.Tensor],
    max_new_tokens: int,
    *,
    draft_num_tokens: int = 4,
    draft_output_mode: str = "causal",
    eos_token_id: int | None = None,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """推测贪心解码，返回扩展后的 ids 与提议/接受统计。

    ``causal`` 模式通过多次 drafter 前向生成连续草稿，正确但未必更快；
    ``future`` 模式供真正的多 token 草稿头使用。目标一次前向验证 K 个草稿，
    全接受时可额外采用目标模型的 bonus token（该 token 不计入 proposed）。
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("当前基础实现只支持 batch size = 1。")
    if max_new_tokens < 0 or draft_num_tokens <= 0:
        raise ValueError("max_new_tokens 必须非负，draft_num_tokens 必须为正。")
    if draft_output_mode not in ("causal", "future"):
        raise ValueError("draft_output_mode 只能是 causal 或 future。")
    result = input_ids
    proposed = accepted_total = 0
    initial_len = input_ids.shape[1]
    while result.shape[1] - initial_len < max_new_tokens:
        remaining = max_new_tokens - (result.shape[1] - initial_len)
        k = min(draft_num_tokens, remaining)
        if draft_output_mode == "future":
            future = _future_logits(draft_fn(result), "future")
            if future.shape[0] != 1:
                raise ValueError("draft_fn batch size 必须为 1。")
            if future.shape[1] < k:
                raise ValueError(f"future draft logits 只给出 {future.shape[1]} 个位置，要求 {k}。")
            if future.shape[-1] == 1:
                draft_ids = future[:, :k, 0].long()
            else:
                draft_ids = future[:, :k, :].argmax(dim=-1)
        else:
            # 标准因果 LM 的整段输出只取最后位置，绝不把 prompt 各位置误当未来草稿。
            draft_ids = []
            provisional = result
            for _ in range(k):
                raw = _extract(draft_fn(provisional))
                if raw.ndim == 4:
                    step_logits = raw[:, -1, :, :]
                    next_id = step_logits[:, 0, :].argmax(dim=-1, keepdim=True)
                elif raw.ndim == 3:
                    next_id = raw[:, -1, :].argmax(dim=-1, keepdim=True)
                elif raw.ndim == 2:
                    next_id = raw[:, -1:].long()
                else:
                    raise TypeError("causal draft_fn 需返回 (B,L,V) logits 或 (B,L) token ids")
                draft_ids.append(next_id)
                provisional = torch.cat((provisional, next_id), dim=1)
            draft_ids = torch.cat(draft_ids, dim=1)

        target_input = torch.cat((result, draft_ids), dim=1)
        target_logits = _extract(target_fn(target_input))
        if target_logits.ndim != 3 or target_logits.shape[0] != 1:
            raise TypeError("target_fn 必须返回 (1, sequence, vocab) logits")
        start = result.shape[1] - 1
        needed = start + draft_ids.shape[1]
        if target_logits.shape[1] < needed:
            raise ValueError("target_fn logits 序列长度不足以验证全部草稿 token。")
        target_ids = target_logits[:, start:needed, :].argmax(dim=-1)
        count, _ = verify_draft(draft_ids[0], target_ids[0])
        proposed += draft_ids.shape[1]
        accepted_total += count
        if count < draft_ids.shape[1]:
            correction = target_ids[:, count : count + 1]
            result = torch.cat((result, draft_ids[:, :count], correction), dim=1)
            if eos_token_id is not None and int(correction.item()) == eos_token_id:
                break
        else:
            result = torch.cat((result, draft_ids), dim=1)
            if eos_token_id is not None and bool((draft_ids == eos_token_id).any()):
                break
            # target 的最后一个位置给出 bonus token；多产一个目标 token，减少下一轮
            # drafter 前向。它不是草稿，不计入 accepted/proposed。
            if result.shape[1] - initial_len < max_new_tokens:
                bonus_index = start + draft_ids.shape[1]
                if bonus_index < target_logits.shape[1]:
                    bonus = target_logits[:, bonus_index, :].argmax(dim=-1, keepdim=True)
                    result = torch.cat((result, bonus), dim=1)
                    if eos_token_id is not None and int(bonus.item()) == eos_token_id:
                        break
    return result, {
        "proposed": proposed,
        "accepted": accepted_total,
        "acceptance_rate": accepted_total / max(proposed, 1),
    }
