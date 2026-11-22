"""外层优化层 2：选择性记忆压缩。

对应任务包第三节 2：
- 在输入上下文层面处理，不碰模型内部状态；
- 对话达到长度阈值（如上下文一半）后，按语义片段打分，
  低分片段换成摘要或降权后再送入模型；
- 受保护区内容不参与压缩；
- 打分用辅助打分头加规则兜底；辅助打分头是独立小模型，需要训练；
- 设计文档写明选择依据，并做开启与关闭的消融对比。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class Segment:
    """语义片段：压缩的最小单位。"""

    text: str
    score: float = 0.0  # 重要性分数，越高越保留
    protected: bool = False  # 受保护区：不参与压缩
    summary: str | None = None  # 低分片段的摘要（压缩后替换文本）


def split_segments(text: str) -> list[Segment]:
    """按语义片段切分：段落/换行/句末标点。"""
    parts = re.split(r"(\n+|(?<=[。！？.!?])\s+)", text)
    segments = []
    for p in parts:
        if p and p.strip():
            segments.append(Segment(text=p.strip()))
    return segments or [Segment(text=text)]


class RuleScorer:
    """规则兜底打分：新近性 + 长度 + 与指令锚点关键词重合。

    辅助打分头未训练/不可用时使用（任务包：辅助打分头加规则兜底）。
    """

    def __init__(self, anchor_keywords: set[str] | None = None):
        self.anchor_keywords = anchor_keywords or set()

    def score(self, segments: list[Segment]) -> list[float]:
        n = len(segments)
        scores = []
        for i, seg in enumerate(segments):
            recency = (i + 1) / max(n, 1)  # 越新越高
            length_score = min(len(seg.text) / 200.0, 1.0)
            overlap = 0.0
            if self.anchor_keywords:
                hits = sum(1 for kw in self.anchor_keywords if kw in seg.text)
                overlap = min(hits / max(len(self.anchor_keywords), 1), 1.0)
            scores.append(0.5 * recency + 0.2 * length_score + 0.3 * overlap)
        return scores


class ScoringHead(nn.Module):
    """辅助打分头：独立小模型，需要训练（任务包第三节 2）。

    输入片段 hidden（或其均值），输出重要性分数 0~1。
    未训练前仅作接口，实际打分走 RuleScorer 兜底。
    """

    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )
        self.trained = False  # 训练后置 True

    def forward(self, segment_hidden: torch.Tensor) -> torch.Tensor:
        return self.net(segment_hidden).squeeze(-1)


def fit_scoring_head(
    scoring_head: ScoringHead,
    segment_hidden: torch.Tensor,
    importance_targets: torch.Tensor,
    *,
    steps: int = 100,
    learning_rate: float = 1e-3,
) -> list[float]:
    """在调用方提供的真实 pooled hidden/人工或任务标签上拟合打分头。"""
    if steps <= 0 or learning_rate <= 0:
        raise ValueError("steps 和 learning_rate 必须为正数。")
    if segment_hidden.ndim != 2 or importance_targets.ndim != 1:
        raise ValueError("特征需为二维张量，重要性标签需为一维张量。")
    if segment_hidden.shape[0] != importance_targets.shape[0] or not segment_hidden.shape[0]:
        raise ValueError("特征与标签的样本数必须相同且非零。")
    if segment_hidden.shape[1] != scoring_head.net[0].in_features:
        raise ValueError("特征 hidden_size 与 ScoringHead 不匹配。")
    if bool(((importance_targets < 0) | (importance_targets > 1)).any()):
        raise ValueError("importance_targets 必须位于 [0, 1]。")
    parameter = next(scoring_head.parameters())
    features = segment_hidden.to(device=parameter.device, dtype=parameter.dtype)
    targets = importance_targets.to(device=parameter.device, dtype=parameter.dtype)
    optimizer = torch.optim.AdamW(scoring_head.parameters(), lr=learning_rate)
    scoring_head.train()
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        predictions = scoring_head(features)
        loss = torch.nn.functional.binary_cross_entropy(predictions, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    scoring_head.trained = True
    scoring_head.eval()
    return losses


def extractive_summary(text: str, max_chars: int | None = None) -> str:
    """确定性抽取式摘要：选取高信息密度完整短语，不把原文机械截断。

    这是无生成模型依赖的规则摘要，不宣称具备语义生成能力。优先保留
    含数字、路径、引号内容或关键因果词的分句，并按原顺序拼接。
    """
    budget = max_chars or max(24, min(160, len(text) // 3))
    clauses = [c.strip() for c in re.split(r"(?<=[。！？.!?；;])|[,，、]", text) if c.strip()]
    if not clauses:
        return "【摘要】" + text
    markers = re.compile(r"\d|/|\\|https?://|[=:]|必须|因此|因为|结论|重要|错误|不要|不能")
    ranked = sorted(
        enumerate(clauses),
        key=lambda item: (bool(markers.search(item[1])), min(len(item[1]), 120), -item[0]),
        reverse=True,
    )
    selected = []
    used = 0
    for _, clause in ranked:
        addition = len(clause) + (1 if selected else 0)
        if used + addition <= budget:
            selected.append(clause)
            used += addition
    if not selected:
        # 没有可抽取的完整短语时，至少给出显式摘要标记和信息词，而非截取前缀。
        tokens = re.findall(r"[A-Za-z0-9_./:-]+|[\u4e00-\u9fff]{2,}", text)
        selected = tokens[: max(1, budget // 8)] or ["内容过长"]
    return "【摘要】" + "；".join(selected)


class SelectiveMemoryCompressor:
    """选择性记忆压缩器：只处理输入上下文，不碰模型内部状态。"""

    def __init__(
        self,
        threshold_ratio: float = 0.5,
        max_context_len: int = 8192,
        enabled: bool = True,
        low_score_threshold: float = 0.3,
        scoring_head: ScoringHead | None = None,
        anchor_keywords: set[str] | None = None,
    ):
        self.threshold_ratio = threshold_ratio
        self.max_context_len = max_context_len
        self.enabled = enabled
        self.low_score_threshold = low_score_threshold
        self.scoring_head = scoring_head
        self.rule_scorer = RuleScorer(anchor_keywords)

    def should_compress(self, current_len: int) -> bool:
        if not self.enabled:
            return False
        return current_len >= self.threshold_ratio * self.max_context_len

    def score_segments(
        self, segments: list[Segment], segment_hidden: torch.Tensor | None = None
    ) -> list[Segment]:
        """用已训练 head 和真实片段特征打分，否则明确回退规则评分。

        不再把文本长度复制成伪 hidden 特征；调用方要使用训练 head，必须传入
        与片段一一对应的真实 pooled hidden states。
        """
        use_head = (
            self.scoring_head is not None
            and self.scoring_head.trained
            and segment_hidden is not None
        )
        if use_head:
            if segment_hidden.ndim != 2 or segment_hidden.shape[0] != len(segments):
                raise ValueError("segment_hidden 必须为 (片段数, hidden_size)。")
            expected = self.scoring_head.net[0].in_features
            if segment_hidden.shape[1] != expected:
                raise ValueError(f"segment_hidden hidden_size 应为 {expected}。")
            parameter = next(self.scoring_head.parameters())
            feats = segment_hidden.to(device=parameter.device, dtype=parameter.dtype)
            with torch.no_grad():
                scores = self.scoring_head(feats).tolist()
        else:
            scores = self.rule_scorer.score(segments)
        for seg, sc in zip(segments, scores):
            seg.score = float(sc)
        return segments

    def compress(
        self, segments: list[Segment], segment_hidden: torch.Tensor | None = None
    ) -> list[Segment]:
        """低分片段换成摘要或降权；受保护区不参与压缩。"""
        if not self.enabled:
            return segments
        self.score_segments(segments, segment_hidden=segment_hidden)
        out = []
        for seg in segments:
            if seg.protected:
                out.append(seg)
            elif seg.score < self.low_score_threshold:
                summary = seg.summary or extractive_summary(seg.text)
                out.append(Segment(text=summary, score=seg.score, protected=False))
            else:
                out.append(seg)
        return out

    def assemble(self, segments: list[Segment]) -> str:
        return "\n".join(s.text for s in segments)

    def ablation(self, segments: list[Segment]) -> dict:
        """开启与关闭的消融对比：返回压缩前后长度与片段数。"""
        before = self.assemble(segments)
        saved_enabled = self.enabled
        self.enabled = True
        after = self.assemble(self.compress(list(segments)))
        self.enabled = saved_enabled
        return {
            "before_chars": len(before),
            "after_chars": len(after),
            "saved_ratio": 1.0 - len(after) / max(len(before), 1),
            "num_segments": len(segments),
        }
