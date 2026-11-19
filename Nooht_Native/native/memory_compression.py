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

    def score_segments(self, segments: list[Segment]) -> list[Segment]:
        """打分：辅助打分头（已训练时）+ 规则兜底。"""
        if self.scoring_head is not None and self.scoring_head.trained:
            # 训练后的打分头路径：此处用文本长度作占位特征，真实接入时替换为 hidden
            feats = torch.tensor(
                [[min(len(s.text), 1000) / 1000.0] * 256 for s in segments],
                dtype=torch.float32,
            )
            with torch.no_grad():
                scores = self.scoring_head(feats).tolist()
        else:
            scores = self.rule_scorer.score(segments)
        for seg, sc in zip(segments, scores):
            seg.score = float(sc)
        return segments

    def compress(self, segments: list[Segment]) -> list[Segment]:
        """低分片段换成摘要或降权；受保护区不参与压缩。"""
        if not self.enabled:
            return segments
        self.score_segments(segments)
        out = []
        for seg in segments:
            if seg.protected:
                out.append(seg)
            elif seg.score < self.low_score_threshold:
                summary = seg.summary or f"[摘要: {seg.text[:40]}…]"
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
