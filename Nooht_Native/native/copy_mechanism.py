"""外层优化层 4：复制机制。

对应任务包第三节 4：
- 目的：代码及其他又长又必须逐字准确的内容（数字、网址、路径、公式、
  表格、引用原文、ID），从思考中直接复制到输出，不再逐字重新生成；
- 做法：思考里写出内容后，输出阶段发「复制区间」指令，解码器直接拷贝；
- 被复制内容必须在思考里已有；代码块要有「定稿」标记，只允许复制定稿内容；
- 复制不纠错，思考里错了输出也错；
- 被复制 token 仍需计算 KV 缓存，但可并行处理（类似预填充）；
- 训练数据自动构造：最长逐字重合区间标成复制区间；
- 本阶段只交付格式和接口，不承诺训练效果。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

COPY_START = "<copy:"
COPY_END = ">"
FINAL_MARKER = "<final>"  # 代码块定稿标记（任务包待确认 11：标记方式待定，此处为默认格式）


@dataclass
class CopySpan:
    """复制区间：思考文本中的 [start, end) 字符区间。"""

    start: int
    end: int
    finalized: bool = False  # 代码块是否已定稿
    source: str = "thinking"


def format_copy_instruction(span: CopySpan) -> str:
    """格式化复制区间指令：<copy:start-end>。"""
    return f"{COPY_START}{span.start}-{span.end}{COPY_END}"


def parse_copy_instruction(text: str) -> list[CopySpan]:
    """解析文本中的全部复制指令。"""
    spans = []
    for m in re.finditer(r"<copy:(\d+)-(\d+)>", text):
        spans.append(CopySpan(start=int(m.group(1)), end=int(m.group(2))))
    return spans


def is_copyable(span: CopySpan, thinking_text: str, is_code: bool = False) -> bool:
    """校验区间是否可复制：必须在思考内；代码块必须已定稿。"""
    if span.start < 0 or span.end > len(thinking_text) or span.start >= span.end:
        return False
    if is_code and not span.finalized:
        return False
    return True


class CopyDecoder:
    """解码器直接拷贝：保证一字不差（不纠错）。"""

    def copy_span(self, thinking_text: str, span: CopySpan, is_code: bool = False) -> str:
        if not is_copyable(span, thinking_text, is_code=is_code):
            raise ValueError(f"区间不可复制: {span}（代码块需定稿标记）")
        return thinking_text[span.start : span.end]

    def copy_with_parallel_kv(self, thinking_text: str, span: CopySpan,
                              is_code: bool = False) -> dict:
        """被复制 token 仍需 KV 缓存，但可并行处理（类似预填充）。

        本阶段只交付接口：返回拷贝文本与并行处理标记，
        实际 KV 并行写入由生成循环/模型侧实现。
        """
        copied = self.copy_span(thinking_text, span, is_code=is_code)
        return {
            "text": copied,
            "num_tokens": len(copied),
            "kv_parallel": True,  # 收益是变快，不是免费
            "note": "复制不纠错：思考里错了输出也错",
        }


def find_longest_overlap(thinking: str, answer: str) -> CopySpan | None:
    """自动构造训练数据：找思考与答案的最长逐字重合区间。

    返回思考侧的区间（复制来源）；无重合（长度 < 4）返回 None。
    """
    n, m = len(thinking), len(answer)
    if n == 0 or m == 0:
        return None
    # 动态规划求最长公共子串（O(n*m)，长文本调用方需截断）
    prev = [0] * (m + 1)
    best_len, best_end = 0, 0
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        for j in range(1, m + 1):
            if thinking[i - 1] == answer[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best_len:
                    best_len, best_end = cur[j], i
        prev = cur
    if best_len < 4:
        return None
    return CopySpan(start=best_end - best_len, end=best_end)


def auto_construct_copy_labels(pairs: list[tuple[str, str]]) -> list[dict]:
    """批量自动构造复制区间标签：[(thinking, answer)] -> 标签列表。"""
    labels = []
    for thinking, answer in pairs:
        span = find_longest_overlap(thinking, answer)
        labels.append(
            {
                "span": span,
                "instruction": format_copy_instruction(span) if span else None,
                "copied_text": thinking[span.start : span.end] if span else None,
            }
        )
    return labels
