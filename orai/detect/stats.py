"""不依赖模型的文本特征：句长起伏、AI 套话、作弊字符。"""

from __future__ import annotations

import re
import statistics
import unicodedata

# 中文 AI 文本里出现频率明显偏高的连接词和套话（种子表，后续可用 log-odds 从数据里补充）
AI_PHRASES = [
    "值得注意的是", "需要注意的是", "值得一提的是", "综上所述", "总而言之", "总的来说", "总体而言",
    "此外", "与此同时", "不仅如此", "进一步", "在此基础上", "由此可见", "因此", "从而",
    "具有重要意义", "具有重要的", "起着至关重要", "至关重要", "不可或缺", "显著", "有效提升",
    "深入探讨", "深入分析", "系统阐述", "全面", "多维度", "多层次", "日益", "愈发", "进而",
    "旨在", "致力于", "助力", "赋能", "为……提供", "提供了有力", "有力支撑", "坚实基础",
    "首先", "其次", "最后", "一方面", "另一方面", "在……背景下", "随着", "不断",
    "通过", "基于", "借助", "围绕", "聚焦",
]

# 改写模型作弊时常用的字符：零宽、方向控制、同形异体字母
_SUSPICIOUS = re.compile(
    "[​-‏‪-‮⁠-⁤﻿­͏᠎]"
    "|[Ѐ-ӿͰ-Ͽ]"  # 西里尔、希腊字母（冒充拉丁字母）
)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?；;])", text)
    return [p for p in (s.strip() for s in parts) if p]


def sentence_len_cv(text: str) -> float:
    lens = [len(s) for s in split_sentences(text)]
    if len(lens) < 2:
        return 0.0
    return statistics.pstdev(lens) / max(statistics.mean(lens), 1)


def ai_phrase_count(text: str) -> int:
    n = 0
    for p in AI_PHRASES:
        if "……" in p:
            a, b = p.split("……")
            n += len(re.findall(re.escape(a) + r".{1,20}?" + re.escape(b), text))
        else:
            n += text.count(p)
    return n


def ai_phrase_density(text: str) -> float:
    """每百字套话数。"""
    return ai_phrase_count(text) * 100 / max(len(text), 1)


def cheat_flags(text: str, ref: str | None = None) -> list[str]:
    """改写结果里新出现的作弊特征。给了 ref（原文）时，原文本来就有的特征不算。"""
    flags = _raw_flags(text)
    if ref is not None:
        base = set(_raw_flags(ref))
        flags = [f for f in flags if f not in base]
    return flags


def _raw_flags(text: str) -> list[str]:
    flags = []
    if _SUSPICIOUS.search(text):
        flags.append("suspicious_char")
    if any(unicodedata.category(c) == "Co" for c in text):
        flags.append("private_use_char")
    half = len(re.findall(r"[,.;:?!](?=[一-鿿])", text))
    if half >= 3:
        flags.append("halfwidth_punct")
    if re.search(r"(.{4,})\1\1", text):
        flags.append("repetition")
    latin = len(re.findall(r"[A-Za-z]{2,}", text))
    if latin > max(6, len(text) / 40):
        flags.append("latin_heavy")
    return flags
