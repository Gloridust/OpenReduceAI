"""意思保持的度量：嵌入相似度、双向 NLI 蕴含、硬信息（数字、英文术语、引号内容）一致。"""

from __future__ import annotations

import re

import torch

NUM_RE = re.compile(r"\d+(?:[.．]\d+)?%?|[零一二三四五六七八九十百千万亿]+(?:分之[零一二三四五六七八九十百]+)")
LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-+.]*[A-Za-z0-9]|[A-Za-z]")
QUOTE_RE = re.compile(r"[“《]([^”》]{1,30})[”》]")


def hard_facts(text: str) -> set[str]:
    nums = {re.sub(r"[．]", ".", n) for n in re.findall(r"\d+(?:[.．]\d+)?%?", text)}
    latin = {w.lower() for w in LATIN_RE.findall(text) if len(w) >= 2}
    quotes = set(QUOTE_RE.findall(text))
    return nums | latin | quotes


def fact_recall(src: str, out: str, numbers_only: bool = False) -> float:
    """源文的硬信息在输出里保留的比例（没有硬信息时记 1）。
    numbers_only：口语语料里的英文多是网络用语（orz、qaq），只核对两位以上的数字。"""
    f = {n for n in re.findall(r"\d{2,}(?:[.．]\d+)?%?", src)} if numbers_only else hard_facts(src)
    if not f:
        return 1.0
    out_l = out.lower()
    return sum(1 for x in f if x.lower() in out_l) / len(f)


CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}
CN_NUM_RE = re.compile(r"[零〇一二两三四五六七八九十百千万亿]+")
LIST_MARK_RE = re.compile(r"(?<![\d.])[1-9](?=[、.．)）](?!\d))|[(（][1-9][)）]")


def cn2int(s: str) -> int | None:
    if all(c in CN_DIGIT for c in s):  # 「二〇〇八」这种逐位写法
        return int("".join(str(CN_DIGIT[c]) for c in s)) if len(s) > 1 else CN_DIGIT[s]
    total, section, digit = 0, 0, None
    for c in s:
        if c in CN_DIGIT:
            digit = CN_DIGIT[c]
        elif c in ("万", "亿"):
            total = (total + section + (digit or 0)) * CN_UNIT[c]
            section, digit = 0, None
        else:
            section += (1 if digit is None else digit) * CN_UNIT[c]
            digit = None
    return total + section + (digit or 0)


def _halfwidth(t: str) -> str:
    return "".join(chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c for c in t)


def added_facts(src: str, out: str) -> list[str]:
    """输出里出现、但源文没有的数字和带数字的型号（如 TY-190）。改写最怕的是凭空多出日期、数量。
    中文数字和阿拉伯数字互认（「三类」写成「3类」不算新增），1、2、3 这类列表编号不算。
    引号内容和普通英文词不查：加引号、换个网络用语不改变事实。"""
    norm = lambda t: re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", _halfwidth(t)).lower()
    src, out = norm(src), norm(out)
    src_flat = re.sub(r"\s+", "", src)
    known = set(re.findall(r"\d+(?:\.\d+)?", src)) | {str(cn2int(m)) for m in CN_NUM_RE.findall(src)}
    out = LIST_MARK_RE.sub("", out)
    facts = set(re.findall(r"\d+(?:\.\d+)?", out))
    facts |= {w for w in LATIN_RE.findall(out) if re.search(r"\d", w)}
    return sorted(f for f in facts if f not in known and re.sub(r"\s+", "", f) not in src_flat)


class SemanticScorer:
    def __init__(self, emb="models/bge-base-zh-v1.5", nli="models/Erlangshen-Roberta-110M-NLI", device="cuda"):
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.emb = SentenceTransformer(emb, device=device)
        self.emb.half()
        self.nli_tok = AutoTokenizer.from_pretrained(nli)
        self.nli = AutoModelForSequenceClassification.from_pretrained(nli).to(device).eval().half()
        labels = {v.lower(): int(k) for k, v in self.nli.config.id2label.items()}
        self.ent_idx = next(i for name, i in labels.items() if name.startswith("entail"))

    @torch.no_grad()
    def similarity(self, a: list[str], b: list[str], batch_size=32) -> list[float]:
        ea = self.emb.encode(a, batch_size=batch_size, normalize_embeddings=True, convert_to_tensor=True)
        eb = self.emb.encode(b, batch_size=batch_size, normalize_embeddings=True, convert_to_tensor=True)
        return (ea * eb).sum(-1).float().tolist()

    @torch.no_grad()
    def entail(self, premise: list[str], hypo: list[str], batch_size=32) -> list[float]:
        out = []
        for k in range(0, len(premise), batch_size):
            enc = self.nli_tok(premise[k:k + batch_size], hypo[k:k + batch_size], truncation=True,
                               max_length=512, padding=True, return_tensors="pt").to(self.device)
            p = torch.softmax(self.nli(**enc).logits.float(), -1)[:, self.ent_idx]
            out.extend(p.tolist())
        return out

    def score(self, src: list[str], out: list[str]) -> list[dict]:
        sim = self.similarity(src, out)
        fwd = self.entail(src, out)
        bwd = self.entail(out, src)
        return [{"sim": s, "nli_fwd": f, "nli_bwd": b, "fact_recall": fact_recall(x, y),
                 "added": len(added_facts(x, y)), "len_ratio": len(y) / max(len(x), 1)}
                for s, f, b, x, y in zip(sim, fwd, bwd, src, out)]
