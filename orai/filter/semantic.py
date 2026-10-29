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
                 "len_ratio": len(y) / max(len(x), 1)}
                for s, f, b, x, y in zip(sim, fwd, bwd, src, out)]
