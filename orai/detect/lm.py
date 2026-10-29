"""基于小语言模型的零样本检测器和「token 平滑度」统计。

- Fast-DetectGPT（解析版，采样模型 = 打分模型）：分数越高越像 AI。
- Binoculars：observer = base，performer = instruct，分数越低越像 AI。
- lp_mean / lp_std：逐 token 对数概率的均值与标准差，std 越大越「不平滑」、越像人。

默认用 Qwen3.5-0.8B-Base / Qwen3.5-0.8B，两者分词器相同，合计约 3.5 GB 显存。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load(path: str, device: str):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
    return tok, model


@dataclass
class LMScores:
    fast_detectgpt: float
    binoculars: float
    lp_mean: float
    lp_std: float
    n_tokens: int


class LMDetector:
    def __init__(self, base="models/Qwen3.5-0.8B-Base-text", instruct="models/Qwen3.5-0.8B-text",
                 device="cuda", max_tokens=512):
        self.device = device
        self.max_tokens = max_tokens
        self.tok, self.base = _load(base, device)
        self.inst = None
        if instruct:
            _, self.inst = _load(instruct, device)

    @torch.no_grad()
    def score_batch(self, texts: list[str]) -> list[LMScores]:
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=self.max_tokens).to(self.device)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        logits_b = self.base(**enc).logits[:, :-1]
        logits_i = self.inst(**enc).logits[:, :-1] if self.inst is not None else None
        labels, m = ids[:, 1:], mask[:, 1:].float()

        # 词表 24.8 万，整段转 float32 放不下，按位置分块算逐 token 统计量
        ll, mean_ref, var_ref, ll_i, xent_t = [], [], [], [], []
        for s in range(0, labels.shape[1], 128):
            lab = labels[:, s:s + 128].unsqueeze(-1)
            logp_b = torch.log_softmax(logits_b[:, s:s + 128].float(), -1)
            p_b = logp_b.exp()
            ll.append(logp_b.gather(-1, lab).squeeze(-1))                    # log p(x_t)
            mu = (p_b * logp_b).sum(-1)                                       # E[log p]
            mean_ref.append(mu)
            var_ref.append((p_b * logp_b.square()).sum(-1) - mu.square())     # Var[log p]
            if logits_i is not None:
                logp_i = torch.log_softmax(logits_i[:, s:s + 128].float(), -1)
                ll_i.append(logp_i.gather(-1, lab).squeeze(-1))
                xent_t.append(-(p_b * logp_i).sum(-1))
            del logp_b, p_b
        ll, mean_ref, var_ref = torch.cat(ll, 1), torch.cat(mean_ref, 1), torch.cat(var_ref, 1)

        n = m.sum(-1)
        fdg = ((ll - mean_ref) * m).sum(-1) / torch.sqrt((var_ref * m).sum(-1).clamp_min(1e-6))
        lp_mean = (ll * m).sum(-1) / n
        lp_std = torch.sqrt((((ll - lp_mean[:, None]) ** 2) * m).sum(-1) / n)

        if logits_i is not None:
            # 与官方实现一致：ppl 用 performer，x-ppl 用 observer 分布对 performer 的交叉熵
            nll_i = -(torch.cat(ll_i, 1) * m).sum(-1) / n
            xent = (torch.cat(xent_t, 1) * m).sum(-1) / n
            bino = nll_i / xent
        else:
            bino = torch.full_like(fdg, float("nan"))

        return [LMScores(float(a), float(b), float(c), float(d), int(e))
                for a, b, c, d, e in zip(fdg, bino, lp_mean, lp_std, n)]

    def score(self, texts: list[str], batch_size=4) -> list[LMScores]:
        out = []
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        res = {}
        for k in range(0, len(order), batch_size):
            idx = order[k:k + batch_size]
            for i, s in zip(idx, self.score_batch([texts[i] for i in idx])):
                res[i] = s
        for i in range(len(texts)):
            out.append(res[i])
        return out


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))
