"""加载 基座 + LoRA，批量采样改写候选。"""

from __future__ import annotations

import torch

from orai.train.format import END, build_prompt


class Rewriter:
    def __init__(self, adapter: str, max_len: int = 1536, quant: str = "16bit"):
        """adapter 可以是 LoRA 目录（自动加载其基座），也可以是完整模型目录（如零样本基线）。
        quant：16bit 质量最好；显存不够时用 8bit（与训练时的基座精度一致）。"""
        from unsloth import FastLanguageModel

        self.model, self.tok = FastLanguageModel.from_pretrained(
            model_name=adapter, max_seq_length=max_len, load_in_4bit=quant == "4bit",
            load_in_8bit=quant == "8bit", load_in_16bit=quant == "16bit")
        FastLanguageModel.for_inference(self.model)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.end_id = self.tok.convert_tokens_to_ids(END)

    @torch.no_grad()
    def generate(self, texts: list[str], style="书面", n=4, temperature=0.9, top_p=0.95,
                 batch_size=4) -> list[list[str]]:
        """返回每条输入的 n 个候选。"""
        results: list[list[str]] = []
        for k in range(0, len(texts), batch_size):
            chunk = texts[k:k + batch_size]
            prompts = [build_prompt(t, style) for t in chunk]
            enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            max_new = int(max(len(t) for t in chunk) * 1.6) + 32
            out = self.model.generate(
                **enc, do_sample=temperature > 0, temperature=max(temperature, 1e-3), top_p=top_p,
                max_new_tokens=max_new, num_return_sequences=n, eos_token_id=[self.end_id, self.tok.eos_token_id],
                pad_token_id=self.tok.pad_token_id, repetition_penalty=1.05,
            )
            gen = out[:, enc["input_ids"].shape[1]:]
            decoded = self.tok.batch_decode(gen, skip_special_tokens=False)
            for i in range(len(chunk)):
                cands = []
                for s in decoded[i * n:(i + 1) * n]:
                    s = s.split(END)[0].replace(self.tok.pad_token, "").replace("<|endoftext|>", "")
                    cands.append(s.strip())
                results.append(cands)
        return results
