"""把 Qwen3.5 多模态 checkpoint 转成纯文本 CausalLM：去掉视觉塔和 MTP 头，省显存，也让 unsloth / peft 按普通 LLM 处理。

python -m orai.train.text_only models/Qwen3.5-4B-Base models/Qwen3.5-4B-Base-text
"""

import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main(src: str, dst: str):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    text = dict(cfg["text_config"])
    text["architectures"] = ["Qwen3_5ForCausalLM"]
    text["tie_word_embeddings"] = cfg.get("tie_word_embeddings", True)
    text.pop("mtp_num_hidden_layers", None)
    (dst / "config.json").write_text(json.dumps(text, indent=2))

    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    shards: dict[str, list[str]] = {}
    for k, f in index.items():
        if k.startswith("model.language_model."):
            shards.setdefault(f, []).append(k)
    new_map = {}
    for i, (f, keys) in enumerate(sorted(shards.items())):
        tensors = {}
        with safe_open(src / f, "pt") as fh:
            for k in keys:
                tensors["model." + k[len("model.language_model."):]] = fh.get_tensor(k)
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(tensors, dst / name, metadata={"format": "pt"})
        new_map.update({k: name for k in tensors})
        print("wrote", name, len(tensors))
    (dst / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": new_map}, indent=1))
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json",
              "chat_template.jinja", "special_tokens_map.json"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    # 多模态版的 tokenizer_config 指向 processor；纯文本版只保留 tokenizer
    tc = json.loads((dst / "tokenizer_config.json").read_text())
    tc.pop("processor_class", None)
    tc["pad_token"] = "<|endoftext|>"
    (dst / "tokenizer_config.json").write_text(json.dumps(tc, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
