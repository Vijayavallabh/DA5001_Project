"""Plan v4 / feat-042: ordinary completions from a second risky model, on the same prompts.

c_use in analysis/anchor_scaling.py is the median per-character excess of a safe model's surprisal
over the risky model's, on text the risky model itself produced. Measured against one risky model,
the cross-safe-model trend is internally valid but a reviewer will ask whether it depends on that
model being Llama. This generates the same workload from another one so the trend can be recomputed.

The prompts are exactly those analysis/anchor_scaling.ordinary_texts uses, so the two runs differ
only in which model wrote the continuations.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python scripts/gen_ordinary.py --model Qwen/Qwen2.5-7B-Instruct \
      --out output/phase4/ordinary_qwen.jsonl
"""
import argparse, json, os, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.anchor_scaling import ordinary_texts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(device).eval()

    items = ordinary_texts(a.n)
    print(f"[gen] {a.model}: {len(items)} prompts", flush=True)
    out, t0 = [], time.time()
    for i in range(0, len(items), a.batch):
        chunk = items[i:i + a.batch]
        prompts = [p for _, _, p, _ in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=True,
                                 temperature=a.temperature, top_p=1.0,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for (cls, uid, prompt, _), seq in zip(chunk, gen):
            text = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            if len(text) >= a.min_chars:
                out.append({"cls": cls, "id": uid, "prompt": prompt, "generation": text})
        if (i // a.batch) % 5 == 0:
            print(f"  {min(i + a.batch, len(items))}/{len(items)}  {time.time() - t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"[gen] wrote {a.out}: {len(out)} completions of >= {a.min_chars} chars "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
