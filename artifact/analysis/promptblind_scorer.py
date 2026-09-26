"""feat-217 (results/onset_prediction_promptblind.md): a vetted scorer. Comma-7B scores each pool draw by its mean
per-token log-likelihood with NO prompt in context; an empty draw scores -1e9. Writes the reward-cache format of
results/selection_rewards64.csv for the prompts in this shard (prompt index mod --shards == --shard).
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python analysis/promptblind_scorer.py --shard 0 --shards 8 --out <csv>
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_decoding import load_candidates  # noqa: E402

EMPTY = -1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="output/phase5/sel_anchor64")
    ap.add_argument("--model", default="common-pile/comma-v0.1-2t")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cands = load_candidates(a.pool, deecho=True)
    pids = sorted(cands)[a.shard::a.shards]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda().eval()
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    rows = []
    for p in pids:
        c = cands[p]
        texts = [x[3] for x in c]
        scores = [EMPTY] * len(texts)
        idx = [i for i, t in enumerate(texts) if t.strip()]
        for s in range(0, len(idx), a.batch):
            chunk = idx[s:s + a.batch]
            ids = [[bos] + tok(texts[i], add_special_tokens=False)["input_ids"] for i in chunk]
            L = max(len(x) for x in ids)
            inp = torch.full((len(ids), L), tok.pad_token_id if tok.pad_token_id is not None else 0, dtype=torch.long)
            mask = torch.zeros((len(ids), L), dtype=torch.long)
            for j, x in enumerate(ids):          # right-padded: each row's tokens start at position 0
                inp[j, :len(x)] = torch.tensor(x)
                mask[j, :len(x)] = 1
            with torch.no_grad():
                logits = model(input_ids=inp.cuda(), attention_mask=mask.cuda()).logits.float()
            lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, inp[:, 1:].cuda().unsqueeze(-1)).squeeze(-1)
            m = mask[:, 1:].cuda().float()
            mean = (lp * m).sum(1) / m.sum(1).clamp(min=1)
            for j, i in enumerate(chunk):
                scores[i] = float(mean[j])
        for r, (x, sc) in enumerate(zip(c, scores)):
            rows.append(dict(prompt_id=p, rank=r, prompt_class=x[1], n_words=len(x[3].split()), reward=round(sc, 5)))
        print(f"[pb] {p} done", flush=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["prompt_id", "rank", "prompt_class", "n_words", "reward"])
        w.writeheader()
        w.writerows(rows)
    print(f"[pb] wrote {len(rows)} rows for {len(pids)} prompts", flush=True)


if __name__ == "__main__":
    main()
