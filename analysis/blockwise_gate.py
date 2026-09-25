"""feat-201 gate G1 (results/onset_prediction_blockwise.md): are the blockwise draws the anchor's law?

Mean anchor surprisal of the served tokens, in nats per token, with each text re-tokenised and scored
with its prompt as context, for (a) the committed pool's rank-0 draws (de-echoed), (b) blk200n1, the
anchor alone through analysis/blockwise_selection.py, and (c) blk200n1_shipped, the same under the
checkpoint's shipped sampling (temperature 0.6, top_p 0.9), the defect the gate must be able to see.
Every text goes through the same function, so tokenisation cannot separate them. Empty texts carry no
token and are skipped. Writes results/blockwise_gate.csv.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/blockwise_gate.py --out results
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.blockwise_selection import CLASSES, block_logprob  # noqa: E402
from analysis.selection_decoding import load_candidates  # noqa: E402


def arm_texts(d, tag):
    out = {}
    for cls in CLASSES:
        path = os.path.join(d, f"trajectories_k{tag}_{cls}.jsonl")
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            out[r["metadata"]["prompt_id"]] = (r["prefix_analysis"]["prefix_text"], r["aggregate"]["generation"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", default="output/phase5/sel_anchor64")
    ap.add_argument("--n1-dir", default="output/feat201/blk200n1")
    ap.add_argument("--shipped-dir", default="output/feat201/blk200n1_shipped")
    ap.add_argument("--anchor", default="jacquelinehe/tinycomma-1.8b-llama3-tokenizer")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.anchor)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(a.anchor, torch_dtype=torch.bfloat16).cuda().eval()

    prefix = {}
    for cls in CLASSES:
        for line in open(os.path.join(a.pool_dir, f"trajectories_k0_{cls}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            prefix.setdefault(r["metadata"]["prompt_id"], r["prefix_analysis"]["prefix_text"])
    pool = {p: (prefix[p], v[0][3]) for p, v in load_candidates(a.pool_dir, deecho=True).items()}
    n1, shipped = arm_texts(a.n1_dir, "blk200n1"), arm_texts(a.shipped_dir, "blk200n1_shipped")
    assert set(n1) == set(pool) and len(pool) == 500, (len(n1), len(pool))
    assert set(shipped) <= set(pool) and len(shipped) == 100, len(shipped)

    def rate(texts, pids):
        keep = [p for p in pids if tok(texts[p][1], add_special_tokens=False).input_ids]
        ctx = [tok(texts[p][0]).input_ids for p in keep]
        gen = [tok(texts[p][1], add_special_tokens=False).input_ids for p in keep]
        lp = block_logprob(model, pad, ctx, gen, a.batch, mean=True)
        return -sum(lp) / len(lp), len(keep)

    rows = []
    for name, texts, pids in (("pool rank 0", pool, sorted(pool)), ("blk200n1", n1, sorted(pool)),
                              ("pool rank 0, probe prompts", pool, sorted(shipped)),
                              ("blk200n1_shipped", shipped, sorted(shipped))):
        r, k = rate(texts, pids)
        rows.append(dict(source=name, prompts=len(pids), nonempty=k, nats_per_token=round(r, 4)))
        print(rows[-1], flush=True)
    ref, ref100 = rows[0]["nats_per_token"], rows[2]["nats_per_token"]
    rel = abs(rows[1]["nats_per_token"] - ref) / ref
    power = abs(rows[3]["nats_per_token"] - ref100) / ref100
    reading = ("UNINFORMATIVE" if power < 0.10 else "PASS" if rel <= 0.05 else "FAIL")
    rows.append(dict(source="G1", prompts=500, nonempty="", nats_per_token=f"rel {rel:.4f}, probe {power:.4f}: {reading}"))
    with open(os.path.join(a.out, "blockwise_gate.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"G1 {reading}: blk200n1 differs from the pool by {rel:.2%}, the shipped probe by {power:.2%}")


if __name__ == "__main__":
    main()
