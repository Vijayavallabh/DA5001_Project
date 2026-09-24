"""Rank-0 (n=1) statistics for one generation directory, as a small committed CSV.

Why this exists. feat-136's G0b compared the RAW mean completion length of two arms and failed
`comma7bhb` at +5.3% on a 5% tolerance. The decomposition showed the mean is exactly

    mean_words = (1 - empty_frac) * mean_words_nonempty

and that the entire failure sat in the empty fraction (0.094 -> 0.020) while length given non-empty
moved only -2.7%. Caution (v) had already recorded that an empty generation is EOS at step 0, that
the step-0 logits depend on the padding pattern, and that an aggregate gate on a stratified rate
gates the wrong quantity. feat-138 and feat-140 therefore gate `mean_words_nonempty` and REPORT the
empty fraction separately, which needs both numbers to exist as data rather than as a one-off.

It also writes the per-class empty counts, because caution (v)'s feat-132 finding was that the whole
signal lived in ONE class (neutral, 45/200 -> 24/200, z = 2.78) while the total passed.

Generations are 2.5 GB per arm and live on whichever host produced them; this reduces an arm to one
row so the gate can run anywhere, off committed data.

Usage: python analysis/arm_rank0_stats.py --gen-dir <dir> [--name <name>] --out results
"""
import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_decoding import CLASSES, load_candidates  # noqa: E402


def pairing(gen_dir):
    """(target_model, anchor_model) off the first NON-EMPTY trajectory file.

    Skipping empty files matters: h1.py creates a zero-byte file for every class it is not
    generating, and reading line 1 of one of those raises rather than returning a model id.
    """
    import glob
    import json
    for f in sorted(glob.glob(os.path.join(gen_dir, "trajectories_k0_*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            line = fh.readline()
        if line.strip():
            m = json.loads(line)["metadata"]
            return m.get("target_model"), m.get("anchor_model")
    return None, None


def stats(gen_dir):
    cand = load_candidates(gen_dir)
    if not cand:
        raise SystemExit(f"no candidates under {gen_dir}")
    per_class = {c: [0, 0] for c in CLASSES}          # [empty, total]
    words = []
    for pid, v in sorted(cand.items()):
        seed, cls, _prompt, gen = v[0]                 # rank 0 == the n=1 candidate
        w = len(gen.split())
        words.append(w)
        if cls in per_class:
            per_class[cls][1] += 1
            per_class[cls][0] += (w == 0)
    nz = [w for w in words if w]
    assert words, "no rank-0 completions were read"
    t, a = pairing(gen_dir)
    row = dict(gen_dir=gen_dir, target_model=t, anchor_model=a, n_prompts=len(words),
               mean_words=round(statistics.mean(words), 4),
               empty_frac=round(1 - len(nz) / len(words), 6),
               n_empty=len(words) - len(nz),
               mean_words_nonempty=round(statistics.mean(nz), 4) if nz else 0.0,
               median_words_nonempty=round(statistics.median(nz), 4) if nz else 0.0)
    for c in CLASSES:
        e, n = per_class[c]
        if n:
            row[f"empty_{c}"] = e
            row[f"n_{c}"] = n
    # The identity the whole repair rests on; if it ever fails, one of the three is not what it says.
    if nz:
        recon = (1 - row["empty_frac"]) * row["mean_words_nonempty"]
        assert abs(recon - row["mean_words"]) < 0.02, (
            f"mean_words {row['mean_words']} != (1-empty)*nonempty {recon:.4f}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    row = stats(a.gen_dir)
    name = a.name or os.path.basename(a.gen_dir.rstrip("/"))
    p = os.path.join(a.out, f"arm_rank0_stats_{name}.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row)); w.writeheader(); w.writerow(row)
    print(f"wrote {p}")
    for k, v in row.items():
        print(f"  {k:<24} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
