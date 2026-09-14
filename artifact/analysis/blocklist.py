"""The baseline the paper argues against and never measured: an n-gram blocklist.

Section 5 contrasts selection anchoring with "an n-gram blocklist, the other cheap inference-time
defence" and cites Ippolito et al. (2023) -- but no blocklist has ever been run here, so the
contrast is an assertion. This runs one, on text already generated, so it costs no GPU.

MemFree-style rule: emit greedily from the risky model, and at each step forbid any token that
would complete an n-gram already present in the protected corpus. Applied POST HOC to a completed
generation the rule is equivalent to asking what fraction of the generation survives -- so this
measures, on the unconstrained risky model's own outputs:

  recall_blocked   near-verbatim recall after every blocked n-gram is broken
  utility_cost     how much of an ordinary answer the same filter would have rewritten

The point of the comparison is not that a blocklist works badly. It is that its guarantee names the
works it was given, so it says nothing about a work nobody listed, about paraphrase, or about an
adversary who supplies the prefix -- while log n bounds all three at once. Measuring it makes that
argument evidence instead of assertion.

Reads:  a generation directory (h1.py output) and the protected corpus it should be filtered against
Writes: <out>/blocklist.csv

Usage:
  .venv/bin/python analysis/blocklist.py --gen-dir output/sweep_plain --out results
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dap.stats import copying_metrics  # noqa: E402

WORD = re.compile(r"\w+|[^\w\s]")


def toks(s):
    return WORD.findall(s or "")


def ngrams(ws, n):
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def build_index(paths, n):
    """Every n-gram of every protected passage -- the blocklist a deployer would ship."""
    idx = set()
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            for field in ("reference_text", "raw_text"):
                idx |= ngrams(toks(r.get(field) or ""), n)
    return idx


def filter_text(text, idx, n):
    """MemFree: scan left to right and drop any token that would complete a blocked n-gram.

    Equivalent to the decode-time rule on the realised sequence, because the rule is causal: the
    decision at step t depends only on the n-1 tokens before it."""
    ws, out, blocked = toks(text), [], 0
    for w in ws:
        if len(out) >= n - 1 and tuple(out[-(n - 1):] + [w]) in idx:
            blocked += 1
            continue
        out.append(w)
    return " ".join(out), blocked, len(ws)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-dir", default="output/sweep_plain")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n", type=int, nargs="+", default=[6, 8, 10],
                    help="blocklist n-gram orders; Ippolito et al. use 10")
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    protected = sorted(glob.glob(os.path.join(a.data_dir, "copybench_*.jsonl")))
    refs = {}
    for p in protected:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            refs[r["prompt_id"]] = r.get("reference_text") or ""
    print(f"[blocklist] {len(refs)} protected passages from {len(protected)} files")

    rows = []
    for n in a.n:
        idx = build_index(protected, n)
        print(f"[blocklist] n={n}: {len(idx)} blocked n-grams", flush=True)
        for kind, pat in (("protected", "trajectories_k*_attack_train.jsonl"),
                          ("ordinary", "trajectories_k*_neutral.jsonl")):
            for path in sorted(glob.glob(os.path.join(a.gen_dir, pat))):
                k = os.path.basename(path).split("_")[1][1:]
                pre, post, blk, tot, m = [], [], 0, 0, 0
                for line in open(path, encoding="utf-8"):
                    r = json.loads(line)
                    gen = r["aggregate"]["generation"]
                    ref = refs.get(r["metadata"]["prompt_id"], "")
                    f, b, t = filter_text(gen, idx, n)
                    blk += b; tot += t; m += 1
                    if kind == "protected" and ref:
                        pre.append(copying_metrics(gen, ref).get("nv_recall", 0.0))
                        post.append(copying_metrics(f, ref).get("nv_recall", 0.0))
                rec = dict(n_gram=n, prompt_class=kind, k=k, n_trajectories=m,
                           tokens_blocked_pct=round(100 * blk / max(1, tot), 3),
                           recall_before=round(sum(pre) / len(pre), 4) if pre else "",
                           recall_after=round(sum(post) / len(post), 4) if post else "",
                           protected_passages=len(refs), blocked_ngrams=len(idx),
                           corpus=os.path.basename(a.data_dir.rstrip("/")) or "data")
                rows.append(rec)
                print(f"  n={n} {kind:10s} k={k:>5s}  blocked {rec['tokens_blocked_pct']:6.3f}% "
                      f"of tokens  recall {rec['recall_before']} -> {rec['recall_after']}",
                      flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"blocklist{a.tag}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {os.path.join(a.out, f'blocklist{a.tag}.csv')} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
