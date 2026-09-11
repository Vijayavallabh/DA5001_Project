"""How much of the cross-judge gain is a property of the responses rather than of their order?

analysis/selection_crossjudge.py shows each pair to judge B in ONE randomly chosen order, which
makes the estimate unbiased but says nothing about how stable a single verdict is. A pairwise LLM
judge carries a position bias, and a gain of 0.08 on a 0/0.5/1 scale is small enough that an
instrument which flips when the two responses swap sides could manufacture it.

This re-judges the identical items -- same candidates, same picks, same prompts -- in BOTH orders,
and reports (C1) how often the two orders agree, (C2) how often the judge takes the first slot, and
(C3) the n=8 minus n=1 gain re-estimated on the order-averaged utility. Bands were committed in
results/onset_prediction_judge_consistency.md before this ran.

It is NOT a human preference study and must not be described as one; it bounds the instrument, and
the human study stays named future work.

No generation: a second scoring pass over text already on disk.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/judge_consistency.py --out results
"""
import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_decoding import (VERDICT_U, boot_mean, load_baseline,  # noqa: E402
                                         load_candidates)
from analysis.utility import judge_batch  # noqa: E402


def u_of(verdict, arm_is_a):
    """Utility for the selection arm: 1 win, 0.5 tie, 0 loss, given which slot it occupied."""
    if verdict == "Tie":
        return 0.5
    won = (verdict == "A") if arm_is_a else (verdict == "B")
    return 1.0 if won else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default="results/selection_candidates.csv")
    ap.add_argument("--gen-dir", default="output/phase5/sel_anchor8")
    ap.add_argument("--baseline-dir", default="output/sweep_plain")
    ap.add_argument("--judge-b", default="microsoft/Phi-3.5-mini-instruct")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--recorded-gain", type=float, default=0.081,
                    help="the single-order gain on record, for the C3 band")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    by = {}
    for r in csv.DictReader(open(a.candidates)):
        by.setdefault(r["prompt_id"], []).append(r)
    for v in by.values():
        v.sort(key=lambda r: int(r["rank"]))
    cands, base = load_candidates(a.gen_dir), load_baseline(a.baseline_dir)
    pids = sorted(p for p in by if p in cands and p in base)
    print(f"[jc] {len(pids)} prompts", flush=True)

    # identical picks to selection_crossjudge.py: judge A's verdict, ties by per-token likelihood
    picks = {}
    for p in pids:
        v = by[p][:a.n]
        picks[(p, 1)] = 0
        picks[(p, a.n)] = max(range(len(v)),
                              key=lambda i: (VERDICT_U[v[i]["outcome"]],
                                             float(v[i]["logp_per_token"])))

    tok = AutoTokenizer.from_pretrained(a.judge_b, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.judge_b,
                                                 torch_dtype=getattr(torch, a.dtype)).cuda().eval()

    keys = [(p, n) for p in pids for n in (1, a.n)]
    # order 1: the arm is A. order 2: the arm is B. Same pair of texts, swapped slots.
    fwd, rev = [], []
    for p, n in keys:
        _, _, prompt, gen = cands[p][picks[(p, n)]]
        fwd.append((prompt, gen, base[p]))
        rev.append((prompt, base[p], gen))

    def judge_all(items, tag):
        out = []
        for i in range(0, len(items), 200):
            out += judge_batch(model, tok, items[i:i + 200], "cuda")
            print(f"[jc] {tag} judged {len(out)}/{len(items)}", flush=True)
        return out

    v1, v2 = judge_all(fwd, "arm-first"), judge_all(rev, "arm-second")

    # C1: the two orders name the same response
    def same(x, y):
        return (x == "A" and y == "B") or (x == "B" and y == "A") or (x == "Tie" and y == "Tie")
    consistent = [same(x, y) for x, y in zip(v1, v2)]
    c1 = sum(consistent) / len(consistent)

    # C2: how often the FIRST slot wins, pooled over both orders (a tie counts as neither)
    firsts = sum(1 for v in v1 + v2 if v == "A")
    decided = sum(1 for v in v1 + v2 if v in ("A", "B"))
    c2 = firsts / decided

    # C3: order-averaged utility per item, paired by prompt
    per = {}
    for (p, n), x, y, ok in zip(keys, v1, v2, consistent):
        per.setdefault(p, {})[n] = dict(u=0.5 * (u_of(x, True) + u_of(y, False)),
                                        u_fwd=u_of(x, True), u_rev=u_of(y, False), consistent=ok)
    order = sorted(per)
    diffs = [per[p][a.n]["u"] - per[p][1]["u"] for p in order]
    gain = sum(diffs) / len(diffs)
    d_lo, d_hi = boot_mean(diffs, rng)

    # secondary, pre-registered as secondary: the same gain on items both arms judged consistently
    keep = [p for p in order if per[p][1]["consistent"] and per[p][a.n]["consistent"]]
    sub = [per[p][a.n]["u"] - per[p][1]["u"] for p in keep]
    s_lo, s_hi = boot_mean(sub, rng) if len(sub) > 20 else (float("nan"), float("nan"))

    c1_read = "STABLE" if c1 >= 0.70 else "NOISY" if c1 >= 0.50 else "UNUSABLE"
    bias = abs(c2 - 0.5)
    c2_read = "BALANCED" if bias <= 0.05 else "MILD" if bias <= 0.15 else "STRONG"
    if d_lo <= 0 <= d_hi:
        c3_read = "DISSOLVES"
    elif abs(gain - a.recorded_gain) <= 0.05:
        c3_read = "SURVIVES"
    else:
        c3_read = "ATTENUATED" if gain < a.recorded_gain else "LARGER under order-averaging"

    rows = [
        dict(criterion="C1 order consistency", value=round(c1, 4), n=len(consistent),
             lo95="", hi95="", reading=c1_read),
        dict(criterion="C2 first-slot win rate", value=round(c2, 4), n=decided,
             lo95="", hi95="", reading=c2_read),
        dict(criterion="C3 gain, order-averaged", value=round(gain, 4), n=len(order),
             lo95=round(d_lo, 4), hi95=round(d_hi, 4), reading=c3_read),
        dict(criterion="C3 secondary, consistent items only", value=round(sum(sub) / len(sub), 4)
             if sub else "", n=len(sub), lo95=round(s_lo, 4), hi95=round(s_hi, 4),
             reading="secondary; conditions on an outcome"),
    ]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "judge_consistency.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(a.out, "judge_consistency_per_prompt.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["prompt_id", "u_n1", "u_n1_fwd", "u_n1_rev", "n1_consistent",
                    f"u_n{a.n}", f"u_n{a.n}_fwd", f"u_n{a.n}_rev", f"n{a.n}_consistent", "diff"])
        for p in order:
            x, y = per[p][1], per[p][a.n]
            w.writerow([p, x["u"], x["u_fwd"], x["u_rev"], x["consistent"],
                        y["u"], y["u_fwd"], y["u_rev"], y["consistent"], round(y["u"] - x["u"], 4)])

    for r in rows:
        print(f"  {r['criterion']:38s} {r['value']}  n={r['n']}  {r['reading']}")
    print(f"  gain {gain:+.4f} [{d_lo:+.4f}, {d_hi:+.4f}] against {a.recorded_gain:+.4f} on record")


if __name__ == "__main__":
    main()
