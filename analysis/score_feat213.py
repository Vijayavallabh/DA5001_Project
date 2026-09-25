"""Score feat-213 (results/onset_prediction_long_outputs.md): every mechanism at T_max = 1000.

Reads the two judge passes (results/matched_h2h_f213_{B,G}.csv and their per-prompt files) and the arms under
output/feat213 for lengths. Writes results/long_outputs.csv (per arm: mean tokens and words, share longer than
200 tokens, empties) and results/long_outputs_scoring.csv (every registered reading, and the descriptive
rows). A length is the true decoded length: the three step counters for a harness arm (caution (ah)), the
served ids for an anchor-alone arm.

Usage (where output/feat213 lives): .venv/bin/python analysis/score_feat213.py [--results results]
"""
import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.order_averaged_h2h import paired_boot  # noqa: E402
from dap.shared import served_generation  # noqa: E402

O = "output/feat213"
CLASSES = ("neutral", "factual", "creative")
ARMS = {"sel_n64": ("sel", "whole1000n64"), "sel_n16": ("sel", "whole1000n16"), "sel_n4": ("sel", "whole1000n4"),
        "sel_n1": ("sel", "whole1000n1"), "inst": ("inst", "blk100n8"), "opp": ("kl", "-1"),
        "anchor_k0": ("kl", "0"), "kl_20.8": ("kl", "0.020794"), "kl_0.1": ("kl", "0.1"), "kl_0.5": ("kl", "0.5"),
        "kl_10": ("kl", "10"), "pw_4.16": ("pw", "0.0041589"), "pw_20.8": ("pw", "0.020794"),
        "front_4.16": ("front", "1e-09"), "win_4.16": ("win", "4.1589")}


def lengths(d, tok):
    """prompt_id -> (true tokens, words) at the lowest seed."""
    best = {}
    for cls in CLASSES:
        for line in open(os.path.join(O, d, f"trajectories_k{tok}_{cls}.jsonl")):
            r = json.loads(line)
            m, a = r["metadata"], r["aggregate"]
            if m["prompt_id"] in best and best[m["prompt_id"]][0] <= m["seed"]:
                continue
            if "steps_active" in a:
                n = (a.get("steps_active") or 0) + (a.get("steps_forced_safe") or 0) + (a.get("steps_risky_unchanged") or 0)
            else:
                n = a["generation_length_tokens"]
            w = len(served_generation(a, r["prefix_analysis"]["prefix_text"]).split())
            best[m["prompt_id"]] = (m["seed"], n, w)
    return {p: v[1:] for p, v in best.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    R = a.results

    L = {n: lengths(*v) for n, v in ARMS.items()}
    rows = []
    for n, d in L.items():
        toks = [x[0] for x in d.values()]
        rows.append(dict(arm=n, prompts=len(d), mean_tokens=round(sum(toks) / len(toks), 1),
                         mean_words=round(sum(x[1] for x in d.values()) / len(d), 1),
                         over_200_tokens_pct=round(100 * sum(t > 200 for t in toks) / len(toks), 1),
                         at_1000_tokens_pct=round(100 * sum(t >= 1000 for t in toks) / len(toks), 1),
                         empty=sum(x[1] == 0 for x in d.values())))
    with open(os.path.join(R, "long_outputs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    out = []

    def add(pid, quantity, v, lo, hi, reading, predicted, right):
        out.append(dict(prediction=pid, quantity=quantity, value=v, lo95=lo, hi95=hi, reading=reading,
                        predicted=predicted, verdict="" if right is None else ("RIGHT" if right else "WRONG")))

    P = {}
    for j in ("B", "G"):
        P[j] = {(r["quantity"], r["arm"]): r for r in csv.DictReader(open(os.path.join(R, f"matched_h2h_f213_{j}.csv")))}

    def get(j, q, arm):
        r = P[j][(q, arm)]
        return float(r["value"]), float(r["lo95"]), float(r["hi95"]), r["reading"]

    reg = [("L1", "gain", "sel_n64 - sel_n1"), ("L2", "difference", "inst - pw_20.8"),
           ("L3", "difference", "inst - kl_20.8"), ("L4", "difference", "sel_n64 - pw_4.16"),
           ("L4", "difference", "sel_n64 - front_4.16"), ("L4", "difference", "sel_n64 - win_4.16"),
           ("L5", "difference", "sel_n64 - kl_0.1"), ("L5", "difference", "sel_n64 - kl_0.5")]
    for pid, q, arm in reg:
        v, lo, hi, rd = get("B", q, arm)
        add(pid, f"B: {arm}", v, lo, hi, rd, "CONFIRMED", rd == "CONFIRMED")
    for pid, q, arm in reg:
        v, lo, hi, rd = get("G", q, arm)
        add("L6", f"G: {arm}", v, lo, hi, rd, "same sign as B", (v > 0) == (get("B", q, arm)[0] > 0))
    for j in ("B", "G"):
        for q, arm in (("gain", "sel_n16 - sel_n1"), ("gain", "sel_n4 - sel_n1"), ("gain", "inst - sel_n1"),
                       ("difference", "inst - sel_n64"), ("difference", "sel_n64 - kl_10"),
                       ("gain", "kl_20.8 - anchor_k0"), ("gain", "kl_0.1 - anchor_k0"), ("gain", "kl_0.5 - anchor_k0"),
                       ("gain", "kl_10 - anchor_k0"), ("gain", "pw_4.16 - anchor_k0"), ("gain", "pw_20.8 - anchor_k0"),
                       ("gain", "front_4.16 - anchor_k0"), ("gain", "win_4.16 - anchor_k0")):
            v, lo, hi, rd = get(j, q, arm)
            add("desc", f"{j}: {arm}", v, lo, hi, rd, "", None)
        pp = {r["prompt_id"]: r for r in csv.DictReader(open(os.path.join(R, f"matched_h2h_per_prompt_f213_{j}.csv")))}
        long_ = [p for p in pp if L["sel_n64"][p][0] > 200]
        xs = [float(pp[p]["u_sel_n64"]) - float(pp[p]["u_sel_n1"]) for p in long_]
        if len(xs) >= 20:
            lo, hi = paired_boot(xs, random.Random(213))
            add("desc", f"{j}: sel_n64 - sel_n1 on prompts whose served n=64 output is over 200 tokens",
                round(sum(xs) / len(xs), 4), round(lo, 4), round(hi, 4), f"n={len(xs)}", "", None)
    with open(os.path.join(R, "long_outputs_scoring.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in rows + out:
        print(r)


if __name__ == "__main__":
    main()
