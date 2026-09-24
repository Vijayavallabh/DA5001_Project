"""Served lengths, a length-controlled head-to-head, and the non-empty restriction (post hoc).

A referee asks whether the headline difference survives length control and whether it survives
restricting to non-empty completions. Neither was registered, so everything here is labelled post
hoc in results/h2h_length_control_note.md; nothing new is judged. It reads the de-echoed headline
pass (feat-184 Part A, results/order_averaged_h2h_per_prompt_deecho.csv) and the same texts that
pass judged, recovered with dap.shared.served_generation (caution (bc)).

Length control, in the spirit of length-controlled AlpacaEval (Dubois et al. 2024): regress each
order-averaged score u (arm against the fixed opponent) on arm indicators plus
beta * tanh(dlen / sd), dlen the arm's word count minus the opponent's; the arm's coefficient is its
score at equal length. The difference of gains is then recomputed from the four coefficients.
Intervals resample prompts and refit.

Usage: .venv/bin/python analysis/h2h_length_control.py --out results
"""
import argparse
import csv
import math
import os
import random
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.order_averaged_h2h import lowest_seed  # noqa: E402
from analysis.selection_decoding import load_baseline, load_candidates  # noqa: E402
from analysis.selection_scaling import load_rewards  # noqa: E402
from analysis.utility import load_arm  # noqa: E402

ARMS = ("sel_n64", "sel_n1", "metered_k10", "anchor_k0")
N_BOOT = 2000


def texts(n=64):
    opp = load_baseline("output/sweep_plain", deecho=True)
    cands = load_candidates("output/phase5/sel_anchor64", deecho=True)
    rewards = load_rewards("results/selection_rewards64.csv")
    met = lowest_seed(load_arm("output/phase2/conc_all", 10.0, "kl", deecho=True))
    anc = lowest_seed(load_arm("output/sweep_plain", 0.0, "kl", deecho=True))
    out = {}
    for p in opp:
        if p not in cands or p not in rewards or p not in met or p not in anc:
            continue
        r = rewards[p][:n]
        best = max(range(len(r)), key=lambda i: r[i])
        out[p] = {"sel_n64": cands[p][best][3], "sel_n1": cands[p][0][3],
                  "metered_k10": met[p][1], "anchor_k0": anc[p][1], "opp": opp[p]}
    return out


def fit(rows, sd):
    """rows: [(arm_index, dlen, u)] -> per-arm score at equal length (4 values)."""
    X = np.zeros((len(rows), 5))
    y = np.zeros(len(rows))
    for i, (a, dl, u) in enumerate(rows):
        X[i, a] = 1.0
        X[i, 4] = math.tanh(dl / sd)
        y[i] = u
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def diff_of_gains(v):
    return (v[0] - v[1]) - (v[2] - v[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-prompt", default="results/order_averaged_h2h_per_prompt_deecho.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=4)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    T = texts()
    U = {r["prompt_id"]: r for r in csv.DictReader(open(a.per_prompt, encoding="utf-8"))}
    pids = sorted(set(T) & set(U))
    assert len(pids) == 500, len(pids)
    words = {p: {k: len(v.split()) for k, v in T[p].items()} for p in pids}
    dl = {p: {arm: words[p][arm] - words[p]["opp"] for arm in ARMS} for p in pids}
    sd = st.pstdev([dl[p][arm] for p in pids for arm in ARMS])

    out = []
    for arm in ARMS + ("opp",):
        w = [words[p][arm] for p in pids]
        q = np.percentile(w, [25, 50, 75])
        out.append(dict(quantity=f"served words, {arm}", value=float(q[1]), lo95=float(q[0]),
                        hi95=float(q[2]), n=len(pids),
                        note=f"median [IQR]; empty on {sum(x == 0 for x in w)} prompts"))

    def stats(sample):
        rows = [(i, dl[p][arm], float(U[p][f"u_{arm}"])) for p in sample for i, arm in enumerate(ARMS)]
        raw = [np.mean([float(U[p][f"u_{arm}"]) for p in sample]) for arm in ARMS]
        coef = fit(rows, sd)
        return diff_of_gains(raw), diff_of_gains(coef[:4]), coef[4]

    raw, lc, beta = stats(pids)
    boots = [stats([pids[rng.randrange(len(pids))] for _ in pids]) for _ in range(N_BOOT)]
    ci = lambda xs: (float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5)))  # noqa: E731
    lo_r, hi_r = ci([b[0] for b in boots])
    lo_l, hi_l = ci([b[1] for b in boots])
    lo_b, hi_b = ci([b[2] for b in boots])
    out.append(dict(quantity="D3, as judged (de-echoed pass)", value=raw, lo95=lo_r, hi95=hi_r,
                    n=len(pids), note="difference of gains; reproduces the pass's own D3"))
    out.append(dict(quantity="D3, length-controlled", value=lc, lo95=lo_l, hi95=hi_l, n=len(pids),
                    note=f"u ~ arm + beta*tanh(dlen/{sd:.1f}); prompts resampled, refit"))
    out.append(dict(quantity="length coefficient beta", value=beta, lo95=lo_b, hi95=hi_b, n=len(pids),
                    note="score change from a much shorter to a much longer answer, per unit tanh"))

    # the non-empty restriction: both controls' anchor draws non-empty
    ne = [p for p in pids if words[p]["sel_n1"] > 0 and words[p]["anchor_k0"] > 0]
    d = [(float(U[p]["u_sel_n64"]) - float(U[p]["u_sel_n1"])) -
         (float(U[p]["u_metered_k10"]) - float(U[p]["u_anchor_k0"])) for p in ne]
    bs = sorted(np.mean([d[rng.randrange(len(d))] for _ in d]) for _ in range(N_BOOT))
    out.append(dict(quantity="D3 on prompts whose two anchor controls are non-empty",
                    value=float(np.mean(d)), lo95=float(bs[int(0.025 * N_BOOT)]),
                    hi95=float(bs[int(0.975 * N_BOOT)]), n=len(ne),
                    note=f"{len(pids) - len(ne)} prompts dropped"))

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "h2h_length_control.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["quantity", "value", "lo95", "hi95", "n", "note"])
        w.writeheader()
        for r in out:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    for r in out:
        print(f"  {r['quantity']:58s} {r['value']:+.4f} [{r['lo95']:+.4f}, {r['hi95']:+.4f}]  n={r['n']}  {r['note']}")
    print("wrote", path)


if __name__ == "__main__":
    main()
