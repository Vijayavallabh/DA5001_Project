"""Does the derived onset law hold on a corpus it has never seen? (plan v5 / feat-082)

Every onset in this paper is measured on sixteen English genre novels, and Limitations says so.
Eq. (eq:req) -- onset/s(x) = 1 - s_r/s_s -- is parameter-free and needs no decoding, so a second
protected corpus admits the one test the appendix cannot run: the prediction written down before
the measurement exists. It was, in results/onset_prediction_gutenberg.md, together with the bands
and the k-grid, before any of these sweeps decoded a token.

The scope is narrow and was narrowed in that file before the numbers landed. The paper ALREADY
rejects Eq. (eq:req) as a predictor: the rank correlation between its predicted ratio and the
measured one is -0.18 over seven CopyBench pairs, where the derivation requires it positive. What
was never rejected is the LEVEL -- on the three anchors used here it puts the onset within about
14%. This asks whether that level agreement belongs to the geometry or to those sixteen novels. A
pass does not reinstate the equation as a predictor, and with n = 3 the smallest attainable exact
p is 1/3, so a correct ranking is reported with its p and given no weight.

Reads the three composition sweeps and results/onset_theory_gutenberg.csv, writes
<out>/onset_gutenberg.csv. No GPU.

  .venv/bin/python analysis/onset_gutenberg.py --out results
"""
from __future__ import annotations

import argparse, csv, itertools, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.onset import crossing, curve  # noqa: E402

# (label in onset_theory_gutenberg.csv, sweep directory, its CopyBench twin in onset.csv)
PAIRS = [
    ("KL3M-520M + mem. KL3M-520M (Gutenberg)", "output/phase5/fineg_kl3m520m",
     "KL3M-520M + mem. KL3M-520M"),
    ("Pleias-1.2B + mem. Pleias-1.2B (Gutenberg)", "output/phase5/fineg_pleias12b",
     "Pleias-1.2B + mem. Pleias-1.2B"),
    ("Phi-3.5-mini + mem. Phi-3.5-mini (Gutenberg)", "output/phase5/fineg_phi35",
     "Phi-3.5-mini + mem. Phi-3.5-mini"),
]
# committed in results/onset_prediction_gutenberg.md before any sweep ran
BAND_TIGHT, BAND_WIDE, ENTRY_GATE = (0.85, 1.15), (0.7, 1.4), 0.10


def baseline(path, k):
    """The k = -1 (risky alone) or k = 0 (anchor alone) arm, which every sweep must carry."""
    for r in csv.DictReader(open(path)):
        if r["mode"] == "single" and r["L"] == "0" and float(r["k"]) == k:
            return float(r["nv_recall_mean"])
    return None


# Average ranks, so ties are handled: the value-keyed version this used to carry collapsed tied
# entries onto one rank and read -0.968 where the tie-aware one reads -0.958 on the seed-words
# table. No series scored here has a tie, so nothing measured moves.
from analysis.seed_effect import spearman  # noqa: E402


def exact_p(a, b):
    """Two-sided permutation p over all orderings of b. n = 3 here, so the floor is 1/3."""
    obs = abs(spearman(a, b))
    perms = list(itertools.permutations(b))
    return sum(1 for q in perms if abs(spearman(a, list(q))) >= obs - 1e-12) / len(perms)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--theory", default="results/onset_theory_gutenberg.csv")
    ap.add_argument("--onset", default="results/onset.csv", help="the CopyBench twins")
    ap.add_argument("--thresh", type=float, default=0.01)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    th = {r["pair"]: r for r in csv.DictReader(open(a.theory))}
    cb = {r["pair"]: r for r in csv.DictReader(open(a.onset)) if r["mode"] == "single"}

    rows = []
    for label, run, twin in PAIRS:
        path = os.path.join(run, "composition_summary.csv")
        if not os.path.exists(path):
            print(f"[og] no sweep at {path}, skipping {label}", file=sys.stderr)
            continue
        c = curve(path, "single", 0)
        lo, hi, est = crossing(c, a.thresh)
        t = th[label]
        pred, s_s = float(t["pred_onset_median"]), float(t["s_safe_median"])
        gate = baseline(path, -1.0)
        row = dict(
            pair=label, n_grid=len(c),
            k_minus1_recall=gate, k_zero_recall=baseline(path, 0.0),
            entered=(gate is not None and gate >= ENTRY_GATE),
            s_safe=round(s_s, 4), s_risky=round(float(t["s_risky_median"]), 4),
            onset_lo=lo, onset_hi=hi,
            onset=round(est, 4) if est else None,
            ratio=round(est / s_s, 4) if est else None,
            pred_onset=round(pred, 4), pred_ratio=round(float(t["pred_ratio_median"]), 4),
            req_q25=round(float(t["req_q25"]), 4), req_q10=round(float(t["req_q10"]), 4),
            req_median=round(float(t["req_median"]), 4),
            pred_over_meas=round(pred / est, 4) if est else None,
            copybench_ratio=round(float(cb[twin]["onset_est_over_s"]), 4) if twin in cb else None,
            copybench_onset=round(float(cb[twin]["onset_est"]), 4) if twin in cb else None,
            grid=" ".join(f"{k:g}:{v:.3f}" for k, v in c.items()))
        # P2: the population onset should sit at a low quantile of r(x), not at its median
        row["p2_between_q10_and_median"] = (
            est is not None and row["req_q10"] <= est <= row["req_median"])
        rows.append(row)

    if not rows:
        print("[og] nothing to score", file=sys.stderr)
        return 1
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "onset_gutenberg.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print("The derived onset on a corpus the law has never seen. Bands committed in")
    print("results/onset_prediction_gutenberg.md before any of these sweeps decoded a token.\n")
    print(f"{'pair':30s}{'k=-1':>7s}{'onset':>9s}{'bracket':>14s}{'ratio':>8s}"
          f"{'pred':>8s}{'pred/meas':>11s}{'CopyBench':>11s}")
    for r in rows:
        br = f"({r['onset_lo']},{r['onset_hi']}]" if r["onset_lo"] else "--"
        m = r["pred_over_meas"]
        print(f"{r['pair'].replace(' (Gutenberg)','')[:29]:30s}{r['k_minus1_recall']:>7.3f}"
              f"{r['onset']:>9.3f}{br:>14s}{r['ratio']:>8.3f}{r['pred_onset']:>8.3f}"
              f"{m:>11.3f}{(r['copybench_ratio'] or 0):>11.3f}")
    ok = [r for r in rows if r["entered"] and r["pred_over_meas"]]
    if not ok:
        print("\nno pair passed the entry gate")
        return 0
    m = [r["pred_over_meas"] for r in ok]
    tight = all(BAND_TIGHT[0] <= x <= BAND_TIGHT[1] for x in m)
    wide = all(BAND_WIDE[0] <= x <= BAND_WIDE[1] for x in m)
    verdict = ("TRANSFERS: every pred/meas inside the committed [0.85, 1.15]" if tight else
               "MAGNITUDE ONLY: inside [0.7, 1.4], outside [0.85, 1.15]" if wide else
               "CORPUS-SPECIFIC: at least one pair outside [0.7, 1.4]")
    print(f"\npred/meas {min(m):.3f} to {max(m):.3f} over {len(ok)} pairs -> {verdict}")
    print("P2 (onset between q10 and the median of r(x)): "
          + ", ".join(f"{r['pair'].split(' ')[0]} {'yes' if r['p2_between_q10_and_median'] else 'NO'}"
                      for r in ok))
    if len(ok) >= 3:
        pr = [r["pred_ratio"] for r in ok]
        me = [r["ratio"] for r in ok]
        print(f"direction (reported, not weighted): rho = {spearman(pr, me):+.2f}, "
              f"exact p = {exact_p(pr, me):.3f} -- the floor at n = {len(ok)} is {1/6 if len(ok)==3 else 0:.3f}, "
              "and the seven-pair CopyBench test that refuted the direction stands")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
