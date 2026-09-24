"""An interval on the introduction's "win by 2.6x": selection's order-averaged gain over the meter's.

One referee report asked whether the ratio survives its own uncertainty, since its denominator's
interval, [+0.014, +0.0655], comes close to zero. Same construction as analysis/lift_ratio_interval.py
(feat-150): resample prompts, recompute both gains on each resample so their correlation is carried
exactly, and report the fraction of resamples on which the ratio is undefined (a non-positive
denominator) rather than conditioning on the ones that behaved (caution (g)). Fieller, from the same
resamples' moments, is printed beside it for comparison only.

Reads results/order_averaged_h2h_per_prompt.csv (the pass behind +0.1045 and +0.0400); no GPU.
Usage:  .venv/bin/python analysis/h2h_ratio_interval.py --out results
"""
import argparse
import csv
import math
import os
import random


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-prompt", default="results/order_averaged_h2h_per_prompt.csv")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=8801)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.per_prompt, encoding="utf-8")))
    s = [float(r["gain_sel"]) for r in rows]
    m = [float(r["gain_metered"]) for r in rows]
    n = len(rows)
    gs, gm = sum(s) / n, sum(m) / n
    rng = random.Random(a.seed)
    bs, bm, ratios = [], [], []
    for _ in range(a.reps):
        idx = [rng.randrange(n) for _ in range(n)]
        x, y = sum(s[i] for i in idx) / n, sum(m[i] for i in idx) / n
        bs.append(x)
        bm.append(y)
        if y > 0:
            ratios.append(x / y)
    ratios.sort()
    undef = 1.0 - len(ratios) / a.reps
    lo, hi = ratios[int(0.025 * len(ratios))], ratios[int(0.975 * len(ratios))]

    def msd(v):
        mu = sum(v) / len(v)
        return mu, math.sqrt(sum((w - mu) ** 2 for w in v) / (len(v) - 1))
    ms, ss = msd(bs)
    mm, sm = msd(bm)
    cov = sum((x - ms) * (y - mm) for x, y in zip(bs, bm)) / (len(bs) - 1)
    t = 1.96
    den = gm ** 2 - t * t * sm ** 2
    disc = (gs * gm - t * t * cov) ** 2 - den * (gs ** 2 - t * t * ss ** 2)
    f_lo = f_hi = float("nan")
    if disc > 0 and den > 0:
        f_lo = (gs * gm - t * t * cov - math.sqrt(disc)) / den
        f_hi = (gs * gm - t * t * cov + math.sqrt(disc)) / den

    out = dict(numerator="selection n=64, order-averaged gain", denominator="metered k=10",
               n_prompts=n, gain_numerator=round(gs, 6), gain_denominator=round(gm, 6),
               ratio=round(gs / gm, 6), ratio_lo95=round(lo, 6), ratio_hi95=round(hi, 6),
               undefined_frac=round(undef, 4), corr_of_gains=round(cov / (ss * sm), 4),
               fieller_lo95="" if math.isnan(f_lo) else round(f_lo, 6),
               fieller_hi95="" if math.isnan(f_hi) else round(f_hi, 6),
               excludes_one="yes" if lo > 1 else "no")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "h2h_ratio_interval.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out))
        w.writeheader()
        w.writerow(out)
    print(f"[h2h-ratio] {gs:+.4f} / {gm:+.4f} = {gs / gm:.3f} [{lo:.3f}, {hi:.3f}] over {n} prompts;"
          f" undefined on {100 * undef:.2f}% of resamples; gain corr {cov / (ss * sm):+.3f};"
          f" Fieller [{f_lo:.3f}, {f_hi:.3f}]")
    print(f"[h2h-ratio] wrote {path}")


if __name__ == "__main__":
    main()
