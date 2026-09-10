"""Is the order's matched-utility advantage a property of the pair, or of the seed? (feat-078)

Every matched-utility number in this paper is at `--seed-tokens 20`, the seed the attack uses, and
Section 4 shows the seed is not innocuous: it splits the seven pairs into two groups with no overlap
on the onset ratio. If it also moves the advantage, then "a stable property of the pair" is wrong
and the finding belongs to one evaluation choice.

This compares `order_seedarm_<pair>_s<seed>_bf16_matched.csv` against that pair's committed
`order_frontier_<pair>_bf16_matched.csv` at seed 20, cell by cell, against the same noise floor the
crossing test uses: that pair's own bfloat16-against-float32 spread. What matters most is not the
levels but whether the *ranking* of the pairs survives, because the ranking is what every predictor
test consumes.

Writes <out>/order_seed.csv. No GPU.

  .venv/bin/python analysis/order_seed.py --out results
"""
from __future__ import annotations

import argparse, csv, glob, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.order_crossings import noise_floor  # noqa: E402
from analysis.order_law import LABEL  # noqa: E402


def load(path):
    return {(float(x["alpha"]), float(x["published_k"])): float(x["nats_per_window"])
            for x in csv.DictReader(open(path))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="results/order_seedarm_*_bf16_matched.csv")
    ap.add_argument("--control-seed", type=int, default=20)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(a.glob)):
        m = re.match(r"order_seedarm_(.+)_s(\d+)_bf16_matched\.csv$", os.path.basename(path))
        if not m:
            continue
        tag, seed = m.group(1), int(m.group(2))
        ctrl = f"results/order_frontier_{tag}_bf16_matched.csv"
        if not os.path.exists(ctrl):
            print(f"  SKIP {tag} s{seed}: no seed-{a.control_seed} control at {ctrl}")
            continue
        arm, base = load(path), load(ctrl)
        floor = noise_floor(tag, 50.0)
        for (al, k) in sorted(set(arm) & set(base)):
            if al == 1.0:
                continue
            d = arm[(al, k)] - base[(al, k)]
            rows.append(dict(pair=LABEL.get(tag, tag), seed=seed, control_seed=a.control_seed,
                             alpha=al, published_k=k,
                             arm=round(arm[(al, k)], 3), control=round(base[(al, k)], 3),
                             diff=round(d, 3), noise_floor=round(floor, 3) if floor else "",
                             beyond_floor=bool(floor and abs(d) > floor),
                             sign_flip=(arm[(al, k)] > 0) != (base[(al, k)] > 0)))
    if not rows:
        raise SystemExit(f"no seed arms matched {a.glob}")

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "order_seed.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print(f"matched-utility advantage against seed length, nats per 50-token window "
          f"(control = seed {a.control_seed})\n")
    print(f"{'pair':16s}{'seed':>5s}{'k':>4s}{'alpha':>6s}{'arm':>9s}{'control':>9s}"
          f"{'diff':>8s}{'floor':>7s}  beyond floor / sign flip")
    for r in rows:
        print(f"{r['pair'][:15]:16s}{r['seed']:>5d}{r['published_k']:>4.0f}{r['alpha']:>6.0f}"
              f"{r['arm']:>9.2f}{r['control']:>9.2f}{r['diff']:>+8.2f}"
              f"{(r['noise_floor'] if r['noise_floor'] != '' else 0):>7.2f}"
              f"   {'YES' if r['beyond_floor'] else 'no':4s}"
              f"  {'SIGN FLIP' if r['sign_flip'] else ''}")
    beyond = [r for r in rows if r["beyond_floor"]]
    flips = [r for r in rows if r["sign_flip"]]
    big = [r for r in rows if abs(r["diff"]) > 2.303]        # one decade
    print(f"\n{len(beyond)} of {len(rows)} cells move beyond their pair's precision floor, "
          f"{len(big)} by more than a decade, {len(flips)} change sign.")
    print("The committed bands: under the floor everywhere is a property of the pair; more than a")
    print("decade makes the finding seed-dependent; in between is a second-order sensitivity whose")
    print("range is reported and whose pair RANKING is what must be checked.")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
