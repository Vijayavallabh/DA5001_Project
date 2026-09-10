"""What is a higher Renyi order worth, and can a deployer predict it? (plan v5 / feat-074)

feat-073 measured the order comparison at matched utility and found the advantage spans four orders
of magnitude between two pairs: at k = 1, alpha = 2 makes an exact 50-token window 871x less likely
on KL3M-520M and 3.5e4 on Pleias-1.2B. Nothing in the paper predicts which pair gets which, and a
deployer choosing an order has to.

One variable is already visible in the k = 3 row group, where the advantage vanishes:

    F(k) = fraction of the theta = 1 fidelity ceiling the audited alpha = 1 decoder captures at k

F -> 1 is the audited decoder unconstrained, and matching a decoder that is not constrained forces
every other order to a budget where it is not constrained either, so the advantage must go to 1.
That is an anchor the hypothesis cannot dodge, and F needs only the anchor and the risky model --
no protected work, no attack, no judge.

This is a re-analysis of the grids feat-073 already wrote, at no new compute: every grid k is
treated in turn as the published budget, and the matched budget and window factor are recomputed
there. Reads results/order_frontier_*.csv, writes <out>/order_law{,_summary}.csv. No GPU.

Pre-registered in results/onset_prediction_orders_matched.md.

  .venv/bin/python analysis/order_law.py --out results
"""
from __future__ import annotations

import argparse, csv, glob, math, os, re, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.order_frontier import _at, interp  # noqa: E402

LABEL = {"kl3m": "KL3M-520M", "pleias": "Pleias-1.2B", "phi": "Phi-3.5-mini",
         "comma": "Comma-7B", "pleias350": "Pleias-350M", "kl3m17b": "KL3M-1.7B",
         "tinycomma": "TinyComma-1.8B", "kl3m170m": "KL3M-170M", "kl3m37b": "KL3M-3.7B",
         "pleias3b": "Pleias-3B", "llama1b": "Llama-3.2-1B", "llama3b": "Llama-3.2-3B",
         "qwen7b": "Qwen2.5-7B"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="results/order_frontier_*.csv",
                    help="`_bf16` files are skipped: the precision control re-runs a pair that is "
                         "already in the set, and counting it would inflate the collapse test with "
                         "a duplicate")
    ap.add_argument("--out", default="results")
    ap.add_argument("--window", type=float, default=50.0)
    a = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(a.glob)):
        # The precision control re-runs a pair that is already in the set, so it is skipped unless
        # the caller asked for bfloat16 runs explicitly -- which is how the homogeneous seven-pair
        # set is analysed, where the bfloat16 files ARE the pairs.
        if path.endswith("_matched.csv") or ("_bf16" in path and "bf16" not in a.glob):
            continue
        if "_seed" in os.path.basename(path):
            # a seed arm re-runs a pair that is already in the set at another --seed-tokens;
            # counting it would double that pair and make the rank test meaningless
            continue
        tag = re.sub(r"^order_frontier_|_bf16|\.csv$", "", os.path.basename(path))
        pair = LABEL.get(tag, tag)
        cells = list(csv.DictReader(open(path)))
        ntok = float(cells[0]["n_tokens"])
        ks = sorted({float(c["k"]) for c in cells})
        orders = sorted({float(c["alpha"]) for c in cells})
        price = {(float(c["alpha"]), float(c["k"])): float(c["price_nats"]) for c in cells}
        leak = {(float(c["alpha"]), float(c["k"])): float(c["logp_target"]) for c in cells}
        frac = {float(c["k"]): float(c["price_frac_ceiling"])
                for c in cells if float(c["alpha"]) == orders[0]}
        for pk in ks:
            base_f, base_l = price[(orders[0], pk)], leak[(orders[0], pk)]
            for o in orders[1:]:
                kp = interp(ks, [price[(o, k)] for k in ks], base_f)
                if kp is None:                      # this order cannot buy that much on the grid
                    continue
                lp = _at(ks, [leak[(o, k)] for k in ks], kp)
                rows.append(dict(pair=pair, alpha=o, published_k=pk,
                                 F=round(frac[pk], 4), k_matched=round(kp, 4),
                                 nats_per_window=round(a.window * (base_l - lp) / ntok, 4),
                                 window_factor=float(f"{math.exp(min(a.window * (base_l - lp) / ntok, 700)):.3g}"),
                                 log10_factor=round(a.window * (base_l - lp) / ntok / math.log(10), 4)))
    if not rows:
        raise SystemExit(f"no grids matched {a.glob}")
    rows.sort(key=lambda r: (r["pair"], r["alpha"], r["F"]))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "order_law.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    pairs = sorted({r["pair"] for r in rows})
    orders = sorted({r["alpha"] for r in rows})
    print("log10 of how many times less likely an exact 50-token window is, at matched utility,")
    print("against F = the fraction of the unconstrained ceiling the audited decoder already has.\n")
    print(f"{'F':>6s}" + "".join(f"{p[:11]:>13s}" for p in pairs))
    grid = sorted({r["F"] for r in rows})
    for o in orders:
        print(f"\n  alpha = {o:.0f}")
        for f in grid:
            cells = {r["pair"]: r for r in rows if r["alpha"] == o and abs(r["F"] - f) < 1e-9}
            if not cells:
                continue
            print(f"{f:>6.3f}" + "".join(
                (f"{cells[p]['log10_factor']:>13.2f}" if p in cells else f"{'':>13s}")
                for p in pairs))

    # The collapse test. The two pairs' grids do not land on the same F, so the curves are
    # interpolated onto a common F grid inside the range both cover -- requiring exact matches
    # would compare one cell and call it a collapse.
    common = [f for f in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99]
              if all(min(r["F"] for r in rows if r["pair"] == p_) <= f <=
                     max(r["F"] for r in rows if r["pair"] == p_) for p_ in pairs)]
    summary = []
    for o in orders:
        curves = {}
        for p_ in pairs:
            c = sorted((r["F"], r["log10_factor"]) for r in rows
                       if r["pair"] == p_ and r["alpha"] == o)
            if len(c) >= 2:
                curves[p_] = ([x for x, _ in c], [y for _, y in c])
        for f in common:
            v = {p_: _at(xs, ys, f) for p_, (xs, ys) in curves.items()}
            if len(v) >= 2:
                summary.append(dict(alpha=o, F=f, n_pairs=len(v),
                                    log10_lo=round(min(v.values()), 3),
                                    log10_hi=round(max(v.values()), 3),
                                    log10_spread=round(max(v.values()) - min(v.values()), 3)))
    with open(os.path.join(a.out, "order_law_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)

    spread = [s_["log10_spread"] for s_ in summary]
    print(f"\ninterpolated onto a common F grid, {len(summary)} cells:")
    print(f"{'F':>6s}" + "".join(f"{'a=%.0f' % o:>10s}" for o in orders) + f"{'spread':>10s}")
    for f in common:
        cells = {s_["alpha"]: s_ for s_ in summary if abs(s_["F"] - f) < 1e-9}
        if not cells:
            continue
        print(f"{f:>6.2f}" + "".join(
            (f"{cells[o]['log10_spread']:>10.2f}" if o in cells else f"{'':>10s}") for o in orders))
    print(f"\nthe pairs differ by a median of {st.median(spread):.2f} decades at matched F "
          f"(range {min(spread):.2f}-{max(spread):.2f})")
    # Would a single per-pair offset collapse them? That is the weaker, one-parameter hypothesis:
    # a common shape in F with one constant per pair. It holds only if the range between the pairs
    # is itself constant in F, so that is what is reported -- not a fitted offset, which on three
    # pairs would be three parameters for three curves and would fit anything.
    for o in orders:
        rng = [c["log10_spread"] for c in summary if c["alpha"] == o]
        if len(rng) >= 3:
            print(f"  alpha = {o:.0f}: pairs span {st.mean(rng):.2f} decades on average, "
                  f"sd {st.stdev(rng):.2f} across F -- a common shape would give sd 0")
    print(f"\nthe anchor point: as F -> 1 the factor must go to 1, i.e. log10 -> 0. "
          f"max |log10| at F >= 0.99 is "
          f"{max(abs(r['log10_factor']) for r in rows if r['F'] >= 0.99):.2f}")
    print("\nThe committed band: within one decade at matched F, and going to 1 as F -> 1, is a law.")
    print(f"wrote {os.path.join(a.out, 'order_law.csv')} and _summary.csv")


if __name__ == "__main__":
    main()
