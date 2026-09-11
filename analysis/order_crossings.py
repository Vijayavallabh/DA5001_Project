"""Do the Renyi orders trace one frontier, or do their curves cross? (plan v5 / feat-077)

feat-073 compared the orders at matched utility by interpolating, for each order, the budget at
which it buys what the audited KL decoder buys at some published `k`. That is the deployer's
question, but it hides a simpler one that needs no interpolation of a budget at all. Each order
traces a curve in the plane the mechanism actually trades in:

    x = fidelity bought,  G(theta) = -D_KL(p_r || p_theta) + const   (a bounded average)
    y = leakage,          L = sum_t log p_theta(x_t)                 (a rare event)

If the four orders traced one frontier, matching x would match y and Table 1's ranking would be an
artefact of comparing at matched budget. If the curves *cross*, no order dominates and which is
safer depends on where the deployer is operating -- which cannot be read off the published budget.

Both readings are stated as a sign test on `L(alpha) - L(1)` swept across the fidelity range every
order covers, in nats per 50-token window. A sign only counts when it clears a **noise floor** taken
from the paper's own precision control: for a pair run in both bfloat16 and float32, that pair's
largest measured |bfloat16 - float32| difference; for a pair run in one precision only, the largest
such difference over the pairs that have a twin, which is the conservative choice.

Writes <out>/order_crossings.csv. No GPU.

  .venv/bin/python analysis/order_crossings.py --out results
"""
from __future__ import annotations

import argparse, csv, glob, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.order_frontier import _at  # noqa: E402
from analysis.order_law import LABEL  # noqa: E402


def curves(path):
    """{alpha: [(fidelity, logp_target), ...] sorted}, plus the token count."""
    cells = list(csv.DictReader(open(path)))
    out = {}
    for c in cells:
        out.setdefault(float(c["alpha"]), []).append(
            (float(c["price_nats"]), float(c["logp_target"])))
    for v in out.values():
        v.sort()
    return out, float(cells[0]["n_tokens"])


def noise_floor(tag, window):
    """That pair's own bfloat16-against-float32 spread, in nats per window, or None."""
    a = f"results/order_frontier_{tag}_bf16_matched.csv"
    b = f"results/order_frontier_{tag}_matched.csv"
    if not (os.path.exists(a) and os.path.exists(b)):
        return None
    load = lambda p: {(float(x["alpha"]), float(x["published_k"])): float(x["nats_per_window"])
                      for x in csv.DictReader(open(p))}
    A, B = load(a), load(b)
    return max(abs(A[k] - B[k]) for k in A if k in B)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="results/order_frontier_*_bf16.csv")
    ap.add_argument("--window", type=float, default=50.0)
    ap.add_argument("--points", type=int, default=41)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    # Same exclusions as the rank tests: a seed arm re-runs a pair at another --seed-tokens and a
    # Gutenberg arm re-runs one anchor on a second protected corpus. Neither is another pair, and
    # counting one would put an anchor into the summary twice.
    paths = [p for p in sorted(glob.glob(a.glob))
             if not p.endswith("_matched.csv")
             and "_seed" not in os.path.basename(p) and "_gut_" not in os.path.basename(p)]
    tags = [re.sub(r"^order_frontier_|_bf16|\.csv$", "", os.path.basename(p)) for p in paths]
    floors = {t: noise_floor(t, a.window) for t in tags}
    have = [v for v in floors.values() if v is not None]
    fallback = max(have) if have else 0.0

    rows = []
    for path, tag in zip(paths, tags):
        G, ntok = curves(path)
        floor = floors[tag] if floors[tag] is not None else fallback
        orders = sorted(G)
        g1 = G[orders[0]]
        lo = max(v[0][0] for v in G.values())
        hi = min(v[-1][0] for v in G.values())
        grid = [lo + (hi - lo) * i / (a.points - 1) for i in range(a.points)]
        for o in orders[1:]:
            d = [(_at([x for x, _ in G[o]], [y for _, y in G[o]], g)
                  - _at([x for x, _ in g1], [y for _, y in g1], g)) * a.window / ntok
                 for g in grid]
            safer = any(x < -floor for x in d)          # L(alpha) below L(1): less likely
            worse = any(x > floor for x in d)
            verdict = ("crosses" if safer and worse else
                       "uniformly safer" if safer else
                       "uniformly more dangerous" if worse else
                       "indistinguishable")
            rows.append(dict(
                pair=LABEL.get(tag, tag), alpha=o, n_points=a.points,
                noise_floor=round(floor, 3), own_floor=floors[tag] is not None,
                fidelity_lo=round(lo, 1), fidelity_hi=round(hi, 1),
                min_nats_per_window=round(min(d), 3), max_nats_per_window=round(max(d), 3),
                frac_safer=round(sum(1 for x in d if x < -floor) / len(d), 3),
                frac_worse=round(sum(1 for x in d if x > floor) / len(d), 3),
                verdict=verdict))

    rows.sort(key=lambda r: (r["pair"], r["alpha"]))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "order_crossings.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print("L(alpha) - L(1) at matched fidelity, swept across the range every order covers.")
    print("Negative is safer. A sign counts only above that pair's own precision noise floor.\n")
    print(f"{'pair':16s}{'alpha':>6s}{'floor':>7s}{'min':>9s}{'max':>9s}"
          f"{'% safer':>9s}{'% worse':>9s}  verdict")
    for r in rows:
        print(f"{r['pair'][:15]:16s}{r['alpha']:>6.0f}{r['noise_floor']:>7.2f}"
              f"{r['min_nats_per_window']:>9.2f}{r['max_nats_per_window']:>9.2f}"
              f"{100*r['frac_safer']:>8.0f}%{100*r['frac_worse']:>8.0f}%  {r['verdict']}"
              + ("" if r["own_floor"] else "  (floor borrowed)"))
    n = len(rows)
    c = sum(1 for r in rows if r["verdict"] == "crosses")
    w_ = sum(1 for r in rows if r["verdict"] == "uniformly more dangerous")
    s_ = sum(1 for r in rows if r["verdict"] == "uniformly safer")
    print(f"\n{c} of {n} (pair, order) cells cross, {s_} are uniformly safer, {w_} uniformly more")
    print("dangerous, and the rest are inside the noise. A crossing means no order dominates and")
    print("which is safer depends on an operating point the published budget does not reveal.")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
