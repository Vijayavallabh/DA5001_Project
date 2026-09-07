"""Plan v4 / feat-041: the published budget is not a sufficient statistic for the protection.

Proposition 1 says the vacuity threshold of the certificate is K = S(x) for every Renyi order
alpha in [1, inf]. Two decoders that publish the same k therefore publish the same certificate,
and at the budgets the mechanism's authors use that certificate is vacuous for both. This script
measures what those two decoders actually deliver.

It merges the per-order composition runs under output/phase4/renyi_*/ with the certificate status
from results/certificate_cap_summary.csv, so each row carries: the order, the budget, the measured
near-verbatim recall, and whether the certificate said anything at all. alpha = 1 is the audited
KL decoder and reproduces its published numbers, which is the cross-check that the independent
bisection solver is right.

With --price-runs it also reads the ordinary-prompt runs and writes the other half of the story:
what each order costs when nobody is attacking. At k=3 the four decoders publish the same budget
and the same 100%-vacuous certificate, and they span from touching 0.4% of ordinary decode steps
while leaking 0.097 of a passage, to touching 88% while leaking 0.001. The published budget
determines neither the protection nor the price.

Writes results/renyi_sweep.csv and, with --price-runs, results/renyi_price.csv.
Usage: .venv/bin/python analysis/renyi_sweep.py --runs output/phase4 --out results
"""
import argparse, csv, glob, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def order_of(dirname):
    """output/phase4/renyi_2 -> 2.0 ; renyi_1_0 -> 1.0 ; pathwise -> inf"""
    tag = os.path.basename(dirname)
    if "pathwise" in tag:
        return float("inf")
    m = re.search(r"renyi_(\d+)(?:_(\d+))?$", tag)
    if not m:
        return None
    return float(f"{m.group(1)}.{m.group(2) or '0'}")


def vacuous_pct(cap_summary, k):
    for r in cap_summary:
        if r.get("split") == "all" and abs(float(r["k"]) - k) < 1e-9:
            return float(r["vacuous_pct"])
    return None


def price(a):
    """What each order costs on ordinary traffic at the same budget.

    Restricted to one prompt class on purpose: a partially finished run has the early classes only,
    and silently averaging over different class mixes would compare different workloads.
    """
    import json, statistics as st
    rows = []
    for d in sorted(glob.glob(a.price_runs)):
        classes = (("neutral", "factual", "creative") if a.price_class == "all"
                   else (a.price_class,))
        files = [os.path.join(d, f"trajectories_k{a.price_k:g}_{c}.jsonl") for c in classes]
        if not os.path.isdir(d) or not all(os.path.exists(x) and os.path.getsize(x) for x in files):
            continue
        tot = free = active = forced = n = 0
        d3 = []
        for line in (ln for x in files for ln in open(x)):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            agg = r["aggregate"]
            tot += len(r["per_step_log"])
            n += 1
            free += agg.get("steps_risky_unchanged") or 0
            active += agg.get("steps_active") or 0
            forced += agg.get("steps_forced_safe") or 0
            w = (agg.get("generation") or "").split()
            if len(w) > 6:
                t3 = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
                d3.append(len(set(t3)) / len(t3))
        if not tot:
            continue
        rows.append({"arm": os.path.basename(d).replace("util_", ""), "k": a.price_k,
                     "prompt_class": a.price_class, "n": n,
                     "risky_unchanged_pct": 100 * free / tot, "active_pct": 100 * active / tot,
                     "forced_anchor_pct": 100 * forced / tot,
                     "distinct3": st.mean(d3) if d3 else float("nan")})
    if not rows:
        print(f"[price] nothing matched {a.price_runs} for k={a.price_k:g} {a.price_class}", file=sys.stderr)
        return
    path = os.path.join(a.out, "renyi_price.csv")
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nprice at k={a.price_k:g} on {a.price_class} prompts")
    print(f"{'arm':>12s} {'n':>4s} {'risky unchanged %':>18s} {'active %':>9s} {'distinct-3':>11s}")
    for r in sorted(rows, key=lambda r: -r["risky_unchanged_pct"]):
        print(f"{r['arm']:>12s} {r['n']:4d} {r['risky_unchanged_pct']:18.2f} "
              f"{r['active_pct']:9.2f} {r['distinct3']:11.4f}")
    print(f"wrote {path} ({len(rows)} arms)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="output/phase4")
    ap.add_argument("--out", default="results")
    ap.add_argument("--caps", default="results/certificate_cap_summary.csv")
    ap.add_argument("--price-runs", default="", metavar="GLOB",
                    help="ordinary-prompt runs to price the orders, e.g. 'output/phase4/util_*'")
    ap.add_argument("--price-class", default="neutral",
                    help="prompt class to price on, or 'all' for the three ordinary classes; must be complete in every run being compared")
    ap.add_argument("--price-k", type=float, default=3.0)
    a = ap.parse_args()

    caps = list(csv.DictReader(open(a.caps))) if os.path.exists(a.caps) else []
    rows = []
    for d in sorted(glob.glob(os.path.join(a.runs, "renyi_*"))):
        alpha = order_of(d)
        path = os.path.join(d, "composition_summary.csv")
        if alpha is None or not os.path.exists(path):
            continue
        for r in csv.DictReader(open(path)):
            k = float(r["k"])
            if k <= 0:                                  # k=-1 and k=0 are the unbudgeted baselines
                continue
            rows.append({
                "alpha": alpha, "k": k, "mode": r["mode"], "L": int(r["L"]),
                "n_passages": int(r["n_passages"]),
                "nv_recall_mean": float(r["nv_recall_mean"]),
                "lcs_word_mean": float(r["lcs_word_mean"]),
                "max_query_Z_over_K": float(r["max_query_Z_over_K"]),
                "invariant_violations": int(r["invariant_violations"]),
                "certificate_vacuous_pct": vacuous_pct(caps, k),
            })
    if not rows:
        print(f"no renyi runs found under {a.runs}", file=sys.stderr)
        return 1
    rows.sort(key=lambda r: (r["k"], r["mode"], r["L"], r["alpha"]))
    if a.price_runs:
        price(a)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "renyi_sweep.csv")
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    orders = sorted({r["alpha"] for r in rows})
    print(f"{'k':>4s} {'mode':>7s} {'L':>3s} {'cert vacuous':>13s}  " +
          "  ".join(f"a={o:g}".rjust(8) for o in orders))
    seen = {}
    for r in rows:
        seen.setdefault((r["k"], r["mode"], r["L"]), {})[r["alpha"]] = r
    for key in sorted(seen):
        k, mode, L = key
        g = seen[key]
        vac = next(iter(g.values()))["certificate_vacuous_pct"]
        cells = "  ".join((f"{g[o]['nv_recall_mean']:.3f}" if o in g else "-").rjust(8) for o in orders)
        print(f"{k:4g} {mode:>7s} {L:3d} {('%.0f%%' % vac) if vac is not None else 'n/a':>13s}  {cells}")
    print(f"\nwrote {path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
