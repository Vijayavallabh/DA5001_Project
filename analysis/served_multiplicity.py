"""Table 1's eight rows under a family-wise correction. No GPU, no judge call. Post hoc and descriptive.

Table 1 prints eight paired differences (selection n=64 minus the meter) with 95% intervals; only the
k=10 row of the released configuration and B1 were registered as primary. A referee asked for the
comparison to be read with multiplicity control, so this re-bootstraps each row's per-prompt
differences -- the same files the table is built from -- at 95% and at the Bonferroni level
1 - 0.05/8, and says which rows still exclude zero. It writes a separate CSV and draws from its own
seed, so no number already in results/served_opponent.csv is re-sampled (caution (j)).

Usage: .venv/bin/python analysis/served_multiplicity.py --out results
"""
import argparse
import csv
import os
import random

N_BOOT = 20000
ROWS = (  # (label, per-prompt file(s), meter column), in Table 1's order
    ("released k=0.5", ("frontier_levels_per_prompt_b.csv",), "u_met_k0.5"),
    ("released k=1", ("frontier_levels_per_prompt_b.csv",), "u_met_k1"),
    ("released k=10", ("frontier_levels_per_prompt_c.csv",), "u_met_k10"),
    ("70B base k=0.5", ("order_averaged_h2h_per_prompt_he70b_k05.csv",), "u_metered_k0.5"),
    ("70B base k=1", ("order_averaged_h2h_per_prompt_he70b_k20.csv",), "u_met70b_k1"),
    ("70B base k=20", ("order_averaged_h2h_per_prompt_he70b_k20.csv",), "u_metered_k20"),
    ("chat template k=1", ("order_averaged_h2h_per_prompt_served_k1chat.csv",), "u_metered_k1"),
    ("chat template k=10", ("order_averaged_h2h_per_prompt_served_k10chat.csv",), "u_metered_k10"),
)


def load(d, files):
    """Per-prompt columns, merged over the files one judge pass was split across."""
    out = {}
    for f in files + (("frontier_levels_per_prompt_b.csv",) if files[0].startswith("frontier") else ()):
        for r in csv.DictReader(open(os.path.join(d, f), encoding="utf-8")):
            out.setdefault(r["prompt_id"], {}).update(r)
    return out


def boot(xs, rng, alphas):
    n = len(xs)
    b = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return {a: (b[int(a / 2 * N_BOOT)], b[int((1 - a / 2) * N_BOOT) - 1]) for a in alphas}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=1840)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    fam = 0.05 / len(ROWS)
    rows = []
    for label, files, col in ROWS:
        d = load(a.dir, files)
        pids = sorted(d)
        assert len(pids) == 500, (label, len(pids))
        xs = [float(d[p]["u_sel_n64"]) - float(d[p][col]) for p in pids]
        ci = boot(xs, rng, (0.05, fam))
        m = sum(xs) / len(xs)
        (lo, hi), (blo, bhi) = ci[0.05], ci[fam]
        rows.append(dict(row=label, value=round(m, 4), lo95=round(lo, 4), hi95=round(hi, 4),
                         lo_bonf=round(blo, 4), hi_bonf=round(bhi, 4), level=round(1 - fam, 5),
                         excludes_zero_95=lo > 0 or hi < 0, excludes_zero_bonf=blo > 0 or bhi < 0, n=len(xs)))
        print(f"{label:20s} {m:+.4f}  95% [{lo:+.4f}, {hi:+.4f}]  bonf [{blo:+.4f}, {bhi:+.4f}]")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "served_multiplicity.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
