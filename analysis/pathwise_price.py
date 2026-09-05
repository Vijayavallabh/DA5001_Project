"""Price of the pathwise certificate (feat-019, plan v2 C8): compare the pathwise and KL decoders on the same prompts
and budgets from h1.py trajectory logs: fraction of decode steps forced to the anchor / constrained / served unchanged,
fraction of generations identical to the risky-only baseline (same seed), utilisation, realised ratio, and the
risky model's mean log-likelihood of the generated text as a utility proxy (from the logs' per-step p_risky_prob).
Usage: .venv/bin/python analysis/pathwise_price.py --kl output/sweep_plain --pathwise output/phase2/pathwise_sweep --out results
"""
import argparse, csv, glob, json, math, os, statistics as st
from collections import defaultdict


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "trajectories_k*_*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            md, ag, steps = r["metadata"], r["aggregate"], r["per_step_log"]
            out[(md["k"], md["prompt_id"], md["seed"])] = (md, ag, steps)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kl", nargs="+", required=True, help="log dirs of the KL decoder (must contain k=-1 rows for identity)")
    ap.add_argument("--pathwise", nargs="+", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--prefix", default="pathwise_price")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = {"kl": {}, "pathwise": {}}
    for d in args.kl:
        runs["kl"].update(load(d))
    for d in args.pathwise:
        runs["pathwise"].update(load(d))
    risky = {(pid, seed): ag["generation"] for (k, pid, seed), (md, ag, steps) in runs["kl"].items() if k == -1.0}
    rows = []
    for cons, data in runs.items():
        groups = defaultdict(list)
        for (k, pid, seed), (md, ag, steps) in data.items():
            if k <= 0:
                continue
            bd = [s["bd"] for s in steps if s.get("bd") is not None]
            forced = sum(b <= 1e-6 for b in bd)
            free = sum(b >= 1 - 1e-6 for b in bd)
            lp = [math.log(s["p_risky_prob"]) for s in steps if s.get("p_risky_prob")]
            ident = int(ag["generation"] == risky.get((pid, seed), object()))
            groups[(k, md["split"])].append(dict(forced=forced / max(1, len(bd)), active=(len(bd) - forced - free) / max(1, len(bd)), free=free / max(1, len(bd)),
                                                util=(ag["total_spend"] / ag["final_budget"]) if ag["final_budget"] > 0 else None,
                                                ratio_util=(ag.get("total_realised_ratio", 0.0) / ag["final_budget"]) if ag["final_budget"] > 0 else None,
                                                risky_nll=-st.mean(lp) if lp else None, ident=ident, len=len(bd)))
            groups[(k, "all")].append(groups[(k, md["split"])][-1])
        for (k, split), R in sorted(groups.items()):
            rows.append(dict(constraint=cons, k=k, split=split, n=len(R),
                             steps_forced_pct=round(100 * st.mean(r["forced"] for r in R), 2), steps_active_pct=round(100 * st.mean(r["active"] for r in R), 2), steps_free_pct=round(100 * st.mean(r["free"] for r in R), 2),
                             identical_to_risky_pct=round(100 * st.mean(r["ident"] for r in R), 1) if risky else None,
                             utilisation_median=round(st.median([r["util"] for r in R if r["util"] is not None] or [0]), 3), utilisation_max=round(max([r["util"] for r in R if r["util"] is not None] or [0]), 3),
                             ratio_utilisation_max=round(max([r["ratio_util"] for r in R if r["ratio_util"] is not None] or [0]), 3),
                             risky_nll_per_token_mean=round(st.mean([r["risky_nll"] for r in R if r["risky_nll"] is not None] or [0]), 4),
                             gen_len_mean=round(st.mean(r["len"] for r in R), 1)))
    with open(os.path.join(args.out, f"{args.prefix}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["split"] == "all":
            print(f"[price] {r['constraint']} k={r['k']:g} n={r['n']}: forced {r['steps_forced_pct']}% active {r['steps_active_pct']}% free {r['steps_free_pct']}%; identical to risky {r['identical_to_risky_pct']}%; util max {r['utilisation_max']}; risky NLL/token {r['risky_nll_per_token_mean']}")
    print("wrote", os.path.join(args.out, f"{args.prefix}.csv"))


if __name__ == "__main__":
    main()
