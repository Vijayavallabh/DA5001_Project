"""Aggregate the natural-memorisation audit (feat-018, plan v2 C7) from the runs of scripts/run_natural_memorisation.sh:
for each run (novel x decoding setting x decoder), per k and strategy: near-verbatim recall, longest span, spend,
utilisation, realised-ratio utilisation, and the decode-step activity of the single queries (fraction of steps forced
to the anchor / served unchanged), with the k = -1 and k = 0 baselines. Writes results/natural_memorisation.csv (single
queries, one row per run and k) and results/composition_70b.csv (all strategies).
Usage: .venv/bin/python analysis/natural_memorisation.py --runs output/phase2/nm --out results
"""
import argparse, csv, glob, json, os, statistics as st
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="output/phase2/nm")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    single, allrows = [], []
    for d in sorted(glob.glob(os.path.join(args.runs, "*"))):
        name = os.path.basename(d)
        summ = os.path.join(d, "composition_summary.csv")
        if not os.path.exists(summ):
            continue
        act = defaultdict(lambda: dict(steps=0, forced=0, free=0, n=0, R=[], Z=[], K=None))
        qf = os.path.join(d, "queries.jsonl")
        if os.path.exists(qf):
            for line in open(qf):
                q = json.loads(line)
                a = act[(q["k"], q["mode"], q["L"])]
                a["steps"] += q.get("steps", 0); a["forced"] += q.get("forced", 0); a["free"] += q.get("free", 0); a["n"] += 1
                a["R"].append(q["R"]); a["Z"].append(q["Z"]); a["K"] = q["K"]
        for r in csv.DictReader(open(summ)):
            k, mode, L = float(r["k"]), r["mode"], int(r["L"])
            a = act.get((k, mode, L))
            row = dict(run=name, novel=name.split("_")[0], setting=("B_authors" if "_B" in name else "A_temp1"), constraint=r.get("constraint", "kl"),
                       prefix_debt=int("nodebt" not in name), k=k, mode=mode, L=L, n_passages=int(r["n_passages"]),
                       nv_recall_mean=float(r["nv_recall_mean"]), nv_recall_ge_0p8_pct=float(r["nv_recall_ge_0p8_pct"]), lcs_word_mean=float(r["lcs_word_mean"]),
                       spend_total_mean=float(r["spend_total_mean"]), max_query_Z_over_K=float(r["max_query_Z_over_K"]), R_total_mean=float(r.get("R_total_mean", 0)),
                       invariant_violations=int(r["invariant_violations"]))
            if a and a["steps"]:
                row.update(steps_forced_pct=round(100 * a["forced"] / a["steps"], 2), steps_free_pct=round(100 * a["free"] / a["steps"], 2),
                           steps_active_pct=round(100 * (a["steps"] - a["forced"] - a["free"]) / a["steps"], 2),
                           R_over_K_max=round(max(a["R"]) / a["K"], 4) if a["K"] and a["K"] > 0 else None,
                           queries=a["n"])
            allrows.append(row)
            if mode == "single":
                single.append(row)
    if not allrows:
        raise SystemExit("no runs found")
    keys = list(allrows[0])
    for r in allrows:
        for k in keys:
            r.setdefault(k, None)
    with open(os.path.join(args.out, "composition_70b.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(allrows)
    with open(os.path.join(args.out, "natural_memorisation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(single)
    for r in single:
        print(f"[nm] {r['run']} k={r['k']:g}: recall {r['nv_recall_mean']} (>=0.8: {r['nv_recall_ge_0p8_pct']}%), LCS {r['lcs_word_mean']}, util max {r['max_query_Z_over_K']}, "
              f"forced {r.get('steps_forced_pct')}% free {r.get('steps_free_pct')}%")
    print("wrote", os.path.join(args.out, "natural_memorisation.csv"))


if __name__ == "__main__":
    main()
