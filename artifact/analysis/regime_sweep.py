"""Regime sweep summary (feat-005, contribution C2).

Reads E1 output directories (one per prompt-format variant) holding trajectories_k{k}_{split}.jsonl for
k in {-1, 0, 0.1, ...} and writes results/regime_sweep.csv with, per (variant, k, split):
% solver-active steps, % forced-safe steps, % risky-unchanged steps, utilisation, % generations byte-identical
to the k=-1 (risky-only) generation with the same prompt and trajectory index, leading tokens forced to the
safe model by prefix debt, and the copying metrics (ROUGE-L, LCS word/char, ACS, nv-recall). Also
figures/regime_sweep.{pdf,png}.

Usage: .venv/bin/python analysis/regime_sweep.py --run plain=output/sweep_plain --run chat=output/sweep_chat --out results
"""
import argparse, csv, glob, json, math, os, re, statistics as st
from collections import defaultdict

FNAME = re.compile(r"trajectories_k(?P<k>-?[0-9.]+)_(?P<split>[a-z_]+)\.jsonl$")
METRICS = ("rouge_l", "lcs_word", "lcs_char", "acs_word", "nv_recall")


def load_run(d):
    """{k: {split: {(prompt_id, trajectory_id): record-summary}}}"""
    out = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(d, "trajectories_k*_*.jsonl"))):
        mm = FNAME.search(path)
        k, split = float(mm["k"]), mm["split"]
        recs = {}
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            a, m = r["aggregate"], r["metadata"]
            own = a["steps_forced_safe"] + a["steps_active"] + a["steps_risky_unchanged"]
            lead = 0
            for s in r["per_step_log"][:own]:
                if s["bd"] is not None and s["bd"] <= 1e-6:
                    lead += 1
                else:
                    break
            recs[(m["prompt_id"], m["trajectory_id"])] = dict(
                gen=a["generation"], own=own, lead=lead, util=a["utilisation"], inv=a["invariant_ok"],
                forced=a["steps_forced_safe"], active=a["steps_active"], free=a["steps_risky_unchanged"],
                debt=r["prefix_analysis"]["true_prefix_debt"], **{x: a[x] for x in METRICS})
        out[k][split] = recs
    return out


def summarise(name, run):
    rows = []
    base = run.get(-1.0, {})
    for k in sorted(run):
        for split in list(run[k]) + ["all"]:
            R = [v for s, d in run[k].items() if split in ("all", s) for v in d.values()]
            if not R:
                continue
            steps = sum(v["own"] for v in R)
            ident = None
            if split != "all" and base.get(split):
                shared = set(run[k][split]) & set(base[split])
                if shared:
                    ident = round(100 * sum(run[k][split][x]["gen"] == base[split][x]["gen"] for x in shared) / len(shared), 2)
            elif split == "all" and base:
                shared = [(s, x) for s in run[k] if s in base for x in set(run[k][s]) & set(base[s])]
                if shared:
                    ident = round(100 * sum(run[k][s][x]["gen"] == base[s][x]["gen"] for s, x in shared) / len(shared), 2)
            utils = [v["util"] for v in R if v["util"] is not None]
            row = dict(variant=name, k=k, split=split, n_traj=len(R),
                       active_step_pct=round(100 * sum(v["active"] for v in R) / steps, 3),
                       forced_safe_step_pct=round(100 * sum(v["forced"] for v in R) / steps, 3),
                       risky_unchanged_step_pct=round(100 * sum(v["free"] for v in R) / steps, 3),
                       traj_with_active_step_pct=round(100 * sum(v["active"] > 0 for v in R) / len(R), 2),
                       identical_to_risky_pct=ident,
                       util_mean=round(st.mean(utils), 4) if utils else None, util_max=round(max(utils), 4) if utils else None,
                       util_gt_0p9_pct=round(100 * sum(u > 0.9 for u in utils) / len(utils), 2) if utils else None,
                       invariant_violations=sum(not v["inv"] for v in R),
                       lead_forced_mean=round(st.mean(v["lead"] for v in R), 3), lead_forced_max=max(v["lead"] for v in R),
                       delta_init_mean=round(st.mean(v["debt"] for v in R if v["debt"] is not None), 3),
                       gen_len_mean=round(st.mean(v["own"] for v in R), 1))
            for x in METRICS:
                row[f"{x}_mean"] = round(st.mean(v[x] for v in R), 4)
            row["nv_recall_gt0_pct"] = round(100 * sum(v["nv_recall"] > 0 for v in R) / len(R), 2)
            row["lcs_word_max"] = max(v["lcs_word"] for v in R)
            rows.append(row)
    return rows


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8})
    from matplotlib.ticker import NullFormatter
    variants = sorted({r["variant"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.1))
    for vi, name in enumerate(variants):
        A = [r for r in rows if r["variant"] == name and r["split"] == "all" and r["k"] > 0]
        base = {r["k"]: r for r in rows if r["variant"] == name and r["split"] == "all" and r["k"] <= 0}
        ks = [r["k"] for r in A]
        ls = "-" if vi == 0 else "--"
        ax = axes[0]
        ax.plot(ks, [r["active_step_pct"] for r in A], "o" + ls, color="C0", label=f"constraint active (0<θ<1), {name}")
        ax.plot(ks, [r["forced_safe_step_pct"] for r in A], "s" + ls, color="C3", label=f"anchor forced (θ=0), {name}")
        ax.plot(ks, [r["identical_to_risky_pct"] for r in A], "^" + ls, color="C2", label=f"identical to risky model alone, {name}")
        ax = axes[1]
        ax.plot(ks, [r["lead_forced_mean"] for r in A], "d" + ls, color="C1", label=f"leading tokens written by the anchor, {name}")
        ax = axes[2]
        ax.plot(ks, [r["rouge_l_mean"] for r in A], "o" + ls, color="C4", label=f"ROUGE-L, {name}")
        ax.plot(ks, [r["nv_recall_mean"] for r in A], "s" + ls, color="C5", label=f"near-verbatim recall, {name}")
        for kb, col, lab in ((-1.0, "C4", "risky model alone"), (0.0, "C4", "anchor alone")):
            if kb in base:
                ax.axhline(base[kb]["rouge_l_mean"], color=col, ls=":" if kb == 0 else "-.", lw=0.8, alpha=0.7, label=f"ROUGE-L {lab}, {name}")
    axes[0].set_ylabel("% of decode steps / generations")
    axes[1].set_ylabel("tokens (mean)")
    axes[2].set_ylabel("copying metric (mean)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([f"{k:g}" for k in ks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("per-token budget k")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="name=dir (repeatable)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--figures", default="figures")
    args = ap.parse_args()
    rows = []
    for spec in args.run:
        name, d = spec.split("=", 1)
        rows += summarise(name, load_run(d))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "regime_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["split"] == "all":
            print(f"{r['variant']} k={r['k']:g}: active {r['active_step_pct']}% forced {r['forced_safe_step_pct']}% identical-to-risky {r['identical_to_risky_pct']} lead_forced {r['lead_forced_mean']} rougeL {r['rouge_l_mean']} nv_recall {r['nv_recall_mean']} util_max {r['util_max']}")
    os.makedirs(args.figures, exist_ok=True)
    plot(rows, os.path.join(args.figures, "regime_sweep"))
    print("wrote", os.path.join(args.out, "regime_sweep.csv"), "and", os.path.join(args.figures, "regime_sweep.pdf"))


if __name__ == "__main__":
    main()
