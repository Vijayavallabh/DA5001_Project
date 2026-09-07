"""feat-025 / feat-032: rebuild results/prefix_debt_ablation.csv from the run directories.

The table was assembled by hand in phase 2 and then had to grow a k=20 row for both risky models
(feat-032), so it is now built by a command. Each source is one composition_attack.py run directory
with a composition_summary.csv; the only thing that differs between a debt-on and a debt-off run of the
same model is the --no-prefix-debt flag, so the run identity carries the `prefix_debt` column.

Reads:  output/phase2/{nm/hp1_B,nm/hp1_B_nodebt,comp8b_kl,prefix_ablation}/composition_summary.csv
        output/phase3/{nm/hp1_B_nodebt_k20,prefix_ablation_k20}/composition_summary.csv
Writes: <out>/prefix_debt_ablation.csv
Usage:  .venv/bin/python analysis/merge_prefix_debt.py --out results
"""
import argparse, csv, os

# (model, decoding setting, prefix_debt, run label, directory)
SOURCES = [
    ("llama-70B", "B", 1, "hp1_B", "output/phase2/nm/hp1_B"),
    ("llama-70B", "B", 0, "hp1_B_nodebt", "output/phase2/nm/hp1_B_nodebt"),
    ("llama-70B", "B", 0, "hp1_B_nodebt_k20", "output/phase3/nm/hp1_B_nodebt_k20"),
    ("memoriser-8B", "A", 1, "comp8b_kl", "output/phase2/comp8b_kl"),
    ("memoriser-8B", "A", 0, "prefix_ablation", "output/phase2/prefix_ablation"),
    ("memoriser-8B", "A", 0, "prefix_ablation_k20", "output/phase3/prefix_ablation_k20"),
]
KEEP = ["nv_recall_mean", "nv_recall_ge_0p8_pct", "lcs_word_mean", "spend_total_mean",
        "max_query_Z_over_K", "invariant_violations"]
FIELDS = ["model", "setting", "prefix_debt", "run", "k", "mode", "L", "n_passages"] + KEEP


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    rows, missing = [], []
    for model, setting, debt, run, d in SOURCES:
        path = os.path.join(d, "composition_summary.csv")
        if not os.path.exists(path):
            missing.append(path)
            continue
        for r in csv.DictReader(open(path)):
            if r["constraint"] != "kl" or int(r["L"] or 0) not in (0, 50):
                continue                                    # the ablation is the KL decoder, single and 50-token windows
            rows.append(dict(model=model, setting=setting, prefix_debt=debt, run=run,
                             k=float(r["k"]), mode=r["mode"], L=int(r["L"] or 0),
                             n_passages=int(r["n_passages"]), **{c: r[c] for c in KEEP}))
    if missing:
        print("[warn] absent, rows omitted:", *missing, sep="\n  ")
    rows.sort(key=lambda r: (r["model"], r["k"], r["mode"], -r["prefix_debt"]))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "prefix_debt_ablation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)

    print(f"\n{'model':<13} {'k':>5} {'mode':<7} {'debt on':>9} {'debt off':>9} {'off/on':>7}")
    for model in ("memoriser-8B", "llama-70B"):
        for k in sorted({r["k"] for r in rows if r["model"] == model}):
            for mode in ("single", "oracle"):
                sel = {r["prefix_debt"]: float(r["nv_recall_mean"]) for r in rows
                       if r["model"] == model and r["k"] == k and r["mode"] == mode}
                if len(sel) < 2:
                    continue
                on, off = sel[1], sel[0]
                ratio = f"{off / on:.1f}x" if on > 0 else "--"
                print(f"{model:<13} {k:>5g} {mode:<7} {on:>9.4f} {off:>9.4f} {ratio:>7}")
    print(f"\nwrote {os.path.join(args.out, 'prefix_debt_ablation.csv')} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
