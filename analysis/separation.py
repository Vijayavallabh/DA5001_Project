"""feat-031 (plan v3, C17): can a per-user KL budget be tuned to admit readers and refuse reconstruction?

Section VIII-C shows that a per-user odometer bounds what many compliant queries reveal. It does not ask
the deployer's question: is there a budget that a reader does not notice and a reconstructor cannot use?
This script answers it by putting both activities in the same currency, nats of KL spend.

Two rates are measured on the same decoder:

  c_legit   the per-token spend of ordinary traffic, from the per-trajectory logs of the released runs
            over all six prompt classes (results/per_trajectory.csv, columns Z and gen_len);
  c_rec(k)  the spend per token of a protected work actually recovered by the windowed adversary,
            from the uncapped odometer replay (results/odometer_per_passage.csv, spend_total and recall
            against the target length).

A per-user budget B then admits about B / c_legit tokens of ordinary completion and holds reconstruction
to about B / (c_rec(k) * |y*|) of the work, so the budget's protective power is the ratio c_rec(k)/c_legit.
That ratio is not a design parameter: it is set by k. The output records it per budget and strategy,
together with B*, the largest replayed budget that holds reconstruction at or below a target fraction,
expressed in the unit a deployer cares about - ordinary completions admitted to a reader.

Reads:  results/per_trajectory.csv, results/odometer_per_passage.csv
Writes: <out>/separation.csv (per k, mode, L, B_user), <out>/separation_summary.csv (per k, mode, L),
        <figures>/separation.{pdf,png} if --figures is given.
Usage:  .venv/bin/python analysis/separation.py --results results --out results --figures figures
"""
import argparse, csv, os, statistics as st
from collections import defaultdict

COMPLETION_TOKENS = 200  # T_max of the released runs: one ordinary answer
TARGET_TOKENS = 260.4    # mean length of the 100 composition targets (results/composition_summary.csv)


def legit_rate(path):
    """Per-token KL spend of ordinary traffic: median over trajectories, and the per-class spread."""
    per_class = defaultdict(list)
    for r in csv.DictReader(open(path)):
        try:
            z, n = float(r["Z"]), float(r["gen_len"])
        except (ValueError, KeyError):
            continue
        if n > 0 and float(r["k"]) > 0:
            per_class[(float(r["k"]), r["split"])].append(z / n)
    allv = [v for vs in per_class.values() for v in vs]
    med = {key: st.median(vs) for key, vs in per_class.items()}
    return st.median(allv), min(med.values()), max(med.values()), len(allv)


def recon_rows(path):
    """(k, mode, L) -> {B_user: [recall...]} plus the uncapped spend, from the odometer replay."""
    recall, spend = defaultdict(lambda: defaultdict(list)), defaultdict(list)
    for r in csv.DictReader(open(path)):
        k = float(r["k"])
        if k <= 0:  # k=-1 (risky only) and k=0 (anchor only) are baselines, not budgeted runs
            continue
        key = (k, r["mode"], int(r["L"]))
        b = float(r["B_user"])
        recall[key][b].append(float(r["recall"]))
        if b == float("inf"):
            spend[key].append(float(r["spend_total"]))
    return recall, spend


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results")
    ap.add_argument("--figures", default="")
    ap.add_argument("--target-recall", type=float, default=0.10, help="fraction of the work the deployer will tolerate")
    args = ap.parse_args()

    c_legit, lo, hi, n_legit = legit_rate(os.path.join(args.results, "per_trajectory.csv"))
    recall, spend = recon_rows(os.path.join(args.results, "odometer_per_passage.csv"))
    nats_per_completion = c_legit * COMPLETION_TOKENS

    per_budget, summary = [], []
    for key in sorted(recall):
        k, mode, L = key
        uncapped = st.median(spend[key]) if spend[key] else float("nan")
        rc_unc = st.mean(recall[key][float("inf")]) if float("inf") in recall[key] else float("nan")
        tokens_recovered = rc_unc * TARGET_TOKENS
        c_rec = uncapped / tokens_recovered if tokens_recovered > 0 else float("inf")
        finite = sorted(b for b in recall[key] if b != float("inf"))
        for b in finite + [float("inf")]:
            rc = st.mean(recall[key][b])
            per_budget.append(dict(
                k=k, mode=mode, L=L, B_user=("inf" if b == float("inf") else f"{b:.0f}"),
                recall_mean=round(rc, 4),
                reader_tokens=("" if b == float("inf") else round(b / c_legit, 1)),
                reader_completions=("" if b == float("inf") else round(b / nats_per_completion, 3)),
            ))
        # B*: the largest replayed budget holding reconstruction at or below the target
        ok = [b for b in finite if st.mean(recall[key][b]) <= args.target_recall]
        b_star = max(ok) if ok else None
        binds = rc_unc > args.target_recall  # does the budget have anything to do at this k?
        summary.append(dict(
            k=k, mode=mode, L=L,
            recall_uncapped=round(rc_unc, 4),
            spend_uncapped_nats=round(uncapped, 1),
            c_rec_nats_per_recovered_token=(round(c_rec, 2) if c_rec != float("inf") else "inf"),
            protective_ratio=(round(c_rec / c_legit, 2) if c_rec != float("inf") else "inf"),
            budget_needed=int(binds),
            B_star_nats=("" if not binds else ("none" if b_star is None else f"{b_star:.0f}")),
            B_star_reader_completions=("" if not binds or b_star is None else round(b_star / nats_per_completion, 2)),
        ))

    os.makedirs(args.out, exist_ok=True)
    for name, data in (("separation.csv", per_budget), ("separation_summary.csv", summary)):
        with open(os.path.join(args.out, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    print(f"c_legit = {c_legit:.3f} nats/token over {n_legit} trajectories "
          f"(per-class medians {lo:.3f}-{hi:.3f}); one {COMPLETION_TOKENS}-token completion = "
          f"{nats_per_completion:.0f} nats")
    for s in summary:
        if s["budget_needed"]:
            print(f"  k={s['k']:<5} {s['mode']:<8} L={s['L']:<3} uncapped recall {s['recall_uncapped']:.3f} "
                  f"| c_rec {s['c_rec_nats_per_recovered_token']} nats/tok | ratio {s['protective_ratio']} "
                  f"| B* {s['B_star_nats']} nats = {s['B_star_reader_completions']} completions")
    if args.figures:
        plot(recall, c_legit, nats_per_completion, args.figures)
    print("wrote", os.path.join(args.out, "separation.csv"), "and separation_summary.csv")


def plot(recall, c_legit, nats_per_completion, figures):
    """Reconstruction against what the same budget leaves a reader. One curve per k, strongest strategy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5})
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ks = sorted({k for k, _, _ in recall})
    colours = plt.cm.viridis([i / max(1, len(ks) - 1) * 0.85 for i in range(len(ks))])
    for k, col in zip(ks, colours):
        # strongest adversary at this k: the (mode, L) with the highest uncapped recall
        cands = [(key, st.mean(v[float("inf")])) for key, v in recall.items()
                 if key[0] == k and float("inf") in v]
        key = max(cands, key=lambda t: t[1])[0]
        bs = sorted(b for b in recall[key] if b != float("inf"))
        ax.plot([b / nats_per_completion for b in bs], [st.mean(recall[key][b]) for b in bs],
                marker="o", ms=3, lw=1.3, color=col, label=f"$k={k:g}$ ({key[1]}, $L={key[2]}$)")
    ax.axhline(0.10, ls=":", lw=1, color="0.35")
    ax.text(0.4, 0.125, "10% of the work", fontsize=6.5, color="0.35")
    ax.set_xscale("log")
    ax.set_xlabel("ordinary completions the budget admits")
    ax.set_ylabel("fraction of the work recovered")
    ax.set_ylim(-0.03, 0.92)
    ax.legend(frameon=True, framealpha=0.9, loc="upper left")
    fig.tight_layout(pad=0.45)
    os.makedirs(figures, exist_ok=True)
    fig.savefig(os.path.join(figures, "separation.pdf"))
    fig.savefig(os.path.join(figures, "separation.png"), dpi=150)
    print("wrote", os.path.join(figures, "separation.pdf"))


if __name__ == "__main__":
    main()
