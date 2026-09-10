"""Plan v4 figures, rebuilt from results/*.csv only (same contract as figures/make_figures.py:
no GPU, no logs, the analysis scripts own the numbers and this script only draws them).

  frontier_scaling   <- results/anchor_scaling_summary.csv
  opening_effect     <- results/opening_effect_summary*.csv
  order_invariance   <- analysis.regimes.event_bound (closed form) + results/renyi_sweep.csv if present

Usage: .venv/bin/python figures/make_figures_v4.py [--copy-to /path/to/manuscript/figures]
"""
import argparse, csv, os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.regimes import event_bound  # noqa: E402

RESULTS, OUT = REPO / "results", REPO / "figures"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.6,
                     "xtick.labelsize": 7, "ytick.labelsize": 7.5})
LABEL = {"commonpile": "Common Pile", "commoncorpus": "Common Corpus", "kl3m": "KL3M"}
COLOR = {"commonpile": "C0", "commoncorpus": "C2", "kl3m": "C3"}


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf")


def frontier_scaling():
    """Both rates fall as the safe model improves; the one that must be ALLOWED falls faster than
    the one that must be FORBIDDEN, so the separation between them widens with capability."""
    rows = list(csv.DictReader(open(RESULTS / "anchor_scaling_summary.csv")))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.05))
    for corpus in ("commonpile", "commoncorpus", "kl3m"):
        g = sorted([r for r in rows if r["corpus"] == corpus], key=lambda r: float(r["params"]))
        if not g:
            continue
        x = [float(r["params"]) for r in g]
        c = COLOR[corpus]
        ax.plot(x, [float(r["s_passage"]) for r in g], "o-", color=c, lw=1.3, ms=3.4,
                label=f"{LABEL[corpus]}: $s(x)$, protected")
        ax.plot(x, [float(r["c_use"]) for r in g], "s--", color=c, lw=1.3, ms=3.4, alpha=0.65,
                label=f"{LABEL[corpus]}: $c_{{\\mathrm{{use}}}}$, ordinary")
        ax2.plot(x, [float(r["margin"]) for r in g], "o-", color=c, lw=1.4, ms=3.8,
                 label=LABEL[corpus])
    ticks = sorted({float(r["params"]) for r in rows})
    for a in (ax, ax2):
        a.set_xscale("log")
        a.set_xlabel("safe model parameters (B)")
        a.grid(alpha=0.25, lw=0.5)
        # default log ticks collide at these sizes; label the models we actually have
        a.set_xticks([0.17, 0.35, 1.0, 3.0, 7.0])
        a.set_xticklabels(["0.17", "0.35", "1", "3", "7"])
        a.set_xticks(ticks, minor=True)
        a.set_xticklabels([], minor=True)
        a.set_xlim(0.13, 9.5)
    ax.set_yscale("log")
    ax.set_yticks([0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3])
    ax.set_yticklabels(["0.15", "0.2", "0.3", "0.5", "0.7", "1.0", "1.3"])
    ax.set_yticks([], minor=True)   # suppress the leftover 4x10^-1 style minor labels
    ax.set_ylabel("nats per character")
    ax.legend(ncol=1, frameon=False, loc="lower left")
    ax2.set_ylabel("margin  $s(x)\\,/\\,c_{\\mathrm{use}}$")
    ax2.axhline(1.0, color="0.4", lw=0.8, ls=":")
    ax2.legend(frameon=False, loc="lower right")
    _save(fig, "frontier_scaling")


def opening_effect():
    """The interval between certifying and protecting is an opening effect: one token of genuine
    prefix removes most of it, and the same holds for every anchor we tried."""
    files = {"Common Pile 7B": "opening_effect_summary.csv",
             "TinyComma 1.8B": "opening_effect_summary_tinycomma18b.csv",
             "Pleias 3B": "opening_effect_summary_pleias3b.csv",
             "KL3M 3.7B": "opening_effect_summary_kl3m37b.csv"}
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for i, (name, f) in enumerate(files.items()):
        p = RESULTS / f
        if not p.exists():
            continue
        g = sorted(csv.DictReader(open(p)), key=lambda r: int(r["skip_tokens"]))
        ax.plot([int(r["skip_tokens"]) for r in g], [float(r["ratio_median"]) for r in g],
                "o-", color=f"C{i}", lw=1.3, ms=3.4, label=name)
    ax.axhline(1.0, color="0.4", lw=0.8, ls=":")
    ax.set_xlabel("tokens of genuine prefix supplied by the adversary")
    ax.set_ylabel("$k_{\\mathrm{crit}}/s(x)$  (uncertified width)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    _save(fig, "opening_effect")


def order_invariance():
    """Every order becomes vacuous at the same budget. A stronger charge buys a tighter bound
    below the threshold and does not move it."""
    S = 205.0
    Ks = [i * 2.0 for i in range(1, 121)]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for i, (a, lab) in enumerate([(1.0, r"$\alpha=1$  (KL, He et al.)"), (2.0, r"$\alpha=2$"),
                                  (8.0, r"$\alpha=8$"), (float("inf"), r"$\alpha=\infty$  (pathwise)")]):
        ax.plot(Ks, [event_bound(S, K, a) for K in Ks], lw=1.4, color=f"C{i}", label=lab)
    ax.axvline(S, color="0.3", lw=0.9, ls="--")
    ax.annotate(f"$K = S(x)$", xy=(S, 1e-4), xytext=(S * 0.42, 1e-4), fontsize=6.6, color="0.3")
    ax.set_yscale("log")
    ax.set_ylim(1e-12, 2)
    ax.set_xlabel("sequence budget $K$ (nats)")
    ax.set_ylabel("bound on $\\Pr[\\mathrm{reproduce}\\ x]$")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.25, lw=0.5)
    _save(fig, "order_invariance")


def onset_collapse():
    """The vacuity threshold is tight, and it is the natural scale: single-query recall from every
    admissible (anchor, risky) pair in results/onset_pairs.tsv falls on one curve once the budget
    is measured in units of the safe model's surprisal rate on the protected work. Pair count is
    data, not code -- it read two when this was written and reads four now."""
    import csv as _csv, statistics as _st, sys as _sys
    _sys.path.insert(0, str(REPO))
    from analysis.onset import curve, load_pairs
    # plan v5: the pair set is data, not code -- same manifest analysis/onset.py reads, so a new
    # admissible pair appears in the figure without editing it.
    _pairs = load_pairs(str(REPO / "results" / "onset_pairs.tsv"))
    _cols = ["C0", "C3", "C2", "C1", "C4", "C5", "C6", "C8"]
    _mks = ["o", "s", "^", "D", "v", "P", "X", "*"]
    P = [(n, c, b, _cols[i % len(_cols)], _mks[i % len(_mks)])
         for i, (n, c, b) in enumerate(_pairs)]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.05))
    for name, comp, per, col, mk in P:
        p_ = REPO / comp
        if not p_.exists():
            raise FileNotFoundError(p_)
        sx = _st.median(float(r["s_mean"]) for r in _csv.DictReader(open(REPO / per)))
        c = curve(str(p_), "single", 0)
        ks = sorted(c)
        ax.plot(ks, [c[k] for k in ks], mk + "-", color=col, lw=1.3, ms=3.6, label=name)
        ax.axvline(sx, color=col, lw=0.8, ls=":")
        ax2.plot([k / sx for k in ks], [c[k] for k in ks], mk + "-", color=col, lw=1.3, ms=3.6,
                 label=f"{name.split('+')[0].strip()}  ($s(x)={sx:.2f}$)")
    ax.set_xlabel("budget $k$ (nats per token)")
    ax.set_ylabel("single-query recall")
    ax.set_title("raw budget", fontsize=8)
    ax.legend(frameon=False, fontsize=6.0)
    ax2.axvline(1.0, color="0.35", lw=0.9, ls="--")
    ax2.annotate("certificate\nvacuous", xy=(1.0, 0.09), xytext=(1.03, 0.085), fontsize=6.2, color="0.35")
    ax2.set_xlabel("rescaled budget  $k / s(x)$")
    ax2.set_title("rescaled by the vacuity threshold", fontsize=8)
    ax2.legend(frameon=False, fontsize=6.2)
    for a_ in (ax, ax2):
        a_.grid(alpha=0.25, lw=0.5)
        a_.set_ylim(bottom=-0.004)
    _save(fig, "onset_collapse")


def seed_effect():
    """feat-064: the onset ratio against how many words of the work the adversary already holds.

    Two kinds of point, drawn differently because they carry different weight. Open circles are the
    seven pairs built for other reasons, where seed words is exactly 20 x characters-per-token and
    so cannot be separated from tokenizer granularity. Filled points joined by a line are the
    intervention, where the seed is varied with the tokenizer, models, corpus and metric all fixed.
    """
    import csv
    rows = list(csv.DictReader(open(RESULTS / "seed_effect.csv")))
    if not rows:
        raise FileNotFoundError("results/seed_effect.csv is empty")
    fig, ax = plt.subplots(figsize=(5.0, 3.1))

    obs = RESULTS / "onset_table.csv"
    if obs.exists():
        # the observational pairs, from the committed seven-pair table
        import json
        words = {"KL3M-1.7B": 7.3, "KL3M-520M": 7.3, "Pleias-350M": 13.0, "Phi-3.5-mini": 13.1,
                 "Comma-7B": 13.4, "Pleias-1.2B": 13.7, "TinyComma-1.8B": 14.4}
        xs, ys = [], []
        for r in csv.DictReader(open(obs)):
            if r["pair"].startswith("ALL"):
                continue
            key = next((k for k in words if r["pair"].startswith(k)), None)
            if key:
                xs.append(words[key]); ys.append(float(r["ratio"]))
        ax.scatter(xs, ys, s=34, facecolors="none", edgecolors="0.45", linewidths=1.1, zorder=2,
                   label="seven pairs (seed fixed at 20 tokens)")

    # The temperature arms hold the seed fixed and vary the warp, so on a "seed words" axis they
    # would stack at one x and be labelled "seed varied", which is false. They belong on
    # units_law(), which plots against s(x) -- the axis they actually move.
    rows = [r for r in rows if not r["pair"].endswith(" tau")]
    markers = ["o", "s", "^", "D"]
    labelled = False   # the first pair in sort order may have a single arm and be skipped below,
                       # so the legend entry has to hang off the first pair actually drawn
    for i, pair in enumerate(sorted({r["pair"] for r in rows})):
        g = sorted((r for r in rows if r["pair"] == pair), key=lambda r: float(r["seed_words"]))
        if len(g) < 2:
            continue
        x = [float(r["seed_words"]) for r in g]
        y = [float(r["ratio"]) for r in g]
        lo = [float(r["ratio"]) - float(r["ratio_lo"]) for r in g]
        hi = [float(r["ratio_hi"]) - float(r["ratio"]) for r in g]
        line = ax.errorbar(x, y, yerr=[lo, hi], marker=markers[i % len(markers)], ms=5, lw=1.4,
                           capsize=2.5, zorder=3, label=f"{pair}, seed varied")
        # Proposition 2's k_crit, rescaled by the pair's control arm and nothing else, so every
        # point but the control is an out-of-sample prediction made from the anchor alone.
        pred = [(xi, float(r["pred_ratio_K"])) for xi, r in zip(x, g) if r.get("pred_ratio_K")]
        if len(pred) >= 2:
            ax.plot(*zip(*pred), ls="--", lw=1.0, marker="x", ms=5, zorder=2,
                    color=line.lines[0].get_color(), alpha=0.75,
                    label=None if labelled else r"$k_{\mathrm{crit}}$ prediction")
            labelled = True

    ax.axhline(1.0, color="0.3", ls=":", lw=1.0)
    ax.text(ax.get_xlim()[1], 1.005, "certificate vacuous above", ha="right", va="bottom", fontsize=7,
            color="0.3")
    ax.set_xlabel("words of the work the adversary is given")
    ax.set_ylabel(r"onset $/\ s(x)$")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    _save(fig, "seed_effect")


def units_law():
    """The units claim on one axis pair: the safe model's surprisal rate against the budget at
    which extraction begins.

    Three kinds of point. Open circles are the two pairs whose adversary is handed about half as
    many words as the rest; filled circles are the five seed-matched pairs, and the fitted line is
    theirs alone. Squares joined to their own control by a thin line are the temperature arms,
    where the anchor, memoriser, corpus, tokenizer and seed are all fixed and only the warp moves
    s(x) -- the only points in the paper that move the x axis with nothing else changing.
    """
    import csv
    tbl = RESULTS / "onset_table.csv"
    if not tbl.exists():
        raise FileNotFoundError(str(tbl))
    words = {}
    sw = RESULTS / "onset_seed_words.csv"
    if sw.exists():
        words = {r["pair"]: float(r["seed_words"]) for r in csv.DictReader(open(sw))}

    def key(name):
        return next((k for k in words if k.split(" + ")[0] == name.split(" + ")[0]), None)

    matched, short = [], []
    for r in csv.DictReader(open(tbl)):
        if r["pair"].startswith("ALL"):
            continue
        x, y = float(r["s_safe"]), float(r["onset"])
        k = key(r["pair"])
        (matched if k and words[k] > 10 else short).append((x, y, r["pair"].split(" + ")[0]))
    if not matched:
        raise FileNotFoundError("no matched-context pairs in onset_table.csv")

    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    c = sum(y / x for x, y, _ in matched) / len(matched)
    lo = min(x for x, _, _ in matched + short) * 0.9
    hi = max([x for x, _, _ in matched + short] + [5.6]) * 1.04
    ax.plot([lo, hi], [lo, hi], color="0.55", ls=":", lw=1.0)
    ax.text(hi, hi, "certificate vacuous above", ha="right", va="bottom", fontsize=7, color="0.45",
            rotation=38, rotation_mode="anchor")
    ax.plot([lo, hi], [c * lo, c * hi], color="C0", lw=1.2,
            label=f"onset $= {c:.2f}\\,s(x)$, fitted on the five")
    ax.scatter([x for x, _, _ in matched], [y for _, y, _ in matched], s=34, color="C0", zorder=3,
               label="matched context (5 pairs)")
    ax.scatter([x for x, _, _ in short], [y for _, y, _ in short], s=38, facecolors="none",
               edgecolors="0.35", linewidths=1.1, zorder=3, label="short context (2 pairs)")

    # the temperature arms, each joined to its own control
    se = RESULTS / "seed_effect.csv"
    n_warp = 0
    if se.exists():
        rows = [r for r in csv.DictReader(open(se)) if r["pair"].endswith(" tau") and r["onset"]]
        n_pairs = len({r["pair"] for r in rows
                       if sum(1 for q in rows if q["pair"] == r["pair"]) >= 2})
        for pair in sorted({r["pair"] for r in rows}):
            g = sorted((r for r in rows if r["pair"] == pair), key=lambda r: float(r["s_x"]))
            if len(g) < 2:
                continue
            xs = [float(r["s_x"]) for r in g]
            ys = [float(r["onset"]) for r in g]
            ax.plot(xs, ys, color="C3", lw=0.9, alpha=0.8, zorder=2)
            ax.scatter(xs, ys, marker="s", s=30, color="C3", zorder=4,
                       label=(f"temperature arms ({n_pairs} pair{'s' if n_pairs != 1 else ''}, "
                              "warped)") if n_warp == 0 else None)
            n_warp += 1

    ax.set_xlabel(r"$s(x)$, the anchor's surprisal rate on the work (nats/token)")
    ax.set_ylabel("onset budget (nats/token)")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    _save(fig, "units_law")


def order_no_collapse():
    """What a higher Renyi order is worth at matched utility, against the one variable that had to
    explain it. F is the fraction of the unconstrained fidelity ceiling the audited decoder has
    already captured at that budget; F -> 1 is the audited decoder unconstrained, where every order
    must converge on it, and the curves do go there. Everywhere else they do not collapse: at a
    matched F the pairs stand decades apart, and two of them cross zero, which is the order leaking
    MORE at equal utility. Drawn from results/order_law.csv (analysis/order_law.py)."""
    import csv
    src = RESULTS / "order_law.csv"
    if not src.exists():
        raise FileNotFoundError(str(src))
    rows = [r for r in csv.DictReader(open(src))]
    orders = sorted({float(r["alpha"]) for r in rows})
    pairs = sorted({r["pair"] for r in rows})
    fig, axes = plt.subplots(1, len(orders), figsize=(6.9, 2.15), sharey=True)
    for ax, o in zip(axes, orders):
        for i, pair in enumerate(pairs):
            c = [(float(r["F"]), float(r["log10_factor"])) for r in rows
                 if r["pair"] == pair and float(r["alpha"]) == o]
            c.sort()
            ax.plot([x for x, _ in c], [y for _, y in c], marker="o", ms=2.2, lw=1.0,
                    color=f"C{i}", label=pair if o == orders[0] else None)
        ax.axhline(0.0, color="0.4", lw=0.7, ls=":")
        ax.set_title(rf"$\alpha = {o:.0f}$")
    axes[0].set_ylabel("$\log_{10}$ times safer, matched utility")
    axes[0].legend(loc="lower left", frameon=False, ncol=1)
    fig.supxlabel("$F$, the fraction of the unconstrained ceiling the audited decoder already has",
                  fontsize=8, y=0.01)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "order_no_collapse")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy-to", default="")
    a = ap.parse_args()
    print("rebuilding plan-v4 figures from results/")
    # One list, so a figure added here is also a figure copied. units_law was generated and not
    # copied for two days, and a missing \includegraphics halts tectonic and leaves the previous
    # PDF in place -- which then measures as if nothing were wrong.
    figures = (frontier_scaling, opening_effect, order_invariance, onset_collapse, seed_effect,
               units_law, order_no_collapse)
    for fn in figures:
        try:
            fn()
        except FileNotFoundError as e:
            print(f"  SKIP {fn.__name__}: {e}")
    if a.copy_to:
        import shutil
        for fn in figures:
            src = OUT / f"{fn.__name__}.pdf"
            if src.exists():
                shutil.copy(src, Path(a.copy_to).expanduser() / f"{fn.__name__}.pdf")
        print(f"copied to {a.copy_to}")


if __name__ == "__main__":
    main()
