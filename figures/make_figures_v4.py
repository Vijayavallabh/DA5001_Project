"""Plan v4 figures, rebuilt from results/*.csv only (same contract as figures/make_figures.py:
no GPU, no logs, the analysis scripts own the numbers and this script only draws them).

  frontier_scaling   <- results/anchor_scaling_summary.csv
  opening_effect     <- results/opening_effect_summary*.csv
  order_invariance   <- analysis.regimes.event_bound (closed form) + results/renyi_sweep.csv if present
  imitation_cost     <- results/imitation_cost.csv + results/imitation_lorenz.csv

Usage: .venv/bin/python figures/make_figures_v4.py [--copy-to /path/to/manuscript/figures]
"""
import argparse, csv, math, os, sys
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


def distinct_styles(n):
    """n (colour, marker) pairs, every one distinct, or an exception.

    Both figures that style a data-driven list of series used `list[i % len(list)]`, which is the
    one thing you must not do when the list is data: onset_collapse drew NINE pairs from an
    eight-colour and eight-marker list, so the ninth wrapped to index 0 and TinyComma-1.8B and
    open-calm-3b came out as the same blue circle -- two indistinguishable curves in both panels
    of an appendix figure, invisible to every grep and to the compiler (found 2026-09-17 by
    looking at the rendered figure). seed_effect uses the same construct over a four-marker
    list but filters its temperature arms out first and draws only three, so it had not yet
    collided; it takes its styles from here for the same reason. The comment
    on the first said the pair set is "data, not code, so a new admissible pair appears in the
    figure without editing it", which is exactly how it happened.

    Wrapping silently is the defect, so this raises instead."""
    cols = ["C0", "C3", "C2", "C1", "C4", "C5", "C6", "C8", "C9", "C7"]
    mks = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]
    if n <= len(cols):
        out = list(zip(cols[:n], mks[:n]))          # distinct colour AND marker
    else:
        # Past ten, colour alone cannot separate them, so walk the cross product: every colour
        # with the first marker, then every colour with the second. The PAIR stays unique, which
        # is what a reader needs. order_no_collapse drew twelve series as `color=f"C{i}"` with a
        # fixed marker, and matplotlib wraps C10 to C0 -- so Comma-7B and Qwen2.5-7B, and
        # KL3M-1.7B and TinyComma-1.8B, were the same blue and the same orange circle across all
        # three panels. `% len(...)` does not appear anywhere in that code: the wrap is inside
        # matplotlib's property cycle, which is why a grep for the explicit form missed it.
        out = [(c, m) for m in mks for c in cols][:n]
        if n > len(cols) * len(mks):
            raise ValueError(f"{n} series but only {len(cols) * len(mks)} distinct styles")
    assert len(set(out)) == n, out
    return out


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf")


def frontier_scaling():
    """Both rates fall as the safe model improves; the one that must be ALLOWED falls faster than
    the one that must be FORBIDDEN, so the separation between them widens with capability."""
    rows = list(csv.DictReader(open(RESULTS / "anchor_scaling_summary.csv")))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.35))
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
    ax2.set_ylabel("margin  $s(x)\\,/\\,c_{\\mathrm{use}}$")
    ax2.axhline(1.0, color="0.4", lw=0.8, ls=":")
    # The left panel's six entries were "<corpus>: <rate>" spelled out, which made the legend as wide
    # as the panel with nowhere to sit: two of the dashed curves ran through their own labels. But
    # colour already encodes the corpus and linestyle already encodes the rate, so the six factor
    # into 3 + 2 much shorter ones, shared with the right panel, below the figure where nothing is
    # drawn. That also drops the right panel's duplicate copy of the three corpus names.
    from matplotlib.lines import Line2D
    keys = [c for c in ("commonpile", "commoncorpus", "kl3m")
            if any(r["corpus"] == c for r in rows)]
    handles = [Line2D([], [], color=COLOR[c], marker="o", ms=3.4, lw=1.3, label=LABEL[c])
               for c in keys]
    # The style keys need a handle long enough to show the dash pattern: at handlelength 1.8 the
    # marker covered the line and both proxies read as a bare dot, which is the one thing these two
    # entries exist to distinguish. They also keep full alpha -- a key should be crisp even where
    # the curve it stands for is drawn faint.
    handles += [
        Line2D([], [], color="0.35", marker="o", ms=3.4, lw=1.3, ls="-",
               label="$s(x)$, protected"),
        Line2D([], [], color="0.35", marker="s", ms=3.4, lw=1.3, ls=(0, (4, 1.6)),
               label="$c_{\\mathrm{use}}$, ordinary"),
    ]
    fig.legend(handles=handles, loc="lower center", frameon=False, ncol=len(handles), fontsize=7,
               bbox_to_anchor=(0.5, 0.0), handlelength=3.0, handletextpad=0.5,
               columnspacing=1.5, borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
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
    # Beside the rule it names, not 120 nats away from it with no arrow (which is where it sat
    # until 2026-09-17): below 1e-4 the region just left of K = S(x) is empty on every curve.
    ax.annotate("$K = S(x)$", xy=(S * 1.02, 3e-8), rotation=90, ha="left", va="center",
                fontsize=6.6, color="0.3")
    ax.set_yscale("log")
    ax.set_ylim(1e-12, 2)
    ax.set_xlabel("sequence budget $K$ (nats)")
    ax.set_ylabel("bound on $\\Pr[\\mathrm{reproduce}\\ x]$")
    # "lower right" is exactly where the K = S(x) rule lands (S is at 85% of the x range), so the
    # dashed line ran through "He" in the first entry. The lower LEFT is empty: the three higher
    # orders stay below 1e-12 until K ~ 150, and the KL curve is already well up the y-axis there.
    ax.legend(frameon=False, loc="lower left")
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
    _st_ = distinct_styles(len(_pairs))
    P = [(n, c, b, _st_[i][0], _st_[i][1]) for i, (n, c, b) in enumerate(_pairs)]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.75))
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
    ax2.axvline(1.0, color="0.35", lw=0.9, ls="--")
    # An opaque bbox, not a bare annotation: the TinyComma curve climbs straight through
    # "certificate" here. Moving the text instead only trades one crossing curve for another --
    # nine series leave no empty corner in this panel (caution (ad)).
    ax2.annotate("certificate\nvacuous", xy=(1.0, 0.09), xytext=(1.03, 0.085), fontsize=6.2,
                 color="0.35", zorder=5,
                 bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
    ax2.set_xlabel("rescaled budget  $k / s(x)$")
    ax2.set_title("rescaled by the vacuity threshold", fontsize=8)
    # Two unframed nine-entry legends, one per panel, both inside the axes: the left one had all
    # nine s(x) rules struck through its labels and the right one printed "certificate vacuous"
    # underneath two of its own entries. One shared legend below the panels instead -- the labels
    # differed only in carrying s(x), so the second legend was a duplicate anyway.
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", frameon=False, ncol=3, fontsize=6.2,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.4, handletextpad=0.4,
               columnspacing=1.4, borderaxespad=0.0)
    for a_ in (ax, ax2):
        a_.grid(alpha=0.25, lw=0.5)
        a_.set_ylim(bottom=-0.004)
    fig.tight_layout(rect=(0, 0.20, 1, 1))
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

    # The observational pairs come out of results/onset_seed_words.csv, which analysis/seed_effect.py
    # writes from the tokenizers themselves. This used to be a hardcoded dict carrying the stale
    # 13.0-14.4 words the manuscript corrected to 13.86-15.02 in five places -- the figure kept
    # plotting the old ones, and it knew about seven pairs after there were nine.
    obs = RESULTS / "onset_seed_words.csv"
    if obs.exists():
        pts = [(float(r["seed_words"]), float(r["ratio"])) for r in csv.DictReader(open(obs))]
        ax.scatter(*zip(*pts), s=34, facecolors="none", edgecolors="0.45", linewidths=1.1, zorder=2,
                   label=f"{len(pts)} pairs (seed fixed at 20 tokens)")

    # The temperature arms hold the seed fixed and vary the warp, so on a "seed words" axis they
    # would stack at one x and be labelled "seed varied", which is false. They belong on
    # units_law(), which plots against s(x) -- the axis they actually move.
    rows = [r for r in rows if not r["pair"].endswith(" tau")]
    # one marker per series, never wrapped: five series were drawn from four markers
    markers = [m for _, m in distinct_styles(10)]
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
    # The KL3M-520M arm struck through "vacuous above" here, and moving the label to the left edge
    # only traded one crossing line for two. It has to sit beside the y=1 rule it names, so it gets
    # an opaque backing instead: the only thing hidden is a short span of one line.
    ax.text(ax.get_xlim()[1], 1.005, "certificate vacuous above", ha="right", va="bottom", fontsize=7,
            color="0.3", bbox=dict(facecolor="white", edgecolor="none", pad=0.8), zorder=5)
    ax.set_xlabel("words of the work the adversary is given")
    ax.set_ylabel(r"onset $/\ s(x)$")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    _save(fig, "seed_effect")


def context_intervention():
    """feat-086: give every pair's adversary the same number of WORDS instead of the same number of
    tokens, and the fan closes. The five pairs already at 13.6-15.0 words do not move by
    construction -- they are the control, and they are drawn as the band the four movers descend
    into. All four move, all four move down, and all four land inside or on the edge of that band.
    """
    import csv
    rows = [r for r in csv.DictReader(open(RESULTS / "context_intervention.csv"))
            if r["pair"] != "ALL"]
    if not rows:
        raise FileNotFoundError("results/context_intervention.csv is empty")
    moved = [r for r in rows if abs(float(r["ratio_matched"]) - float(r["ratio_20"])) > 1e-9]
    flat = [float(r["ratio_20"]) for r in rows if r not in moved]

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.axhspan(min(flat), max(flat), color="0.85", zorder=1)
    ax.text(-0.55, (min(flat) + max(flat)) / 2,
            f"{len(flat)} pairs already at\nthe matched context\n"
            f"{min(flat):.3f}-{max(flat):.3f}", fontsize=6.8, va="center", color="0.35")
    for r in sorted(moved, key=lambda r: -float(r["ratio_20"])):
        y0, y1 = float(r["ratio_20"]), float(r["ratio_matched"])
        ax.plot([0, 1], [y0, y1], marker="o", ms=4, lw=1.6, zorder=3)
        ax.annotate(f"{r['pair']}  {float(r['words_20']):.1f}$\\to${float(r['words_matched']):.1f}w",
                    (1.03, y1), fontsize=6.8, va="center")
    ax.axhline(1.0, color="0.3", ls=":", lw=1.0, zorder=2)
    ax.set_xlim(-0.62, 1.95)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["20-token seed\n(what the benchmark fixes)",
                        "matched seed\n13.6-15.0 words"], fontsize=8)
    ax.set_ylabel(r"onset $/\ s(x)$")
    ax.set_title("spread $0.289 \\to 0.113$", fontsize=8.5, loc="left")
    _save(fig, "context_intervention")


def selection_frontier():
    """feat-087/088: the paper's thesis in two panels.

    (a) Why the two mechanisms differ in kind. A metered budget is K = kT, a ray through the origin
        whose slope is k; the price of the work is S(x) = s(x)T, another ray. Vacuity is K >= S(x),
        so for a metered decoder it is decided by the ratio k/s(x) and never by the length: the
        k = 10 ray that buys utility is above the band at EVERY length, and the k = 0.5 ray that
        stays below it buys nothing measurable. A selection budget is log n, a horizontal line,
        which the work's price overtakes after one or two tokens and never catches again.
    (b) What a nat buys. x is the realised sequence divergence from the anchor -- for the metered
        decoder the mean measured spend, for selection the exact log n - (n-1)/n -- and y is the
        judged utility both are scored on. The dotted line is the Cramer rate function of that
        utility under the anchor, which Theorem 1 says no policy can sit to the left of.

    Every input is read from a CSV: s(x) per token per pair from onset_table.csv, the median total
    surprisal of a protected target from odometer.csv, the decoder's spend and utility from
    utility_price.csv, and the selection arms from selection_decoding.csv.
    """
    import csv
    import math
    sel = list(csv.DictReader(open(RESULTS / "selection_decoding.csv")))
    dec = list(csv.DictReader(open(RESULTS / "utility_price.csv")))
    onset = [r for r in csv.DictReader(open(RESULTS / "onset_table.csv"))
             if not r["pair"].startswith("ALL")]
    odo = list(csv.DictReader(open(RESULTS / "odometer.csv")))
    if not sel or not dec or not onset or not odo:
        raise FileNotFoundError("an input CSV for selection_frontier is empty")
    fig, (axL, ax) = plt.subplots(1, 2, figsize=(6.9, 2.15))
    # Printed at \textwidth: pre-scale the type by the shrink it will take. Raising this to
    # 6.9/4.05 on 2026-09-15 to allow a narrower \includegraphics broke the layout -- axis
    # labels of the two panels collided and the legend sat on the data -- because figsize is
    # fixed. If this figure must get narrower, shrink figsize too and RENDER THE PAGE.
    F = 6.9 / 5.5

    # ---- panel (a): the contribution at a glance -------------------------------------------
    # The claim is structural, so this panel stays structural: certified budgets against the price
    # of the work, both as functions of the SAME length. The measured spends belong in panel (b),
    # whose x-axis is realised divergence -- putting 171.3 nats of ORDINARY-traffic spend on an
    # axis labelled "length of the protected work" would be the denominator error this paper
    # spends Section 4 complaining about.
    ss = sorted(float(r["s_safe"]) for r in onset)          # nats per token, nine pairs
    s_med = ss[len(ss) // 2]
    s_tot = float(odo[0]["S_total_median"])                  # 849 nats, the median protected target
    t_star = s_tot / s_med                                   # its length in tokens
    T = [10 ** (1 + 0.02 * i) for i in range(101)]           # 10 .. 1000 tokens
    TOP = 1.2e5
    # Everything above the band is a budget that exceeds what the work is worth, which is exactly
    # where Proposition 2 says the certificate stops excluding anything. Shading it is the whole
    # argument: the red rays enter it and never leave; the blue lines never enter it.
    axL.fill_between(T, [ss[-1] * t for t in T], TOP, color="#c1443c", alpha=0.07, lw=0)
    axL.annotate("certificate vacuous", (T[3], 4.2e4), fontsize=6.3 * F, color="#8c3230",
                 ha="left", va="center")
    axL.fill_between(T, [ss[0] * t for t in T], [ss[-1] * t for t in T],
                     color="0.72", alpha=0.5, lw=0)
    axL.plot(T, [s_med * t for t in T], color="0.25", lw=1.4)
    axL.annotate("$S(x) = s(x)T$", (T[26], s_med * T[26]), fontsize=6.3 * F, color="0.2",
                 rotation=31, rotation_mode="anchor", xytext=(0, 5), textcoords="offset points")
    # k=3's label sat at T[40], where the "$S(x) = s(x)T$" annotation anchored at T[26] runs into
    # it: the two lines are nearly collinear here -- s(x) is about 3 nats/token, which is the
    # paper's point -- so their labels overlap and the compiled page read "S(x) = s(x)T/k = 3" as
    # one smear. Only a reader of the rendered page sees it (caution (ad)). Moved right to T[52]:
    # the S(x) label runs from T[26] to about T[44], the median-target rule stands at
    # t_star (about T[70] on this grid) and k=0.5 sits at T[85], so T[52] is the one gap
    # between them. T[62] was tried first and put the label on the vertical rule --
    # caution (ad): moving a label trades one collision for another, so look again.
    # `dy` is which SIDE of its own ray the label sits on, and it is not cosmetic. Every ray here
    # has slope 1 on log-log axes, so the gap to the next ray is constant -- while a label rotated
    # to 31 degrees RISES about 0.67 decades over its own length. k=0.5 sits 0.78 decades under the
    # S(x)/k=3 pair, so a label placed above its ray runs its top edge into both of them; measured
    # off the compiled page rather than guessed (caution (aj)): the anchors are exact, the extent is
    # what collides. Putting that one label below its ray is the only free direction.
    for k, style, xi, dy in ((10.0, "-", 11, 4), (3.0, "--", 52, 4), (0.5, ":", 85, -4)):
        axL.plot(T, [k * t for t in T], style, color="#c1443c", lw=1.4)
        axL.annotate(f"$k={k:g}$", (T[xi], k * T[xi]), fontsize=6.3 * F,
                     color="#c1443c", rotation=31, rotation_mode="anchor",
                     va="baseline" if dy > 0 else "top",
                     xytext=(0, dy), textcoords="offset points")
    import math as _m
    for n, style in ((64, "--"), (8, "-")):
        axL.axhline(_m.log(n), color="#2f6f9f", lw=1.5, ls=style)
    # e^K is the factor a rights-holder is promised -- 8 and 64 against e^2000 -- and that reading
    # is in the caption, not here: every in-plot home for it collided, with the x-axis tick labels
    # below the n=8 line and with the k=0.5 ray above the n=64 one.
    axL.annotate("$\\log n$, $n=8,64$", (T[97], _m.log(64)), fontsize=6.3 * F, ha="right",
                 color="#2f6f9f", xytext=(0, 4), textcoords="offset points")
    axL.axvline(t_star, color="0.5", lw=0.8, ls="-.")
    axL.annotate(f"median target,\n$S(x)={s_tot:.0f}$", (t_star, 5.5e4), fontsize=6.3 * F,
                 color="0.35", ha="left", va="top", xytext=(4, 0), textcoords="offset points")
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlim(10, 1000); axL.set_ylim(1.0, TOP)
    axL.set_xlabel("length of the work, tokens")
    axL.set_ylabel("certified budget $K$, nats")
    axL.set_title("(c) $kT$ grows with the work; $\\log n$ does not",
                  fontsize=7.6 * F, loc="left")

    # Panel (b) is ONE pass family (feat-186, results/frontier_levels.csv): every arm judged by
    # judge B in BOTH presentation orders against the same opponent, on de-echoed text. Until
    # 2026-09-24 this panel put the meter's levels from one single-order pass beside selection's
    # from another -- two passes on one axis, which caution (ap) forbids -- on text that carried the
    # prompt's tail (caution (bc)). Order-averaged levels against one opponent ARE comparable to one
    # another (a greedy judge shown both orders is a function of the two texts alone), so the y axis
    # is now a level with 0.5 = parity with serving the risky model, and it says so.
    import glob
    from collections import Counter
    fl = {r["arm"]: r for r in csv.DictReader(open(RESULTS / "frontier_levels.csv"))}
    anc = None
    for f in sorted(glob.glob(str(RESULTS / "frontier_levels_per_prompt_*.csv"))):
        rows_ = list(csv.DictReader(open(f)))
        if rows_ and "u_anchor_k0" in rows_[0]:
            anc = [float(r["u_anchor_k0"]) for r in rows_]
    if anc is None:
        raise FileNotFoundError("no per-prompt file carries the anchor-alone arm")
    cnt = Counter(anc)
    vals = sorted(cnt)
    probs = [cnt[v] / len(anc) for v in vals]
    u0 = sum(anc) / len(anc)

    def rate(u):
        # Cramer rate function of the anchor's own order-averaged utility, the empirical law over
        # the 500 prompts (Theorem 1 applied to the joint law of prompt and output).
        best = 0.0
        for j in range(1, 4001):
            lam = j * 0.02
            best = max(best, lam * u - math.log(sum(q * math.exp(lam * v) for v, q in zip(vals, probs))))
        return best
    us = [u0 + i * (0.9 - u0) / 120 for i in range(1, 121)]
    ax.plot([rate(u) for u in us], us, ls=":", color="0.35", lw=1.2, label=r"$\Lambda^*_s(u)$, Thm. 1")
    ax.axhline(0.5, color="0.55", lw=0.7, ls="--")
    # Right-hand end: the Lambda* curve crosses 0.5 near x = 0.1, which struck through a
    # left-anchored label, and the meter's points all sit below the line (caution (ad)).
    ax.annotate("parity", (7e3, 0.5), fontsize=6.3 * F, color="0.4",
                ha="right", va="bottom", xytext=(0, 1.5), textcoords="offset points")
    ax.axhline(u0, color="0.55", lw=0.7, ls="-.")
    ax.annotate("anchor alone", (4e-4, u0), fontsize=6.3 * F, color="0.4", va="top",
                xytext=(0, -1.5), textcoords="offset points")

    met = sorted((a_ for a_ in fl if a_.startswith("met_k")), key=lambda a_: float(a_[5:]))
    x = [float(fl[m]["x_nats"]) for m in met]
    y = [float(fl[m]["level"]) for m in met]
    # Filled = a measured spend, open = a certified bound (review 3, seventh round: the two were drawn
    # alike on one axis, which reads as one kind of quantity).
    ax.plot(x, y, "o-", ms=4.5, lw=1.5, color="#c1443c", label="anchored decoding, spent")
    for m, xv, yv in zip(met, x, y):
        if m in (met[0], met[-1]):
            ax.annotate(f"$k={float(m[5:]):g}$", (xv, yv), fontsize=6.5 * F,
                        xytext=(6, -3) if m == met[-1] else (-26, -9), textcoords="offset points")
    sel = sorted((a_ for a_ in fl if a_.startswith("sel_n")), key=lambda a_: int(a_[5:]))
    xs = [max(float(fl[s]["x_nats"]), 1e-3) for s in sel]
    ys = [float(fl[s]["level"]) for s in sel]
    ax.plot(xs, ys, "s-", ms=4.5, lw=1.5, color="#2f6f9f", mfc="white", mew=1.2,
            label="selection anchoring, bound")
    for s, xv, yv in zip(sel, xs, ys):
        if s in ("sel_n1", "sel_n8", "sel_n64"):
            ax.annotate(f"$n={s[5:]}$", (xv, yv), fontsize=6.5 * F,
                        xytext={"sel_n1": (4, 3), "sel_n64": (6, -8)}.get(s, (5, -10)),
                        textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlim(3e-4, 8e3)
    # Headroom for the legend: at 0.66 its box covered selection's n = 16..64 points (u up to
    # 0.555 at x ~ 3); the box needs about 0.1 of this range above them.
    ax.set_ylim(0.40, 0.72)
    # NOT "realised": only the metered points are measurements; selection's are the closed form
    # log n - (n-1)/n, a BOUND (cautions (ae)/(ah)/(am)). Short, so the two panels' x labels no
    # longer run into each other at \textwidth (a referee saw them collide).
    ax.set_xlabel("nats from the anchor (bound / spent)")
    ax.set_ylabel("level vs. the risky model")
    ax.set_title("(d) what a nat buys, one judge, one opponent", fontsize=7.6 * F, loc="left")
    # The legend labels lost their ", k swept" / ", n swept" tails and the panel gained headroom:
    # at readable type sizes the long three-line legend was as wide as the axes and sat on the
    # n=64 point whichever corner it was pinned to. The swept variable is on the curve labels.
    # NOT "upper left" and NOT frameon=False: the dotted Lambda* curve rises through exactly
    # that corner and struck through "Thm. 1" and "selection anchoring" in the compiled PDF
    # (caution (ad)). Upper right is empty -- the curve exits the top by x~0.5 and the anchored
    # cluster tops out at u=0.52 -- and the opaque frame occludes anything that ever reaches it.
    # 7.0 * F, with the figure printed at \textwidth: F = 6.9/5.5 is exactly the compensation for
    # that placement, so nominal size is printed size. It was 6.6 at 0.70\textwidth, where it
    # measured 5.4pt on the page and was briefly raised to 8.0 instead; widening the figure is the
    # fix that removes the need. RENDER THE PAGE after changing this.
    ax.legend(fontsize=6.6 * F, frameon=True, framealpha=1.0, edgecolor="none",
              loc="upper right", handlelength=1.6, borderaxespad=0.3)
    for _a in (axL, ax):
        _a.tick_params(labelsize=8 * F)
        _a.xaxis.label.set_size(8 * F)
        _a.yaxis.label.set_size(8 * F)
    fig.tight_layout(w_pad=1.8)
    _save(fig, "selection_frontier")


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
    _sty = distinct_styles(len(pairs))
    fig, axes = plt.subplots(1, len(orders), figsize=(6.9, 2.55), sharey=True)
    for ax, o in zip(axes, orders):
        for i, pair in enumerate(pairs):
            c = [(float(r["F"]), float(r["log10_factor"])) for r in rows
                 if r["pair"] == pair and float(r["alpha"]) == o]
            c.sort()
            ax.plot([x for x, _ in c], [y for _, y in c], marker=_sty[i][1], ms=2.6, lw=1.0,
                    color=_sty[i][0], label=pair if o == orders[0] else None)
        ax.axhline(0.0, color="0.4", lw=0.7, ls=":")
        ax.set_title(rf"$\alpha = {o:.0f}$")
    axes[0].set_ylabel(r"$\log_{10}$ times safer, matched utility")
    # Twelve entries in one column, unframed, inside a 2.2in axes: the legend was taller than the
    # panel it sat in, so it overflowed onto the alpha=2 title and the y-axis label and every curve
    # ran through its text. Put it under the panels instead, where nothing is drawn.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", frameon=False, ncol=6, fontsize=6.5,
               bbox_to_anchor=(0.5, 0.055), handlelength=1.3, handletextpad=0.4,
               columnspacing=1.1, borderaxespad=0.0)
    fig.supxlabel("$F$, the fraction of the unconstrained ceiling the audited decoder already has",
                  fontsize=8, y=0.008)
    fig.tight_layout(rect=(0, 0.19, 1, 1))
    _save(fig, "order_no_collapse")


def imitation_cost():
    """Propositions 3 and 5, in the two shapes they make claims about.

    (a) the rate. The certificate is written against k nats per token; the decoder spends the
    imitation cost and stops, so above the crossover the allowance is unreachable.
    (b) the shape. Proposition 5 needs an O(1)-budget policy to put its spend on O(1) steps; the
    deployed rule spreads it over the sequence, barely above the uniform diagonal.
    """
    rows = [r for r in csv.DictReader(open(RESULTS / "imitation_cost.csv"))
            if r["prompt_class"] == "ordinary"]
    lz = [r for r in csv.DictReader(open(RESULTS / "imitation_lorenz.csv"))
          if r["prompt_class"] == "ordinary"]
    ks = [float(r["k"]) for r in rows]
    realised = [float(r["realised_rate_nats_per_token"]) for r in rows]
    sat = float(rows[-1]["imitation_rate_nats_per_token"])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.6, 2.5))
    axL.plot(ks, ks, color="0.55", lw=1.1, ls="--", label="cap $k$ (what is certified)")
    axL.plot(ks, realised, "o-", color="C3", lw=1.5, ms=3.4, label="realised rate (measured)")
    axL.axhline(sat, color="C0", lw=1.0, ls=":")
    axL.annotate(f"imitation rate {sat:.3f}", xy=(0.12, sat), xytext=(0.115, sat * 1.35),
                 fontsize=6.4, color="C0")
    axL.axvline(sat, color="0.3", lw=0.8, ls="-.")
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_ylim(0.03, 30)
    axL.annotate("meter binds", xy=(0.105, 0.038), fontsize=6.4, color="0.3")
    axL.annotate("allowance unreachable", xy=(1.05, 0.038), fontsize=6.4, color="0.3")
    axL.set_xlabel("budget $k$ (nats per token)")
    axL.set_ylabel("nats per token")
    axL.set_title("(a) what the decoder spends", fontsize=8)
    # An OPAQUE frame, not frameon=False: the axvline at the imitation rate runs up through the
    # upper-left corner and struck through "what is certified" in the compiled PDF (caution (ad)).
    axL.legend(frameon=True, framealpha=1.0, edgecolor="none", loc="upper left", fontsize=6.6)
    axL.grid(alpha=0.25, lw=0.5)

    axR.plot([0, 1], [0, 1], color="0.55", lw=1.1, ls="--", label="uniform over steps")
    # k = 3 and k = 20 lie on top of each other, which is the point: once the meter stops binding
    # the shape of the spend stops depending on the cap. Dashed so both are visible.
    budget_lines = {}
    for k, ls, col in (("0.5", "-", "C0"), ("20", "-", "C2"), ("3", "--", "C1")):
        v = [(float(r["frac_of_steps"]), float(r["frac_of_spend"])) for r in lz if r["k"] == k]
        v.sort()
        budget_lines[k], = axR.plot([0] + [x for x, _ in v], [0] + [y for _, y in v],
                                    lw=1.4, ls=ls, color=col, label=f"$k = {k}$")
    axR.plot([0, 0.02, 1], [0, 1, 1], color="C3", lw=1.2, ls=":",
             label="what Proposition 5 needs")
    axR.set_xlabel("fraction of steps, busiest first")
    axR.set_ylabel("share of the spend")
    axR.set_title("(b) where it spends it", fontsize=8)
    # The DRAW order has to stay 0.5, 20, 3 so the dashed k=3 lands on top of the solid k=20 and
    # both are visible; the legend does not, and read "k = 0.5, k = 20, k = 3" until 2026-09-17.
    h, l = axR.get_legend_handles_labels()
    by = dict(zip(l, h))
    order = ["uniform over steps"] + [f"$k = {k}$" for k in sorted(budget_lines, key=float)] + \
            ["what Proposition 5 needs"]
    assert set(order) == set(l), (order, l)
    axR.legend([by[x] for x in order], order, frameon=False, loc="lower right")
    axR.grid(alpha=0.25, lw=0.5)
    _save(fig, "imitation_cost")


def selection_forest_rows():
    """The rows of Figure~\\ref{fig:breadth}, read out of results/*.csv and nothing else.

    Split out of the drawing code on purpose. When a band moves from prose into a figure, the
    guard that used to grep the prose for it goes quiet and the number becomes unchecked -- which
    is caution (ag) with the deletion done by a figure instead of a page trim. Tests assert
    against THIS, so a band is guarded wherever the paper chooses to print it.

    Returns (items, groups): items are (label, (gain, lo, hi), cost_nats, certified, gate_passed),
    groups are (index the group starts at, group heading).
    """
    def rows_of(name):
        return list(csv.DictReader(open(RESULTS / name)))

    def band(r, g="gain"):
        return float(r[g]), float(r[f"{g}_lo95"]), float(r[f"{g}_hi95"])

    JB = "Phi-3.5-mini-instruct"
    # Every judged row on recovered text (results/forest_deecho_note.md): the entry gate re-measured
    # on the true generations (an empty draw that carried the prompt's tail looked non-empty, so the
    # committed CSV passed Comma-7B at 3.0% where its text is empty on 18.4%), and the gains
    # re-judged with the judge reading the repaired text, same picks, same judges, same seed.
    breadth = {r["anchor"]: r for r in rows_of("selection_breadth_rejudged.csv") if JB in r["judge"]}
    scal64 = next(r for r in rows_of("selection_scaling_deecho.csv")
                  if JB in r["judge"] and int(float(r["n"])) == 64)

    def dom(name, tag=""):
        rs = [r for r in rows_of(name) if JB in r["judge"]]
        mx = max(int(float(r["n"])) for r in rs)
        return next(r for r in rs if int(float(r["n"])) == mx)

    def verif(name, arm, n):
        return next(r for r in rows_of(name)
                    if r["arm"].startswith(arm) and int(float(r["n"])) == n)

    # (label, (gain, lo, hi), cost in nats, certified?, entry gate passed)
    items, groups = [], []

    def add(label, r, cost, certified=True, gate=True, g="gain"):
        items.append((label, band(r, g), cost, certified, gate))

    groups.append((len(items), "six anchors, 500 prompts, judge B"))
    for a in ("Pleias-1.2B", "KL3M-1.7B", "Pleias-3B", "Comma-7B (1T tokens)",
              "TinyComma-1.8B (audited)", "Comma-7B"):
        r = breadth[a]
        add(f"{a}, $n=8$", r, math.log(int(float(r["n"]))), gate=r["entry_gate"] == "PASS")
    # The n=64 arm's own first draw, de-echoed, as counted by analysis/h2h_length_control.py
    # ("served words, sel_n1 ... empty on 64 prompts"): 12.8%, above the same 5% gate.
    lc = {r["quantity"]: r for r in rows_of("h2h_length_control.csv")}
    e64 = int(lc["served words, sel_n1"]["note"].split("empty on ")[1].split()[0]) / int(
        lc["served words, sel_n1"]["n"])
    add("TinyComma-1.8B (audited), $n=64$", scal64, math.log(64), gate=e64 < 0.05)

    groups.append((len(items), "other workloads, judge B, $n = 8$"))
    add("AlpacaEval-805, TinyComma-1.8B", dom("selection_scaling_alpaca_deecho.csv"), math.log(8))
    add("AlpacaEval-805, Comma-7B", dom("selection_scaling_alpaca_comma7b_deecho.csv"), math.log(8))
    add("MT-Bench-80, TinyComma-1.8B", dom("selection_scaling_mtbench_deecho.csv"), math.log(8))

    groups.append((len(items), "exact match, no judge, Comma-7B"))
    V, T = "selection_verifiable_comma7b.csv", "selection_verifiable_tqa_comma7b.csv"
    add("GSM8K, majority vote, $n=32$", verif(V, "majority", 32), math.log(32))
    add("GSM8K, pointwise reward, $n=64$", verif(V, "pointwise", 64), math.log(64))
    add("TriviaQA, majority vote, $n=64$", verif(T, "majority", 64), math.log(64))
    add("TriviaQA, pointwise reward, $n=16$", verif(T, "pointwise", 16), math.log(16))

    # No metered row since 2026-09-24. This figure is selection against its OWN anchor; the
    # head-to-head against the meter is Figure 1(b) and Table 1, both order-averaged against one
    # opponent. A single-order meter point here set one construction beside another, and a
    # referee read the figure as the head-to-head it is not.
    return items, groups


def selection_breadth_forest():
    """Every judged and exact-match gain selection anchoring has been measured at, with its 95%
    interval and the certificate that bought it, against the metered decoder's best arm.

    This replaces a paragraph. Fifteen effect estimates were being carried as prose, which is a
    figure's job; read as a forest plot the shape of the evidence is immediate -- positive nearly
    everywhere for 2 to 4 nats, against 171.3 for the comparator -- and so are the two places it
    fails, both of which are identifiable rather than mysterious.
    """
    items, groups = selection_forest_rows()

    # A group header gets its OWN line. Sharing a line with the row above it is caution (ad) in
    # miniature: the first draft printed three headers straight through three row labels.
    starts = dict(groups)
    slots, y = [], 0.0
    for i in range(len(items)):
        if i in starts:
            y -= 1.0
            slots.append(("head", y, starts[i]))
            y -= 0.18
        y -= 1.0
        slots.append(("row", y, i))
    top, bot = -0.2, y - 0.8

    # DRAW AT THE WIDTH IT WILL PRINT AT. The label gutters live outside the axes, and
    # bbox_inches="tight" expands the canvas to fit them: at figsize 6.9 the saved PDF came out
    # 9.09in wide, so placing it at ICLR's 5.984in \textwidth shrank everything by 0.658 and 7pt
    # labels printed at 5.0pt -- caution (aj) again, and neither arithmetic on figsize nor the
    # advance-width ratio caught it (the ratio compares DejaVu Sans against Times and overstates
    # by the font-width difference). Reserve the gutters INSIDE a 6.0in canvas instead, so the
    # shrink is ~1 and a drawn point is a printed point. Measure the SAVED file, never figsize.
    fig, ax = plt.subplots(figsize=(6.0, 1.98))
    fig.subplots_adjust(left=0.345, right=0.875, bottom=0.145, top=0.985)
    # x in AXES fraction, y in data: the label gutters sit outside the data area by construction,
    # so no interval can ever print through a row label (the first draft's MT-Bench and TriviaQA
    # bars both did, reaching -0.0875 and -0.068 into a column anchored at -0.052).
    gut = ax.get_yaxis_transform()
    for kind, y, payload in slots:
        if kind == "head":
            ax.axhline(y + 0.62, color="#cccccc", lw=0.6, zorder=0)
            ax.text(-0.02, y, payload, ha="right", va="center", fontsize=6.6,
                    style="italic", color="#555555", transform=gut)
            continue
        label, (g, lo, hi), cost, certified, gate = items[payload]
        neg = hi is not None and hi < 0
        col = "#b02318" if neg else ("#1f4e79" if certified else "#6b6b6b")
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=col, lw=1.3, solid_capstyle="butt", zorder=2)
            ax.plot([lo, lo, None, hi, hi], [y - .17, y + .17, y, y - .17, y + .17],
                    color=col, lw=1.0, zorder=2)
        ax.plot([g], [y], marker="o", ms=4.4, color=col if gate else "white",
                mec=col, mew=1.2, zorder=3)
        ax.text(-0.02, y, label, ha="right", va="center", fontsize=6.9, transform=gut)
        ax.text(1.145, y, f"{cost:.2f}" if certified else f"{cost:.1f} spent",
                ha="right", va="center", fontsize=6.9, transform=gut,
                color=col if certified else "#b02318")

    ax.axvline(0, color="black", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(1.145, top - 0.5, "certificate, nats", ha="right", va="center", fontsize=6.6,
            style="italic", color="#555555", transform=gut)

    ax.set_xlim(-0.098, 0.288)
    ax.set_ylim(bot + 0.25, top)
    ax.set_yticks([])
    ax.set_xlabel("gain over the same arm's own anchor-alone control, 95% CI", fontsize=7.6)
    ax.tick_params(axis="x", labelsize=6.9)
    ax.set_xticks([-0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.18, lw=0.5)
    ax.set_axisbelow(True)
    _save(fig, "selection_breadth_forest")



# --------------------------------------------------------------------------------------------
# Main-text evidence figures, 2026-09-19. Both replace prose that carried the paper's two most
# important non-utility claims. Data extraction is split from drawing so a test can read the
# plotted rows: caution (al) -- a number that moves from prose into a figure takes its guard with
# it, or the guard passes by never running.


def safety_rows():
    """(label, [(n, nv_recall_mean)], baseline) per extraction arm, read from results/."""
    import glob
    arms = []
    want = [("selection_extraction.csv", "audited anchor, attacker's scorer"),
            ("selection_extraction_n256.csv", "audited anchor, to $n{=}256$"),
            ("selection_extraction_comma7b.csv", "Comma-7B"),
            ("selection_extraction_comma1t.csv", "Comma-7B (1T)"),
            ("selection_extraction_kl3m17b.csv", "KL3M-1.7B"),
            ("selection_extraction_pleias12b.csv", "Pleias-1.2B"),
            ("selection_extraction_pleias3b.csv", "Pleias-3B"),
            ("selection_extraction_paraphrase.csv", "paraphrase event"),
            ("selection_extraction_70b_hp2.csv", "Llama-3.1-70B, pre-trained memoriser")]
    for fname, label in want:
        p = RESULTS / fname
        if not p.exists():
            continue
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        if "nv_recall_mean" not in rows[0]:
            continue
        pos = sorted(((int(r["n"]), float(r["nv_recall_mean"])) for r in rows
                      if int(r["n"]) >= 1), key=lambda t: t[0])
        base = [float(r["nv_recall_mean"]) for r in rows if r["n"] == "-1"]
        arms.append((label, pos, base[0] if base else None))
    if not arms:
        raise FileNotFoundError("no selection_extraction*.csv")
    return arms


def contamination_rows(fname="selector_n256.csv"):
    """(anchor, event, base_rate, realised amplification) at n=64, where the premise FAILS.

    Read from feat-179's corrected-selector arm (results/selector_n256.csv), or with
    fname="selector_redraw.csv" from feat-182's independent re-draw, whose registration requires the
    factor to be shown per draw, both draws, never pooled. The earlier contaminated_anchor.csv was
    measured with a selector that ranked partly on padding (caution (ba)) and is never shown."""
    p = RESULTS / fname
    if not p.exists():
        raise FileNotFoundError(p)
    out = []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["n"] == "64" and r["amplification_vs_n1"]:
            out.append((r["anchor"], r["event"], float(r["base_rate_n1"]),
                        float(r["amplification_vs_n1"])))
    return sorted(out, key=lambda t: t[3])


def safety_envelope():
    arms, contam, redraw = safety_rows(), contamination_rows(), contamination_rows("selector_redraw.csv")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.98, 1.48))
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.185, top=0.87, wspace=0.28)

    # (a) Every selection arm sits exactly on zero, so nine legend entries would be nine labels
    # for one line. Draw them all -- the markers show which n each arm covers -- and let ONE entry
    # speak for them, with the memoriser baselines labelled in place at the right edge. The first
    # version put a 9-entry legend and a floating annotation in the same corner and they collided
    # (caution (ad): render the page and look at it).
    styles = distinct_styles(len(arms))
    for i, ((label, pos, base), (c, m)) in enumerate(zip(arms, styles)):
        ax.plot([n for n, _ in pos], [v for _, v in pos], marker=m, color=c, ms=4.2,
                lw=1.0, alpha=0.9, zorder=3,
                label=f"all {len(arms)} arms, every $n$: $0.0000$" if i == 0 else None)
    for b in sorted({b for _, _, b in arms if b}):
        ax.axhline(b, ls="--", lw=0.9, color="0.35", zorder=1)
        ax.annotate(f"${b:.4f}$", xy=(300, b), fontsize=6.0, color="0.15",
                    bbox=dict(fc="white", ec="none", pad=0.8),
                    va="center", ha="right")
    ax.annotate("memoriser alone,\nsame passages and seeds", xy=(1.05, 0.515),
                fontsize=6.3, color="0.2", va="top")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("$n$, completions drawn from the anchor")
    ax.set_ylabel("near-verbatim recall")
    ax.set_ylim(-0.035, 0.52)
    ax.set_xlim(0.78, 340)
    ax.set_title("(a) selection reproduces nothing, at every $n$", fontsize=8)
    ax.legend(loc="lower left", fontsize=6.4, frameon=False, bbox_to_anchor=(0.0, 0.055))

    # (b) Where the premise fails. The first version was 17 labelled bars in a 1.5in panel, whose
    # row labels would have printed at about 4.8pt -- below the legibility floor (caution (af)), and
    # the anchor NAMES are not the claim: the claim is that every realised amplification sits near
    # 1 while the certificate permits 64. A strip plot carries that and needs no row labels.
    import random as _rnd
    _rnd.seed(0)
    for ev, c, mk in (("E_001", "C0", "o"), ("E_08", "C3", "s")):
        # feat-182: the second, independent draw is drawn hollow beside the first, never pooled
        for rows, fill, lab in ((contam, True, ""), (redraw, False, ", second draw")):
            xs = [a for _, e, _, a in rows if e == ev]
            if not xs:
                continue
            ys = [0.5 + 0.30 * (_rnd.random() - 0.5) + (0.22 if ev == "E_08" else -0.22) for _ in xs]
            bx.scatter(xs, ys, s=17, marker=mk, alpha=0.85, zorder=3,
                       facecolors=c if fill else "none", edgecolors=c, linewidths=0.8,
                       label=({"E_001": "recall 0.01", "E_08": "recall 0.8"}[ev] + lab))  # E_001 is >= 0.01
    bx.axvline(64, color="0.25", ls="--", lw=1.1, zorder=4)
    bx.annotate("$n=64$: what the\ncertificate permits", xy=(56, 1.22), ha="right",
                fontsize=6.3, color="0.15", va="top")
    bx.axvspan(min(a for *_, a in contam), max(a for *_, a in contam), color="0.85",
               alpha=0.45, zorder=0)
    # matplotlib is not LaTeX: "--" prints as two hyphens, not an en-dash (caution (ad)).
    lo2, hi2 = min(a for *_, a in redraw), max(a for *_, a in redraw)
    bx.annotate(f"realised: {min(a for *_, a in contam):.1f} to {max(a for *_, a in contam):.1f};\n"
                f"second draw {lo2:.1f} to {hi2:.1f}",
                xy=(0.87, 0.93), fontsize=6.3, color="0.15", ha="left", va="bottom")
    bx.set_yticks([])
    bx.set_ylim(0, 1.25)          # headroom: the two-draw annotation sat on the recall-0.01 points
    bx.set_xscale("log")
    bx.set_xlim(0.8, 140)
    bx.set_xlabel("realised amplification at $n=64$, twelve contaminated anchors")
    bx.set_title("(b) and degrades gracefully where it fails", fontsize=8)
    # Three keys, not four: a two-column legend of both draws covered the recall-0.01 points
    # (caution (ad)), so the second draw is one proxy entry for the hollow style.
    from matplotlib.lines import Line2D
    keys = [Line2D([], [], marker="o", ls="", color="C0", ms=4, label="recall 0.01"),
            Line2D([], [], marker="s", ls="", color="C3", ms=4, label="recall 0.8"),
            Line2D([], [], marker="s", ls="", mfc="none", mec="0.3", ms=4, label="hollow: second draw")]
    bx.legend(handles=keys, loc="lower right", fontsize=5.8, frameon=True, framealpha=1.0,
              facecolor="white", edgecolor="none", handletextpad=0.3,
              bbox_to_anchor=(1.0, 0.02))
    _save(fig, "safety_envelope")


def judge_free_rows():
    """{task: ({rule: [(n, acc, lo, hi)]}, {baseline: acc})} for the two no-judge arms.

    The rule is matched EXPLICITLY and anything unrecognised raises. The first version classified
    with `if arm.startswith("majority") else "pointwise"`, which swept the two mandatory k=-1
    baselines ("risky model alone", greedy and sampled) into the pointwise curve and drew it
    spiking to 0.79 at n=1, where by construction both rules must equal the anchor's own first
    draw. An `else` branch over a data-driven column is how a baseline becomes a result.
    """
    out = {}
    for task, fname in (("GSM8K", "selection_verifiable_comma7b.csv"),
                        ("TriviaQA", "selection_verifiable_tqa_comma7b.csv")):
        p = RESULTS / fname
        if not p.exists():
            raise FileNotFoundError(p)
        by, base = {}, {}
        for r in csv.DictReader(open(p, encoding="utf-8")):
            arm, n = r["arm"], int(r["n"])
            pt = (n, float(r["acc"]), float(r["acc_lo95"]), float(r["acc_hi95"]))
            if arm.startswith("majority"):
                by.setdefault("majority vote", []).append(pt)
            elif arm.startswith("pointwise"):
                by.setdefault("pointwise reward", []).append(pt)
            elif arm.startswith("risky model alone"):
                base[arm] = float(r["acc"])
            else:
                raise ValueError(f"{fname}: unclassified arm {arm!r}")
        for rule, pts in by.items():
            pts.sort()
            assert pts[0][0] == 1, (fname, rule)
        n1 = {rule: pts[0][1] for rule, pts in by.items()}
        assert len(set(n1.values())) == 1, \
            f"{fname}: the rules disagree at n=1, where neither has selected anything: {n1}"
        out[task] = ({k: sorted(v) for k, v in by.items()}, base)
    return out


def judge_free_metered_rows():
    """The judge-free HEAD-TO-HEAD: both mechanisms, one anchor, one task, no judge anywhere.

    results/verifiable_metered_tqa.csv is the only arm in the paper where the two mechanisms are
    compared on an objective metric at a SHARED anchor (TinyComma-1.8B is the one openly licensed
    safe model whose vocabulary the metered decoder can fuse with). Returned as
    (selection points, metered points, baselines), each point (nats, acc, lo, hi).

    The x value is not the same KIND of quantity for the two arms and the caller must say so:
    selection's is `kl_nats`, the CLOSED FORM log n - (n-1)/n, while the metered decoder's is
    `realised_nats`, a measured mean. That is the conservative direction -- it charges selection a
    bound and the meter only what it spent -- but it is not like for like (caution (am)).
    """
    p = RESULTS / "verifiable_metered_tqa.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    sel, met, base = [], [], {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        acc = (float(r["acc"]), float(r["acc_lo95"]), float(r["acc_hi95"]))
        if r["mechanism"].startswith("selection"):
            n = int(r["arm"].split("=")[1])
            if n > 1:
                sel.append((float(r["kl_nats"]), *acc))
        elif r["arm"] == "k=-1":
            base["risky"] = acc[0]
        elif r["arm"] == "k=0":
            base["anchor"] = acc[0]
        else:
            met.append((float(r["realised_nats"]), *acc, r["arm"], float(r["certificate_nats"])))
    assert sel and met and len(base) == 2, (len(sel), len(met), base)
    # The meter's best arm must BE the unconstrained risky model, which is the whole point of the
    # panel; if that ever stops being true the caption's claim has to change with it.
    assert abs(max(m[1] for m in met) - base["risky"]) < 1e-9, \
        "the metered decoder's best arm is no longer the risky model's own accuracy"
    return sorted(sel), sorted(met), base


def judge_free():
    data = judge_free_rows()
    fig = plt.figure(figsize=(5.98, 1.62))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 1.30), wspace=0.34)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    cx = fig.add_subplot(gs[2])
    fig.subplots_adjust(left=0.068, right=0.995, bottom=0.215, top=0.855)
    style = {"majority vote": ("C0", "o", "-"), "pointwise reward": ("C3", "s", "--")}
    for axi, (task, (by, base)) in zip(axes, data.items()):
        for rule, pts in by.items():
            c, m, ls = style[rule]
            ns = [n for n, _, _, _ in pts]
            axi.plot(ns, [a for _, a, _, _ in pts], marker=m, color=c, ls=ls, ms=3.6,
                     lw=1.2, label=rule, zorder=3)
            axi.fill_between(ns, [lo for _, _, lo, _ in pts], [hi for _, _, _, hi in pts],
                             color=c, alpha=0.13, lw=0, zorder=1)
        if base:
            b = max(base.values())
            axi.axhline(b, color="0.35", ls=":", lw=1.0, zorder=2)
            axi.annotate(f"risky alone, $k={{-1}}$: ${b:.3f}$", xy=(1.05, b),
                         fontsize=6.2, color="0.2", va="bottom")
            lo = min(v for _, (by_, _) in [(0, (by, base))] for pts in by_.values()
                     for _, _, v, _ in pts)
            axi.set_ylim(lo - 0.03, b + 0.075)
        axi.set_xscale("log", base=2)
        axi.set_xlabel("$n$")
        axi.grid(alpha=0.22, lw=0.5)
    axes[0].set_ylabel("exact match")
    axes[0].set_title("(a) GSM8K", fontsize=7.6)
    axes[1].set_title("(b) TriviaQA", fontsize=7.6)
    # caution (ad): at "lower right" both curves ran through the legend text (seen on the rendered
    # page, 2026-09-24). Upper left, under the risky model's rule, is empty at every n <= 8.
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.84), fontsize=6.2, frameon=True,
                   framealpha=1.0, facecolor="white", edgecolor="none", handletextpad=0.4,
                   borderpad=0.15, labelspacing=0.2)

    # (c) the two mechanisms on one objective axis, one shared anchor, no judge anywhere.
    sel, met, base = judge_free_metered_rows()
    cx.axhline(base["anchor"], color="0.55", ls=":", lw=0.9, zorder=1)
    cx.axhline(base["risky"], color="0.35", ls=":", lw=0.9, zorder=1)
    cx.plot([m[0] for m in met], [m[1] for m in met], marker="^", color="C3", ls="--",
            ms=3.6, lw=1.2, label="metered decoder", zorder=3)
    cx.plot([p[0] for p in sel], [p[1] for p in sel], marker="o", color="C0", ls="-",
            ms=3.6, lw=1.2, label="selection (majority vote)", zorder=3)
    cx.fill_between([p[0] for p in sel], [p[2] for p in sel], [p[3] for p in sel],
                    color="C0", alpha=0.13, lw=0, zorder=1)
    top = max(m for m in (base["risky"],))
    k20 = max(met, key=lambda m: m[1])
    cx.annotate(f"$k{{=}}20$: the risky model,\ncertified at ${k20[5]:.0f}$ nats",
                xy=(k20[0], k20[1]), xytext=(0.26, 0.58), textcoords="axes fraction",
                fontsize=6.4, color="0.15", ha="left", va="top",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="0.45"))
    # lower right, under the rule: at the left the label's box hid the k=0.5 arm, which spends
    # 0.0109 nats and sits on the anchor's line (seen on the rendered page, 2026-09-24)
    cx.annotate("anchor alone", xy=(0.98, base["anchor"]), xycoords=("axes fraction", "data"),
                xytext=(0, -1.5), textcoords="offset points",
                fontsize=6.2, color="0.35", ha="right", va="top")
    cx.set_xscale("log")
    cx.set_xlabel("nats from the anchor (bound / spent)")
    cx.set_ylim(base["anchor"] - 0.085, top + 0.30)
    cx.grid(alpha=0.22, lw=0.5)
    cx.set_title("(c) both mechanisms, one anchor", fontsize=7.6)
    # caution (ad): a legend pinned inside a small axes collides with the data. Here the risky
    # model's dotted rule runs straight through the second entry, so the box is opaque and the
    # ylim carries the headroom that keeps it off the curves.
    cx.legend(loc="upper left", fontsize=6.4, frameon=True, framealpha=1.0, facecolor="white",
              edgecolor="none", handletextpad=0.4, borderpad=0.15, labelspacing=0.2)
    _save(fig, "judge_free")


# --------------------------------------------------------------------------------------------
# v10 (2026-09-24): the restructure around the contribution. Two main-text figures, each with its
# rows split from its drawing so a test reads what is plotted (caution (al)).


def meter_horns_rows():
    """(pairs, (lo, hi), imit): per onset pair (name, s(x), [(k/s(x), recall)]); the range of the
    nine onset ratios in results/onset_table.csv; and per budget the metered decoder's
    (k, realised rate, imitation rate, spend, K) on ordinary traffic (results/imitation_cost.csv)."""
    import statistics as _st
    from analysis.onset import curve, load_pairs
    pairs = []
    for name, comp, per in load_pairs(str(RESULTS / "onset_pairs.tsv")):
        if not (REPO / comp).exists():
            raise FileNotFoundError(REPO / comp)
        sx = _st.median(float(r["s_mean"]) for r in csv.DictReader(open(REPO / per)))
        c = curve(str(REPO / comp), "single", 0)
        pairs.append((name, sx, [(k / sx, c[k]) for k in sorted(c)]))
    ratios = [float(r["ratio"]) for r in csv.DictReader(open(RESULTS / "onset_table.csv"))
              if not r["pair"].startswith("ALL")]
    imit = [(float(r["k"]), float(r["realised_rate_nats_per_token"]),
             float(r["imitation_rate_nats_per_token"]), float(r["spend_nats"]), float(r["budget_K"]))
            for r in csv.DictReader(open(RESULTS / "imitation_cost.csv"))
            if r["prompt_class"] == "ordinary"]
    assert len(pairs) == len(ratios) == 9 and imit, (len(pairs), len(ratios))
    return pairs, (min(ratios), max(ratios)), sorted(imit)


def meter_horns():
    """Figure 2: both horns of Proposition 3, measured on the deployed meter. (a) Extraction begins
    where Proposition 2 says the certificate stops saying anything. (b) Where the budget is large
    enough to buy anything, the meter spends the imitation rate and the rest of the cap is unused."""
    pairs, (r_lo, r_hi), imit = meter_horns_rows()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.98, 1.92))
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.215, top=0.885, wspace=0.27)
    ax.axvspan(r_lo, r_hi, color="#c1443c", alpha=0.12, lw=0, zorder=0)
    for name, sx, pts in pairs:
        ax.plot([x for x, _ in pts], [y for _, y in pts], "-", color="#2f4f6f", lw=1.0,
                alpha=0.75, marker="o", ms=2.2, zorder=2)
    ax.axvline(1.0, color="0.2", lw=1.0, ls="--", zorder=3)
    ax.annotate(f"nine onsets:\n{r_lo:.2f} to {r_hi:.2f}", xy=(r_hi, 0.205), xytext=(1.24, 0.215),
                fontsize=6.6, color="#8c3230", va="top")
    ax.annotate("$k = s(x)$: certificate\nvacuous for $x$", xy=(1.0, 0.16), xytext=(0.43, 0.215),
                fontsize=6.6, color="0.15", va="top")
    ax.set_xlim(0.4, 2.45)
    ax.set_ylim(-0.005, 0.26)
    ax.set_xlabel("budget in units of the work's surprisal rate, $k/s(x)$", fontsize=7.4)
    ax.set_ylabel("near-verbatim recall", fontsize=7.4)
    ax.set_title("(a) leakage begins at $k \\approx s(x)$, nine pairs", fontsize=7.6, loc="left")

    ks = [k for k, *_ in imit]
    bx.plot(ks, ks, color="0.55", lw=1.1, ls="--", label="cap $k$ (certified)")
    bx.plot(ks, [r for _, r, *_ in imit], "o-", color="#c1443c", lw=1.5, ms=3.2,
            label="spend per token (measured)")
    sat = imit[-1][2]
    bx.axhline(sat, color="#2f6f9f", lw=1.0, ls=":")
    bx.annotate(f"imitation rate {sat:.3f}", xy=(0.105, sat), xytext=(0.105, sat * 1.4),
                fontsize=6.6, color="#2f6f9f")
    kk, _, _, spend, K = imit[-1]
    bx.annotate(f"$k={kk:g}$: spends {spend:.1f}\nof {K:.0f} nats", xy=(kk, sat),
                xytext=(2.2, 0.075), fontsize=6.6, color="0.15",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="0.45"))
    bx.set_xscale("log"); bx.set_yscale("log")
    bx.set_ylim(0.03, 30)
    bx.set_xlabel("budget $k$, nats per token", fontsize=7.4)
    bx.set_ylabel("nats per token", fontsize=7.4)
    bx.set_title("(b) the meter spends its imitation rate, not its cap", fontsize=7.6, loc="left")
    bx.legend(frameon=True, framealpha=1.0, edgecolor="none", loc="upper left", fontsize=6.6)
    for a_ in (ax, bx):
        a_.tick_params(labelsize=7)
        a_.grid(alpha=0.22, lw=0.5)
    _save(fig, "meter_horns")


def h2h_forest_rows():
    """Rows of the head-to-head robustness figure, every one on the repaired text (caution (bc)):
    (group, label, (D3, lo95, hi95)), D3 = selection's order-averaged gain at n = 64 minus the
    meter's at k = 10 unless the label names another budget, paired over prompts, read from
    results/*.csv only."""
    def d3(name):
        r = next(x for x in csv.DictReader(open(RESULTS / name)) if x["quantity"].startswith("D3"))
        return float(r["value"]), float(r["lo95"]), float(r["hi95"])
    rj = {(r["pass"], r["judge"].split()[0]): (float(r["value_deecho"]), float(r["lo"]), float(r["hi"]))
          for r in csv.DictReader(open(RESULTS / "deecho_rejudge.csv")) if r["quantity"].startswith("D3")}
    lc = {r["quantity"]: (float(r["value"]), float(r["lo95"]), float(r["hi95"]))
          for r in csv.DictReader(open(RESULTS / "h2h_length_control.csv")) if r["lo95"]}
    G1, G2, G3 = "fresh draws, judge B", "same texts, other judges", "post-hoc controls, judge B"
    G4, G5 = "stronger fixed opponents, judge B", "other workloads, binding budget, judge B"
    rows = [("headline", "headline: judge B, 500 prompts", d3("order_averaged_h2h_deecho.csv")),
            (G1, "selection re-drawn, seeds 52-54", d3("order_averaged_h2h_seed52_deecho.csv")),
            (G1, "every arm re-drawn, seeds 82-84", d3("order_averaged_h2h_replic_opp.csv")),
            (G1, "same, committed opponent", d3("order_averaged_h2h_replic.csv")),
            (G2, "C: Llama-3.1-8B (the opponent)", rj[("ladder opp Llama-3.1-8B", "C")]),
            (G2, "D: Qwen2.5-72B", rj[("panel k=10", "D")]),
            (G2, "E: Mixtral-8x7B", rj[("panel k=10", "E")]),
            (G2, "F: Qwen2.5-14B", rj[("panel k=10", "F")]),
            (G2, "G: gemma-2-27b", rj[("panel k=10", "G")]),
            (G3, "length-controlled", lc["D3, length-controlled"]),
            (G3, "both anchor controls non-empty (414)",
             lc["D3 on prompts whose two anchor controls are non-empty"]),
            (G4, "Qwen2.5-0.5B-Instruct", rj[("ladder opp Qwen2.5-0.5B", "B")]),
            (G4, "Qwen2.5-1.5B-Instruct", rj[("ladder opp Qwen2.5-1.5B", "B")]),
            (G4, "Qwen2.5-3B-Instruct", rj[("ladder opp Qwen2.5-3B", "B")]),
            (G4, "Qwen2.5-14B-Instruct", rj[("ladder opp Qwen2.5-14B", "B")]),
            (G5, "ours, 850 prompts, $k=0.9$", rj[("workload ours (Arm C) k=0.9", "B")]),
            (G5, "public-domain books, $k=0.9$", rj[("workload Gutenberg k=0.9", "B")]),
            (G5, "BookMIA unseen books, $k=1$", rj[("workload unseen books k=1.0", "B")]),
            (G5, "CoTaEval NewsQA, $k=1.4$", rj[("workload CoTaEval-QA k=1.4", "B")]),
            (G5, "MT-Bench-80, $k=1$", rj[("workload MT-Bench k=1.0", "B")]),
            (G5, "AlpacaEval-805, $k=1$", rj[("workload AlpacaEval k=1", "B")])]
    return rows


def h2h_forest():
    """Figure 3: the head-to-head difference under every perturbation run, one row each."""
    rows = h2h_forest_rows()
    slots, y, prev = [], 0.0, None
    for i, (g, *_rest) in enumerate(rows):
        if g != prev and g != "headline":
            y -= 1.0
            slots.append(("head", y, g))
            y -= 0.12
        prev = g
        y -= 1.0
        slots.append(("row", y, i))
    fig, ax = plt.subplots(figsize=(5.98, 3.05))
    fig.subplots_adjust(left=0.33, right=0.77, bottom=0.085, top=0.995)
    gut = ax.get_yaxis_transform()
    for kind, yy, payload in slots:
        if kind == "head":
            ax.axhline(yy + 0.6, color="#cccccc", lw=0.6, zorder=0)
            ax.text(-0.02, yy, payload, ha="right", va="center", fontsize=6.6, style="italic",
                    color="#555555", transform=gut)
            continue
        g, label, (v, lo, hi) = rows[payload]
        col = "#1f4e79" if lo > 0 else ("#b02318" if hi < 0 else "#7a7a7a")
        ax.plot([lo, hi], [yy, yy], color=col, lw=1.3, solid_capstyle="butt", zorder=2)
        ax.plot([v], [yy], marker="D" if g == "headline" else "o", ms=4.2, color=col, zorder=3)
        ax.text(-0.02, yy, label, ha="right", va="center", fontsize=6.9, transform=gut,
                weight="bold" if g == "headline" else "normal")
        ax.text(1.02, yy, f"{v:+.4f} [{lo:+.4f}, {hi:+.4f}]".replace("-", "\u2212"), ha="left", va="center",
                fontsize=6.4, transform=gut, color=col)
    ax.axvline(0, color="black", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_ylim(y - 0.7, 0.3)
    ax.set_xlim(-0.09, 0.17)
    ax.set_yticks([])
    ax.set_xlabel("selection ($n=64$) minus the metered decoder: difference of gains, 95% CI",
                  fontsize=7.2)
    ax.tick_params(axis="x", labelsize=6.9)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.18, lw=0.5)
    ax.set_axisbelow(True)
    _save(fig, "h2h_forest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy-to", default="")
    a = ap.parse_args()
    print("rebuilding plan-v4 figures from results/")
    # One list, so a figure added here is also a figure copied. units_law was generated and not
    # copied for two days, and a missing \includegraphics halts tectonic and leaves the previous
    # PDF in place -- which then measures as if nothing were wrong.
    figures = (frontier_scaling, opening_effect, order_invariance, onset_collapse, seed_effect,
               context_intervention, selection_frontier, selection_breadth_forest,
               units_law, order_no_collapse, imitation_cost,
               safety_envelope, judge_free, meter_horns, h2h_forest)
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
