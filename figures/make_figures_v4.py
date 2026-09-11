"""Plan v4 figures, rebuilt from results/*.csv only (same contract as figures/make_figures.py:
no GPU, no logs, the analysis scripts own the numbers and this script only draws them).

  frontier_scaling   <- results/anchor_scaling_summary.csv
  opening_effect     <- results/opening_effect_summary*.csv
  order_invariance   <- analysis.regimes.event_bound (closed form) + results/renyi_sweep.csv if present
  imitation_cost     <- results/imitation_cost.csv + results/imitation_lorenz.csv

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
    fig, (axL, ax) = plt.subplots(1, 2, figsize=(9.4, 2.6))

    # ---- panel (a): a budget that scales with the work against one that does not -------------
    ss = sorted(float(r["s_safe"]) for r in onset)          # nats per token, nine pairs
    s_med = ss[len(ss) // 2]
    s_tot = float(odo[0]["S_total_median"])                  # 849 nats, the median protected target
    t_star = s_tot / s_med                                   # its length in tokens
    T = [10 ** (1 + 0.02 * i) for i in range(101)]           # 10 .. 1000 tokens
    axL.fill_between(T, [ss[0] * t for t in T], [ss[-1] * t for t in T],
                     color="0.72", alpha=0.5, lw=0)
    axL.plot(T, [s_med * t for t in T], color="0.25", lw=1.4)
    axL.annotate("$S(x)$, the price of the work\n(nine measured pairs)", (T[22], s_med * T[22]),
                 fontsize=6.3, color="0.2", rotation=31, rotation_mode="anchor",
                 xytext=(0, 6), textcoords="offset points")
    for k, style, xi, note in ((10.0, "-", 12, "never certified,\nthe only useful arm"),
                               (3.0, "--", 50, "the authors' budget"),
                               (0.5, ":", 76, "certified at every length,\nno measurable gain")):
        axL.plot(T, [k * t for t in T], style, color="#c1443c", lw=1.4)
        axL.annotate(f"$K=kT$, $k={k:g}$\n{note}", (T[xi], k * T[xi]), fontsize=6.3,
                     color="#c1443c", rotation=31, rotation_mode="anchor",
                     xytext=(0, 5 if k >= 3 else -17), textcoords="offset points")
    import math as _m
    for n, style in ((64, "--"), (8, "-")):
        axL.axhline(_m.log(n), color="#2f6f9f", lw=1.5, ls=style)
        axL.annotate(f"$K=\\log n$, $n={n}$", (T[0], _m.log(n)), fontsize=6.3,
                     color="#2f6f9f", xytext=(3, 3), textcoords="offset points")
    axL.annotate("certified at every length,\nand useful", (T[62], 1.08),
                 fontsize=6.3, color="#2f6f9f", ha="center")
    axL.axvline(t_star, color="0.5", lw=0.8, ls="-.")
    axL.annotate(f"median protected\ntarget, $S(x)={s_tot:.0f}$", (t_star, 2.4e4), fontsize=6.3,
                 color="0.35", ha="center")
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlim(10, 1000); axL.set_ylim(1.0, 1.2e5)
    axL.set_xlabel("length of the protected work, tokens")
    axL.set_ylabel("certified budget $K$, nats")
    axL.set_title("(a) a budget indexed to the work, and one that is not", fontsize=7.6)

    # Panel (b) is ONE judge. The arms on record are judged by different models and the absolute
    # levels are not comparable across them (caution (e)); plotting a judge-A curve beside a judge-B
    # curve would invite exactly the reading the paper refuses. Both series here are judge B.
    j2 = {float(r["k"]): r for r in csv.DictReader(open(RESULTS / "judge_separation_v6_judge2.csv"))
          if r["decoder"] == "KL"}
    price = {float(r["k"]): r for r in csv.DictReader(open(RESULTS / "utility_price.csv"))}
    sweep = [r for r in csv.DictReader(open(RESULTS / "selection_scaling.csv"))
             if "Phi-3.5" in r["judge"]]
    u_safe = float(sweep[0]["u"])
    w = math.exp(-float(dec[0]["lambda_star_u_max"]))          # P_{p_s}[U = 1]
    t = 2 * (u_safe - w)                                        # from u = w + t/2
    us = [u_safe + i * (1.0 - u_safe) / 200 for i in range(1, 201)]

    def rate(u):
        best = 0.0
        for j in range(1, 4001):
            lam = j * 0.02
            f = lam * u - math.log(w * math.exp(lam) + t * math.exp(lam / 2) + (1 - w - t))
            best = max(best, f)
        return best
    ax.plot([rate(u) for u in us], us, ls=":", color="0.35", lw=1.2,
            label=r"$\Lambda^*_s(u)$, the frontier of Thm. 1")

    ks = sorted(k for k in j2 if k in price)
    x = [float(price[k]["mean_spend_nats"]) for k in ks]
    y = [float(j2[k]["utility"]) for k in ks]
    ax.plot(x, y, "o-", ms=4.5, lw=1.5, color="#c1443c", label="anchored decoding, $k$ swept")
    for k, xv, yv in zip(ks, x, y):
        if k in (min(ks), max(ks)):      # the sweep is dense; two labels bracket it
            ax.annotate(f"$k={k:g}$", (xv, yv), fontsize=6.5,
                        xytext=(6, -3) if k == max(ks) else (6, -8),
                        textcoords="offset points")

    xs = [max(float(r["kl_nats"]), 1e-3) for r in sweep]
    ys = [float(r["u"]) for r in sweep]
    ax.plot(xs, ys, "s-", ms=4.5, lw=1.5, color="#2f6f9f",
            label="selection anchoring, $n$ swept")
    for r, xv, yv in zip(sweep, xs, ys):
        if r["n"] in ("1", "8", "64"):
            ax.annotate(f"$n={r['n']}$", (xv, yv), fontsize=6.5,
                        xytext=(4, -10) if r["n"] == "1" else (-4, 6),
                        textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlim(3e-4, 2e3)
    ax.set_ylim(0.38, 0.68)
    ax.set_xlabel("realised divergence from the anchor, nats per trajectory")
    ax.set_ylabel("judged utility $u$ (judge B)")
    ax.set_title("(b) what a nat buys, both under one judge", fontsize=7.6)
    ax.legend(fontsize=6.6, frameon=False, loc="upper left", handlelength=1.6,
              borderaxespad=0.3)
    fig.tight_layout()
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
    axL.legend(frameon=False, loc="upper left")
    axL.grid(alpha=0.25, lw=0.5)

    axR.plot([0, 1], [0, 1], color="0.55", lw=1.1, ls="--", label="uniform over steps")
    # k = 3 and k = 20 lie on top of each other, which is the point: once the meter stops binding
    # the shape of the spend stops depending on the cap. Dashed so both are visible.
    for k, ls, col in (("0.5", "-", "C0"), ("20", "-", "C2"), ("3", "--", "C1")):
        v = [(float(r["frac_of_steps"]), float(r["frac_of_spend"])) for r in lz if r["k"] == k]
        v.sort()
        axR.plot([0] + [x for x, _ in v], [0] + [y for _, y in v], lw=1.4, ls=ls, color=col,
                 label=f"$k = {k}$")
    axR.plot([0, 0.02, 1], [0, 1, 1], color="C3", lw=1.2, ls=":",
             label="what Proposition 5 needs")
    axR.set_xlabel("fraction of steps, busiest first")
    axR.set_ylabel("share of the spend")
    axR.set_title("(b) where it spends it", fontsize=8)
    axR.legend(frameon=False, loc="lower right")
    axR.grid(alpha=0.25, lw=0.5)
    _save(fig, "imitation_cost")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy-to", default="")
    a = ap.parse_args()
    print("rebuilding plan-v4 figures from results/")
    # One list, so a figure added here is also a figure copied. units_law was generated and not
    # copied for two days, and a missing \includegraphics halts tectonic and leaves the previous
    # PDF in place -- which then measures as if nothing were wrong.
    figures = (frontier_scaling, opening_effect, order_invariance, onset_collapse, seed_effect,
               context_intervention, selection_frontier, units_law, order_no_collapse,
               imitation_cost)
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
