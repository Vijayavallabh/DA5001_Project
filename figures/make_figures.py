import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
})


def load_heldout(path):
    rows = []
    for line in open(path):
        rows.append(json.loads(line))
    return rows


def fig_heldout_scatter():
    k3 = load_heldout(REPO / "output/h2_outputs/heldout_validation.jsonl")
    k5 = load_heldout(REPO / "output/h2_k5_outputs/heldout_validation.jsonl")

    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.4))

    def points(rows, K):
        xs, ys, ms = [], [], []
        for r in rows:
            B = r.get("effective_budget_min", 0)
            U = r.get("U_EBB", 0)
            N = r.get("N", 0)
            if B <= 0:
                continue
            xs.append(N)
            ys.append(U / B)
            ms.append(r.get("mean_spend", 0))
        return np.array(xs), np.array(ys), np.array(ms)

    x3, y3, m3 = points(k3, 600)
    x5, y5, m5 = points(k5, 1000)

    ax.scatter(x3 + 0.15, y3, s=55, c="#c0392b", marker="o", label=r"$k=3$ heldout",
               edgecolors="black", linewidths=0.5, zorder=3)
    ax.scatter(x5 - 0.15, y5, s=55, c="#2980b9", marker="s", label=r"$k=5$ heldout",
               edgecolors="black", linewidths=0.5, zorder=3)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, zorder=1)
    ax.text(21, 1.02, r"$\rho=1$ (proxy violation)", fontsize=7.5, ha="right", va="bottom")

    ax.set_xlabel(r"Realized trajectories $N$ per prompt (after adaptive allocation)")
    ax.set_ylabel(r"Proxy spend ratio $\rho = U_{\mathrm{EBB}} / B_{\mathrm{eff}}$")
    ax.set_xticks([4, 10, 20])
    ax.set_xlim(2, 22)
    ax.set_ylim(0, 1.65)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.92)

    fig.tight_layout()
    fig.savefig(OUT / "heldout_scatter.pdf", bbox_inches="tight")
    fig.savefig(OUT / "heldout_scatter.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote heldout_scatter.pdf")


def fig_bernstein_term():
    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.4))

    delta = 0.0033
    log_term = math.log(2.0 / delta)
    Ns = np.arange(2, 41)

    R_eff_values = [60, 90, 120, 160, 200]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(R_eff_values)))

    for R_eff, c in zip(R_eff_values, colors):
        widths = 3.0 * R_eff * log_term / Ns
        ax.plot(Ns, widths, color=c, linewidth=1.8, label=fr"$R_{{\mathrm{{eff}}}}={R_eff}$")

    # Reference lines
    ax.axhline(600, color="#c0392b", linestyle="--", linewidth=0.9, zorder=1)
    ax.text(2.2, 615, r"$K=600\,(k=3)$", fontsize=7.5, ha="left", va="bottom", color="#c0392b")
    ax.axhline(1000, color="#2980b9", linestyle="--", linewidth=0.9, zorder=1)
    ax.text(2.2, 1015, r"$K=1000\,(k=5)$", fontsize=7.5, ha="left", va="bottom", color="#2980b9")

    # Highlight the N=4 column
    ax.axvline(4, color="grey", alpha=0.20, linewidth=10, zorder=0)
    ax.text(4, 1480, r"$N=4$ floor", fontsize=7.5, ha="center", va="top", color="#444")

    ax.set_xlabel(r"Trajectories $N$ per prompt")
    ax.set_ylabel(r"$\dfrac{3\,R_{\mathrm{eff}}\,\log(2/\delta)}{N}$  (Bernstein constant term)")
    ax.set_xlim(2, 40)
    ax.set_ylim(0, 1500)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.92, title=r"$\delta=0.0033$", title_fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "bernstein_constant.pdf", bbox_inches="tight")
    fig.savefig(OUT / "bernstein_constant.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote bernstein_constant.pdf")


def fig_spend_distributions():
    k3 = load_heldout(REPO / "output/h2_outputs/heldout_validation.jsonl")

    bad = sorted(
        [r for r in k3 if r.get("effective_budget_min", 0) > 0 and r.get("rho", 0) > 1.0],
        key=lambda r: -r.get("rho", 0),
    )
    ok = sorted(
        [r for r in k3 if r.get("effective_budget_min", 0) > 0
         and 0 < r.get("rho", 0) <= 1.0 and r.get("N", 0) >= 10],
        key=lambda r: -r.get("rho", 0),
    )

    if not bad or not ok:
        print("skip spend distributions: insufficient data")
        return

    fig, ax = plt.subplots(1, 1, figsize=(5.6, 3.4))
    all_spends = []
    for r in bad + ok[:3]:
        all_spends += r["spends"]
    bins = np.linspace(0, max(all_spends) + 20, 18)

    ok_pool = []
    for r in ok[:3]:
        ok_pool += r["spends"]
    bad_pool = []
    for r in bad:
        bad_pool += r["spends"]

    ax.hist(
        ok_pool, bins=bins, alpha=0.55, color="#2980b9",
        edgecolor="black", linewidth=0.5,
        label=fr"$N=20$, $\rho \in [{min(r['rho'] for r in ok[:3]):.2f},{max(r['rho'] for r in ok[:3]):.2f}]$ (3 prompts)",
    )
    ax.hist(
        bad_pool, bins=bins, alpha=0.55, color="#c0392b",
        edgecolor="black", linewidth=0.5,
        label=fr"$N=4$, $\rho \in [{min(r['rho'] for r in bad):.2f},{max(r['rho'] for r in bad):.2f}]$ ({len(bad)} prompts)",
    )

    ax.axvline(600, color="black", linestyle="--", linewidth=0.9)
    ymax = ax.get_ylim()[1]
    ax.text(595, ymax * 0.95, r"$K=600$", ha="right", va="top", fontsize=8)
    ax.axvline(np.mean(bad_pool), color="#c0392b", linestyle=":", linewidth=1.2)
    ax.text(
        np.mean(bad_pool) + 5, ymax * 0.88,
        fr"mean$=${np.mean(bad_pool):.0f}", color="#c0392b", fontsize=8, va="top",
    )
    ax.set_xlabel(r"Per-trajectory KL spend $Z_i$")
    ax.set_ylabel("count")
    ax.set_xlim(0, 620)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT / "spend_distributions.pdf", bbox_inches="tight")
    fig.savefig(OUT / "spend_distributions.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote spend_distributions.pdf")


if __name__ == "__main__":
    fig_heldout_scatter()
    fig_bernstein_term()
    fig_spend_distributions()
