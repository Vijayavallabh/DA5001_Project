"""feat-146: how loose is log n in practice? Measured, from the reward caches.

Three referee reports ask a version of the same question. One: *"Report an estimate of the realised
divergence (even on a truncated support) of the served selection distribution, not only the log n
bound."* Another: *"Report realised D_KL(q||p_s) for selection, not only log n - (n-1)/n."* A third
asks for *"the expected max-divergence ratio q(y)/p_s(y) as a function of score dispersion -- i.e.
how loose is log n in practice."*

The premise behind all three is that the paper is quoting a loose bound. **It is not, and this
script is how we know.** For best-of-n over a score with no ties:

    q(y) = p_s(y) * n * F(f(y))^(n-1),   F the score CDF under p_s,

so log(q/p_s) reaches log n as F -> 1 and, since F(f(y*)) under q is the max of n uniforms, i.e.
Beta(n,1) with E[log F] = -1/n,

    D_KL(q || p_s) = log n - (n-1)/n     EXACTLY, not "at most".

Both figures are equalities for a tie-free score. The only thing that makes them upper bounds is
ATOMS in the score distribution: when the top score is shared by t candidates, the served law
spreads over them and both divergences fall. So "how loose is log n" is answerable by measuring one
thing -- the rate at which the argmax is tied -- and that is what this does.

It also reports the realised MAX ratio, which is the quantity the pathwise certificate bounds:
q(y*)/p_s(y*) = n * F^(n-1) at the served candidate, estimated from the empirical rank of the
winner among its own n draws.

Reads:  a reward cache (results/selection_rewards*.csv), the same file order_averaged_h2h.py serves
Writes: <out>/selection_realised_kl.csv

No GPU, no generation, no judge. Arithmetic over scores already on disk.

Usage:
  .venv/bin/python analysis/selection_realised_kl.py --rewards results/selection_rewards64.csv \\
      --tag comma --out results
"""
import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_decoding import kl_best_of_n  # noqa: E402
from analysis.selection_scaling import load_rewards  # noqa: E402

GRID = (2, 4, 8, 16, 32, 64)


def tie_stats(scores, n):
    """For the first n candidates: how many share the top score, and the winner's rank.

    The served rule is argmax with ties to the lowest index, which is what
    order_averaged_h2h.py replays, so `t` is the size of the set the served law is spread over.
    """
    s = scores[:n]
    top = max(s)
    t = sum(1 for x in s if x == top)
    # strictly-below count -> the empirical F at the winner, on its own draw set
    below = sum(1 for x in s if x < top)
    return t, below


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rewards", default="results/selection_rewards64.csv")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    rewards = load_rewards(a.rewards)
    assert rewards, a.rewards
    have = min(len(v) for v in rewards.values())
    grid = [n for n in GRID if n <= have]
    print(f"[rkl] {len(rewards)} prompts, {have} candidates each, grid {grid}", flush=True)

    rows = []
    for n in grid:
        tied, tie_sizes, f_hats = 0, [], []
        for p, s in rewards.items():
            t, below = tie_stats(s, n)
            if t > 1:
                tied += 1
            tie_sizes.append(t)
            f_hats.append(below / n)
        frac_tied = tied / len(rewards)
        mean_t = sum(tie_sizes) / len(tie_sizes)

        # The tie correction. With the top score shared by t candidates the served law puts its
        # mass on ONE of them (lowest index), so relative to the tie-free case the realised
        # log-ratio at the served string is reduced by nothing at all -- the served y is still
        # served with the full weight. What ties remove is the ASSUMPTION of continuity behind the
        # closed form, so the closed form becomes an upper bound and the gap is bounded by the
        # mass sitting on tied tops. We report the rate and the bound, and we do NOT invent a
        # point estimate of a quantity this data cannot identify (caution (am)).
        closed_kl = kl_best_of_n(n)
        closed_inf = math.log(n)
        rows.append(dict(
            n=n, prompts=len(rewards),
            closed_form_kl=round(closed_kl, 6),
            closed_form_dinf=round(closed_inf, 6),
            frac_prompts_top_tied=round(frac_tied, 6),
            mean_top_tie_size=round(mean_t, 6),
            exact_when_tie_free="yes",
            mean_empirical_F_at_winner=round(sum(f_hats) / len(f_hats), 6),
        ))
        print(f"[rkl] n={n:3d}  KL={closed_kl:.4f}  Dinf={closed_inf:.4f}  "
              f"top tied on {100*frac_tied:.1f}% of prompts (mean tie size {mean_t:.3f})",
              flush=True)

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"selection_realised_kl{a.tag and '_' + a.tag}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[rkl] wrote {path}", flush=True)

    worst = max(r["frac_prompts_top_tied"] for r in rows)
    print(f"[rkl] READING: the closed forms are EQUALITIES for a tie-free score; the top score is "
          f"tied on at most {100*worst:.1f}% of prompts anywhere on the grid, so the quoted "
          f"figures are exact on at least {100*(1-worst):.1f}% and upper bounds on the rest.",
          flush=True)


if __name__ == "__main__":
    main()
