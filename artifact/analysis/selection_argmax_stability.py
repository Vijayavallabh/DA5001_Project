"""How reproducible is the completion that best-of-n actually SERVES?

Zero GPU: every number here comes from reward caches already on disk.

This arm had NO committed bands and is exploratory, which is why its write-up is
`results/selection_argmax_stability_note.md` and never an `onset_prediction_*.md`. It exists because
feat-136's instrument gate failed and the control that diagnosed it turned out to measure something
the paper does not have.

The question. Selection anchoring draws n completions from the anchor, scores them, and serves the
argmax. Proposition 1 bounds q(y) <= n p_s(y) for ANY score and ANY tie rule, so the CERTIFICATE does
not care which candidate wins a near-tie. But a deployer does: they cannot reproduce a served answer
they cannot re-derive. Nothing in this paper has ever asked whether the argmax is reproducible.

Three things are measured per comparison and per n, because the disagreement RATE alone would
overstate the problem:

  agree_frac      the fraction of prompts whose served candidate is the same under both caches, and
                  disagree_pct, its complement as a percentage, stored so that the percentage the
                  manuscript quotes rounds from this CSV once and not at the point of writing.
  margin          the mean gap between the winning reward and the runner-up, under the reference
                  cache. This is the MECHANISM: the argmax is fragile exactly where the top two
                  candidates are close, and drawing more candidates makes close calls more likely.
  reward_loss     the mean reward the second cache's pick gives up, measured under the FIRST cache's
                  own numbers. This is the quantity that matters: if two caches disagree about which
                  candidate wins but agree it is worth the same, the served text changed and the
                  served QUALITY did not. Reported as both a mean over all cells and a mean over the
                  disagreeing cells only, since the first is diluted by every cell that agrees.

The nesting rule is selection_scaling.py's own: arm n serves the argmax over the FIRST n candidates
in seed order, so the arms nest and a candidate that wins at n wins at every larger n unless a later
one beats it.

Usage:
  .venv/bin/python analysis/selection_argmax_stability.py --out results
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.score_host_transfer_gate import read_cache  # noqa: E402
from analysis.selection_scaling import n_grid  # noqa: E402

MAX_N = 64

# (label, what differs, reference cache, other cache). The reference is the cache whose reward
# numbers define "loss": we ask what the OTHER cache's pick is worth on the reference's own scale.
COMPARISONS = (
    ("host", "different GPU architecture, both bf16, batch 8",
     "results/selection_rewards64_comma7b.csv",
     "results/hostb/selection_rewards64_comma7b_hostb.csv"),
    ("batch", "same host, bf16, batch 8 against batch 16",
     "results/hostb/selection_rewards64_comma7b_hostb.csv",
     "results/hostb/selection_rewards64_comma7b_hostb_b16.csv"),
    ("precision", "same host, batch 8, bf16 against fp32",
     "results/hostb/selection_rewards64_comma7b_hostb_fp32.csv",
     "results/hostb/selection_rewards64_comma7b_hostb.csv"),
    ("host_fp32", "different GPU architecture, both fp32, batch 8",
     "results/selection_rewards64_comma7b_local_fp32.csv",
     "results/hostb/selection_rewards64_comma7b_hostb_fp32.csv"),
    ("batch_fp32", "same host, fp32, batch 8 against batch 16",
     "results/hostb/selection_rewards64_comma7b_hostb_fp32.csv",
     "results/hostb/selection_rewards64_comma7b_hostb_fp32_b16.csv"),
)


def pick(rewards, p, n):
    """selection_scaling.py's nesting rule: argmax over the FIRST n candidates in seed order."""
    return max(range(n), key=lambda j: rewards[(p, j)])


def margin(rewards, p, n):
    """Winner minus runner-up under this cache. Undefined at n=1, where there is no runner-up."""
    if n < 2:
        return None
    vals = sorted((rewards[(p, j)] for j in range(n)), reverse=True)
    return vals[0] - vals[1]


def analyse(ref, oth, max_n=MAX_N):
    pids = sorted({p for p, _ in ref} & {p for p, _ in oth})
    full = [p for p in pids if all((p, j) in ref and (p, j) in oth for j in range(max_n))]
    assert full, "the two caches share no prompt with a complete rank range"
    out = []
    for n in [g for g in n_grid(max_n) if g <= max_n]:
        agree = losses = 0
        dis_losses = []
        margins = []
        for p in full:
            a, b = pick(ref, p, n), pick(oth, p, n)
            m = margin(ref, p, n)
            if m is not None:
                margins.append(m)
            if a == b:
                agree += 1
            else:
                # what the other cache's pick gives up, on the REFERENCE cache's own scale
                dis_losses.append(ref[(p, a)] - ref[(p, b)])
        losses = sum(dis_losses)
        out.append(dict(
            n=n, n_prompts=len(full),
            agree_frac=round(agree / len(full), 5),
            # stored so the manuscript quotes a number that is IN a CSV rather than one derived at
            # the point of writing (caution (j): a paper number must round from the CSV, once)
            disagree_pct=round(100 * (len(full) - agree) / len(full), 5),
            n_disagree=len(full) - agree,
            mean_margin=(round(sum(margins) / len(margins), 5) if margins else ""),
            reward_loss_all=round(losses / len(full), 5),
            reward_loss_on_disagreements=(round(losses / len(dis_losses), 5)
                                          if dis_losses else 0.0),
        ))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    rows, skipped = [], []
    for label, what, ref_p, oth_p in COMPARISONS:
        if not (os.path.exists(ref_p) and os.path.exists(oth_p)):
            skipped.append((label, ref_p if not os.path.exists(ref_p) else oth_p))
            continue
        ref, oth = read_cache(ref_p), read_cache(oth_p)
        res = analyse(ref, oth)
        print(f"\n=== {label}: {what}")
        print(f"    {'n':>4} {'served the same':>16} {'disagree':>9} {'mean margin':>12} "
              f"{'loss (all)':>11} {'loss | disagree':>16}")
        for r in res:
            print(f"    {r['n']:>4} {r['agree_frac']:>16.5f} {r['n_disagree']:>9} "
                  f"{str(r['mean_margin']):>12} {r['reward_loss_all']:>11.5f} "
                  f"{r['reward_loss_on_disagreements']:>16.5f}")
        for r in res:
            rows.append(dict(comparison=label, differs=what, **r))
    for label, missing in skipped:
        print(f"\n=== {label}: SKIPPED, {missing} is not on disk yet")

    assert rows, "no comparison could be run"
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "selection_argmax_stability.csv")
    cols = ["comparison", "differs", "n", "n_prompts", "agree_frac", "disagree_pct", "n_disagree",
            "mean_margin", "reward_loss_all", "reward_loss_on_disagreements"]
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"\nwrote {p} ({len(rows)} rows, {len(skipped)} comparison(s) still pending)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
