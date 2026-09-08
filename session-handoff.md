# Session handoff

## Current Objective
Plan v5, branch `iclr-2027`, target ICLR 2027 (abstract Sep 18, paper Sep 25). The onset law is
now measured on **four** (anchor, risky) pairs and the manuscript reports them.

## What just happened
- Pair 4 (Pleias-1.2B) swept and registered: onset 2.819, ratio 0.878, 0% bootstrap no-crossing.
- Pair 3 re-measured at 458 passages. Its first grid stopped at k=3.6, where the curve is still
  flat at the threshold, so a third of bootstrap resamples never crossed; two extra points (3.8,
  4.2) fixed that (0.1%) and moved the onset 3.18 -> 3.271, ratio 0.895 -> 0.920.
- Held-out scoring (`analysis/score_predictions.py`) **reverses the paper's earlier claim**: the
  parameter-free rule P1 (median s_s - s_r) has mean absolute error 0.034 nats on the two pairs
  predicted before measurement, against 0.073 for the fitted constant and 0.224 for the calibrated
  q25, which is rejected. P1 still fails the directional claim (Spearman +0.20).
- Manuscript rewritten across abstract, intro, Section 4, conclusion and two appendices;
  18 pages, main text ends on page 9, 0 overfull, 0 `??`, 19 quoted numbers audited against CSVs.
- `recipes/finetune_memorizing.py` had been silently truncating at 448 tokens, which killed three
  KL3M runs (100% of KL3M texts exceed it; no other family's do). Fixed, and **KL3M-003-1.7b is
  now admissible at 0.759 sampled recall**.

## Recommended next step
Run **pair 5, KL3M-003-1.7b**. It is the highest-value pair left: 1.103 nats/char against
0.685-0.878 for every anchor used so far, from a different architecture (GPT-NeoX) and tokenizer,
widening the tokenizer-free dynamic range from 1.33x to about 1.67x. The order matters:

    scripts/add_pair.sh "KL3M-1.7B + mem. KL3M-1.7B" alea-institute/kl3m-003-1.7b \
      output/phase5/mem_kl3m-003-1_7b output/phase5/fine_kl3m17b/composition_summary.csv 4
    # -> computes the budget path and the prediction, and says the sweep is pending.
    # COMMIT THE PREDICTION, then run composition_attack.py on a grid bracketing it, then re-run.

Give the sweep a grid whose top end is well past the predicted onset -- pair 3's truncated grid is
the mistake to avoid. kl3m-002-520m is still fine-tuning in `output/phase5/ft_kl3m_v3.log`.

## Open, not started
- The judged grid at k in {1.5, 2, 2.5} was killed by system memory pressure and has not been rerun.
- A second judge, and the alpha=4 judged arm, remain unmeasured (plan section D).
- Plan v5 says to retitle around the no-free-lunch; the title is unchanged because AGENTS.md
  requires asking first.
