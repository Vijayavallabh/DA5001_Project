# Session handoff

## Current Objective
Plan v5, branch `iclr-2027`, target ICLR 2027 (abstract Sep 18, paper Sep 25). The onset is now
measured on **six** (anchor, risky) pairs plus one natural memoriser, and the manuscript reports
them. The paper was retitled to match what survived.

## Where the science landed

The onset ratio takes **two values**, and both causes are now named:

    pair                            chars/tok  tgt tok  s_r/s_s   ratio    95% CI (ratio)
    TinyComma-1.8B + mem. Llama-8B       4.20      276    0.060    0.887   [0.84, 1.07]
    Comma-7B + mem. Comma-7B             3.63      276    0.075    0.892   [0.84, 1.22]
    Pleias-350M + mem. Pleias-350M       4.05      276    0.092    0.920   [0.86, 1.08]   n=458
    Pleias-1.2B + mem. Pleias-1.2B       4.05      276    0.114    0.878   [0.79, 0.96]
    KL3M-520M + mem. KL3M-520M           1.96      580    0.089    1.053   [1.02, 1.24]
    KL3M-1.7B + mem. KL3M-1.7B           1.96      580    0.353    1.166   [1.14, 1.41]

- **Not memoriser strength** (feat-059): pair 6 memorises thoroughly, s_r/s_s 0.089 sits between
  two pairs at 0.892 and 0.920, and it still lands at 1.053.
- **Not target length** (feat-060): truncating KL3M-520M to 276 decode steps with the tokenizer
  fixed leaves the ratio at 1.091, CI [1.04, 1.24].
- **It is the tokenizer.** Which property of it is **open**, and the paper offers no guess. The
  span-length explanation it used to offer is refuted by the same experiment.

All three pre-registered onset rules are refuted on both KL3M pairs. Held-out mean absolute error
over four predicted pairs: P1 0.390, constant 0.288, q25 0.584. P1's directional claim inverts
(Spearman -0.37). The paper claims a **unit, not a law**: s(x) locates the onset within about a
fifth across a 1.87x spread.

## State
- 55 features, feat-047 and feat-053..060 done; **144 tests**; `./init.sh` passes.
- Manuscript: 19 pages, main text exactly 9, 0 overfull, 0 `??`. Retitled
  *A Divergence Budget Is Uninformative Without the Work It Protects / Vacuity, Extraction and
  Utility in Metered Decoding*. 22 quoted numbers audited against `results/*.csv`.
- Every pair was pre-registered before its sweep: `results/onset_prediction_pair{4,5,6}.md` and
  `onset_prediction_trunc276.md`, each committed before the run it predicts.

## Recommended next step
Two candidates, in order:

1. **A third tokenizer family.** Every pair is either ~4 or ~2 characters per token, so "the
   tokenizer" is a two-valued variable and nothing identifies *which* property. A ~3 chars/token
   anchor would say whether the ratio moves continuously with granularity or jumps. Cached
   candidates are in `hf_cache/`; pre-register before sweeping.
2. **The judged arms.** k in {1.5, 2, 2.5} was killed by memory pressure and never rerun; the
   alpha=4 arm and a second judge are still unmeasured (plan section D). These bear on
   Theorem 1's measured side, which is the other half of the paper.

## Open, logged, not fixed
- **Checkpoint-trust posture is inconsistent.** `a_patch/factory.py` refuses pickled checkpoints
  (`use_safetensors=True` at three load sites); `analysis/budget_path.py`, `regimes.py`,
  `onset_theory.py` and `surprisal_cdf.py` do not, and `budget_path.py` had already executed a
  `.bin` that the decoder then declined. Tightening the scripts or relaxing the decoder is a
  security decision for the user, not a side effect of a sweep. `output/phase5/anchor_kl3m-002-520m`
  is a re-serialised copy of `alea-institute/kl3m-002-520m`, verified bit-identical (147 tensors).
- Plan v5's ladder (`results/onset_ladder.md`) is stale: it predates pairs 4-6.

## Two process notes worth keeping
- **Edit scripts must write after each successful replacement.** A script that made three
  replacements in memory and hit a failed assertion before its single `write_text` silently
  dropped a table row and a section heading while the compile still succeeded. The numeric audit
  caught it. Anchor-and-index splicing beats multi-line literals, which line wrapping breaks.
- **`pgrep -f <pattern>` matches the shell running it.** That produced a false "job died" report
  and two false "still running" reports. Use `pgrep -f "python.*<name>"`.
