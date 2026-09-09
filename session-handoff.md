# Session handoff

## Current Objective
Plan v5, branch `iclr-2027`, target ICLR 2027 (abstract Sep 18, paper Sep 25). The onset is now
measured on **seven** (anchor, risky) pairs plus one natural memoriser, and the manuscript reports
them all.

## Where the science landed

The onset ratio takes **two values**, and what selects between them is now pinned to one variable:

    pair                            chars/tok  vocab    s_r/s_s   ratio    95% CI (ratio)
    TinyComma-1.8B + mem. Llama-8B       4.20  128,256    0.060    0.887   [0.84, 1.07]
    Comma-7B + mem. Comma-7B             3.63   64,000    0.075    0.892   [0.84, 1.22]
    Pleias-350M + mem. Pleias-350M       4.05   65,536    0.092    0.920   [0.86, 1.08]   n=458
    Pleias-1.2B + mem. Pleias-1.2B       4.05   65,536    0.114    0.878   [0.79, 0.96]
    Phi-3.5-mini + mem. Phi-3.5-mini     3.81   32,011    0.003    0.926   [0.80, 1.09]
    KL3M-520M + mem. KL3M-520M           1.96   32,768    0.089    1.053   [1.02, 1.24]
    KL3M-1.7B + mem. KL3M-1.7B           1.96   32,768    0.353    1.166   [1.14, 1.41]

- **Not memoriser strength** (feat-059, and again at its limit in feat-062): Phi memorises to
  s_r/s_s = 0.003, the most thorough of the seven, and still lands with the coarse family.
- **Not target length** (feat-060): truncating KL3M-520M to 276 decode steps with the tokenizer
  fixed leaves the ratio at 1.091, CI [1.04, 1.24].
- **Not vocabulary size** (feat-062): Phi carries fewer types than KL3M (32,011 vs 32,768) and
  still cuts at 3.8 characters per token; its ratio is 0.926.
- **It is tokenizer granularity.** Which property of it is **open**. `results/tokenizer_rates.csv`
  says no cached anchor sits between the two groups (2.4-3.4 chars/token is empty across 14
  tokenizers), so the next test needs a tokenizer trained for it, not one chosen off the shelf.

All three pre-registered onset rules are refuted on both KL3M pairs. Held-out mean absolute error
over five predicted pairs: P1 0.352, constant 0.251, q25 0.481 (in CI 3/5, 3/5, 2/5). P1's
directional claim still has the wrong sign (Spearman -0.18). The paper claims a **unit, not a
law**: s(x) locates the onset within about a fifth across a 1.87x spread.

## State
- 57 features, feat-047 and feat-053..062 done; **148 tests**; `./init.sh` passes.
- Manuscript: 19 pages, main text exactly 9 (Ethics starts at the top of page 10 with nothing
  above it), 0 overfull, 0 `??`. All seven table rows re-audited against `results/onset_table.csv`.
- Every pair was pre-registered before its sweep: `results/onset_prediction_pair{4,5,6}.md`,
  `onset_prediction_trunc276.md` and the Phi bands at `07f8717` (+ addendum `a2df3f1`).

## Recommended next step
**The judged arms** (plan section D), which bear on Theorem 1's measured side: k in {1.5, 2, 2.5}
was killed by memory pressure and never rerun, and the alpha=4 arm and a second judge are still
unmeasured. The GPUs are currently shared with another user's job (55-68 GB used on each A100), so
check `nvidia-smi` before taking a card.

A third granularity level is **not available off the shelf** -- see `results/tokenizer_rates.csv` --
so treat it as a training task, not a download, if it is attempted at all.

## Open, logged, not fixed
- **Checkpoint-trust posture is inconsistent.** `a_patch/factory.py` refuses pickled checkpoints
  (`use_safetensors=True` at three load sites); `analysis/budget_path.py`, `regimes.py`,
  `onset_theory.py` and `surprisal_cdf.py` do not, and `budget_path.py` had already executed a
  `.bin` that the decoder then declined. Tightening the scripts or relaxing the decoder is a
  security decision for the user, not a side effect of a sweep.
- Plan v5's ladder (`results/onset_ladder.md`) is stale: it predates pairs 4-7.
- `output/phase5/anchor_phi35mini` is a config-patched copy of `microsoft/Phi-3.5-mini-instruct`
  (weights symlinked, `auto_map` stripped so it loads offline), logits verified identical to
  0.00e+00. `results/onset_pairs.tsv` names the hub id, not that path, so a clean checkout can
  re-run `onset_units.py`.

## Three process notes worth keeping
- **A shared scorer must not hardcode its output filename.** Reusing `score_truncation.py` for the
  Phi pair silently overwrote `results/truncation_score.csv`, feat-060's committed evidence. It
  now takes `--out-name` and row labels.
- **Edit scripts must write after each successful replacement.** A script that made three
  replacements in memory and hit a failed assertion before its single `write_text` silently
  dropped a table row and a section heading while the compile still succeeded.
- **`pgrep -f <pattern>` matches the shell running it.** Use `pgrep -f "python.*<name>"`.
