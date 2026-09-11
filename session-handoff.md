# Session handoff

## Current Objective

Plan v5 on branch `iclr-2027`, targeting **ICLR 2027** (abstract Sep 18, paper Sep 25). The
manuscript `~/sub/satml/iclr_2027.tex` is complete and verified: main text **exactly 9 of 9 pages**
(`Ethics` at char 264 of `pdftotext` page 10), 31 pages total, **0 overfull, 0 `??`**, 1373 numeric
literals audited with one expected miss (`64256`, the Comma-7B padded embedding count). **210 tests**
green. **feat-035..081 `done`; nothing in progress.**

The last thread finished this session was the matched-utility line (feat-072..081): what a higher
Renyi order is worth when each order is given the budget that buys exactly the utility the published
one buys, and whether anything a deployer can compute predicts it.

## What landed since the last handoff

- **feat-079 — seven families, the committed endpoint.** Llama-3.2-1B/-3B gave family six; Qwen2.5-7B
  blew up at `--lr 3e-4` (0.10 -> 2.26) and the single retry at `--lr 1e-4` already committed for
  Pleias-3B reached 0.0415 / recall 0.944, so family seven entered. Twelve pairs, seven families.
  Largest of six candidates: **+0.62** (memoriser log p per token at alpha=4, exact p = 0.035),
  below the committed 0.7 -- **the negative is earned**. The family-mean version read +0.90 at five
  families, +0.89 at six, **+0.71 at seven**: it decayed as families were added.
- **feat-080 — a second protected corpus.** 600 excerpts of 50 public-domain books against the
  sixteen CopyBench novels, three anchors spanning the advantage range. Levels move up to 3.59 nats
  per window; **no sign changes, ranking holds**.
- **feat-081 — the price side's workload.** `--ordinary-split factual|creative` against the committed
  `neutral`. The most sensitive axis: 10 of 12 cells beyond their floor, largest 2.12, and **two sign
  flips**, both on cells already inside their own floor. Ranking holds. Summary for the paper: **a
  cell is an order of magnitude and a rank, never a factor.**
- **Precision floors are now per pair** -- all twelve have their own float32 twin, so no pair borrows
  a floor that is not its own. bf16-vs-fp32 spread over 72 cells: -2.34 to +1.54, median |d| 0.30.
  The exploratory crossing count is **18 of 36** (16 uniformly safer, 2 uniformly more dangerous); it
  rose from 7/21 because three pairs stopped borrowing the largest floor measured anywhere.
- **Compute.** `analysis/compute_hours.py` now detects a fine-tune from the `[ft]` lines in a job's
  own log rather than from the job's name, which had undercounted the share by 10 hours:
  **128.4 GPU-hours**, at most **25.5** containing a fine-tune. The manuscript reads 128 / "at most 26".
- **Artifact.** The builder was shipping and an earlier session had committed `data/gutenberg/`
  (44 MB), against `README_artifact.md`'s own statement. Excluded; 23 MB -> 11 MB, 559 files. Git
  history was not rewritten.

## Files Changed

`analysis/compute_hours.py` (`has_finetune`), `analysis/order_{law,predictors,crossings}.py` (the
`_work` guard beside `_seed`/`_gut_`), `tests/test_compute_hours.py`, `tests/test_order_seed.py`,
`scripts/build_artifact.sh`, `feature_list.json` (feat-079/080/081), `progress.md`,
`README_artifact.md`, `AGENTS.md`, `results/{compute_hours,compute_hours_summary,order_crossings,
order_predictors*}.csv`, `artifact/`. In `~/sub/satml`: `iclr_2027.tex` (LLM-Usage compute figure).

## Recommended Next Step

Everything the plan opened is executed and every number in the paper is sourced. The two things worth
doing next, in order:

1. **A full adversarial read-through of the compiled PDF**, abstract to Appendix H, against the CSVs
   -- the last one (2026-09-10) found twelve claims that did not survive checking, and the appendix
   has been substantially rewritten since. Check especially that every quoted cell in
   Appendix~\ref{app:matched} carries its workload caveat, since feat-081's sign flips made that a
   requirement rather than a nicety.
2. **Whatever the abstract deadline needs.** Sep 18 is abstract registration, which is `feat-016` --
   **human-only, never to be started by the agent.**

## Standing constraints worth re-reading before touching anything

`AGENTS.md` in full, and in particular: never push to a remote; `feat-016` is human-only; do not
modify `~/sub/neurips_2026.tex`, `output.zip`, or the committed prompt sets under `data/` (the one
writable path there is `data/gutenberg/`); nothing in `~/sub/satml/` or the artifact may identify the
authors, the sole exception being third-person `\cite{vijayavallabh2026audit}` as "an earlier audit";
`HF_TOKEN` returns 401, so run local jobs with `HF_HUB_OFFLINE=1`; ask before any **new** gated
download; never GPU 3; always `CUDA_DEVICE_ORDER=PCI_BUS_ID`; the manuscript tree sits inside a stray
home git repo -- always run git with an explicit path into `DA5001_Project`; after any manuscript
edit recompile and check exit status, 0 `??`, 0 overfull, <= 9 pages of main text; a budget violation
is per-trajectory; `master` holds the verified SaTML paper at `dd7e801` as the fallback.
