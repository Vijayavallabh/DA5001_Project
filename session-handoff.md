# Session handoff

## Current Objective

Plan v5, ICLR 2027 (abstract Sep 18, paper Sep 25). Branch `iclr-2027`; `master` holds the verified
SaTML paper at `dd7e801` and must not be deleted. The manuscript is `~/sub/satml/iclr_2027.tex`
(absolute `/mnt/md0/IITM/BackUp/Home/vijayavallabh/sub/satml`), **not in this repo** -- never run
git after a `cd` into that tree.

State: main text **exactly 9 pages** (page 10 opens with the Ethics heading), 0 errors, 0 overfull,
0 `??`, 1120 numeric literals audited with 1 expected miss. **183 tests.** feat-035..071 `done`;
feat-072 in progress.

## What landed this session

- **feat-070 -- the paper's open problem, answered in the negative.** Section 2 previously stated
  the open problem as closing the three-to-four orders of magnitude between the decoder's spend and
  Theorem 1's floor, and named the route: stop decoding greedily against the bucket. That route is
  now measured and it is worth percent. On the geodesic the charge and the fidelity
  `G(theta) = -D(p_r||p_theta) + const` are both closed forms in `psi`, so the optimal allocation of
  a fixed total across a trajectory is computable offline with no decoding run. Over 2 pairs x 2
  target types x 4 budgets, reallocating the bucket's own spend buys **1.008 to 1.125**.
  The stronger form needs no allocation argument at all: at `k=1` the bucket already captures
  **77-85%** of the fidelity that serving the risky model outright would buy, so an *unlimited*
  budget is worth 1.2-1.3x. Three orders of magnitude are not there to recover. Section 2 now says
  this instead of speculating, and `sections/appendix_proofs.tex` carries the design.
- **feat-071 -- the prescription made executable.** The conclusion says publish `k/s(x)`; Section 3
  concedes a deployer has not seen `x`. The anchor's rate on 50 public-domain Gutenberg texts
  predicts its protected rate to **5.6%** leave-one-anchor-out over ten anchors spanning 1.86x,
  **4.13x better than a constant**, while `c_use` does *worse* than a constant. In
  `sections/appendix_robustness.tex`, with a clause in the conclusion.
- Disk: `/` had filled to 100%, which fails every Bash call before the command runs. Package caches
  cleared with the user's approval (62 GB free). **`CLAUDE_CODE_TMPDIR`/`TMPDIR` in
  `.claude/settings.local.json` are inert** -- the harness owns them; `TECTONIC_CACHE_DIR` works.

## Running when this file was written

`analysis/order_price.py` on GPU 4, `results/order_price_kl3m_k3.csv` -- feat-072, pre-registered in
`results/onset_prediction_orders_matched.md`. It asks the question feat-070 raises: if the overhead
is the price of the *target* rather than of the schedule, does changing the target help? For each
Renyi order at one published `k` it reads off fidelity bought on ordinary generations (the price)
and on protected passages (the leakage), deterministically, which is the matched-utility axis
`sections/appendix_robustness.tex` currently concedes it lacks and that Section 5 shows no judge at
n = 150 can supply.

## Recommended next step

1. Score `order_price_kl3m_k3.csv` against the committed bands. **Check the embarrassment condition
   first**: if the four arms' ordering by this instrument disagrees with their ordering by
   intervention rate in Table 1 (94.0%, 91.4%, 8.7%, 0.07% risky-unchanged), the instrument is
   withdrawn, not reinterpreted.
2. Replicate on Pleias-1.2B before anything reaches the manuscript.
3. Rebuild the artifact and re-run `analysis/compute_hours.py`; both are stale by two features.

## Cautions that cost time this session

- The probe allocates thousands of `[T, V]` tensors per passage; without an explicit `del` plus
  `torch.cuda.empty_cache()` per passage the caching allocator reached **80 GB on a 0.5B model** and
  the run crawled. The fix is in `analysis/marginal_price.py`.
- Moving `TECTONIC_CACHE_DIR` makes the first compile re-download the whole LaTeX package set; it
  takes minutes and looks like a hang.
- Page-budget edits reflow rather than shed lines. Micro-trimming a sentence three times in a row
  moved nothing; one structural cut of a whole sentence moved two lines.
