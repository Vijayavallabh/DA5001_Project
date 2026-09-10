# Session handoff

## Current objective
Plan v5, branch `iclr-2027`, targeting ICLR 2027 (abstract Sep 18, paper Sep 25). The manuscript is
complete and verified; the remaining work is whatever the plan opens next.

## What this session added
A correction and five features, all pre-registered before their runs and scored against their
committed bands in `results/onset_prediction_orders_matched.md`.

1. **A split bug that was already in the paper.** A pre-registered sanity bracket fired on its first
   run: the memoriser assigned the "protected" passage `e^14063` *less* mass than the clean anchor.
   Every phase-5 memoriser trains on `attack_train` + `val` with `test` held out, and the two splits
   are disjoint in novel, so `marginal_price.py` and `order_price.py` were both scoring a novel the
   model has never seen. Both now default to `attack_train`; feat-070's table was re-run and gained
   the memorised trajectory as a third target type rather than losing one.
2. **feat-073** — the order comparison at matched *utility*, the axis Appendix D concedes it lacks.
3. **feat-074/075/076** — what predicts it? Nothing measured does, and the negative is now earned at
   a power that can carry it: nine pairs, exact permutation `p` over all `9!` orderings, largest
   `|rho| = 0.53`. The seven-pair leader fell from `+0.75` to `+0.48` when two pairs were added.
4. **feat-077 (exploratory, labelled)** — the orders' fidelity-leakage curves: 19 of 27 cells
   uniformly safer, 7 cross, 1 uniformly more dangerous.
5. **feat-078** — the advantage belongs to the pair, not the evaluation's seed.

## State
- **204 tests** (`./init.sh` green), 73 features, none in progress, feat-035..078 `done`.
- Manuscript: main text **exactly 9 of 9 pages** (Ethics at char 264 of page 10, i.e. the body ends
  at the foot of page 9), 31 total, 0 overfull, 0 `??`, 1346 numeric literals audited with 1
  expected miss (`64256`). Compute 119 GPU-hours.
- Artifact rebuilt: 573 files, `artifact.zip` 27M.

## Recommended next step
The order thread is finished and its limitations are stated in the paper rather than left implicit.
Three candidates, in order of expected value:

1. **A sixth family.** The conservative family-clustered test is stuck at `n = 5` and its one
   surviving signal (the memoriser's own confidence, `+0.90`, `p = 0.083`) cannot be resolved
   without a family the cached model set does not contain. Downloading and memorising one new
   permissively licensed family would decide it. Ask first: a new download plus a fine-tune.
2. **The one-corpus limitation.** Everything runs on sixteen English genre novels and Limitations
   says so. A memoriser on public-domain prose would give a second corpus for both the onset and the
   frontier.
3. **Nothing.** The paper is verified end to end and both deadlines have slack. Stopping is a
   legitimate choice and the fallback at `dd7e801` on `master` is intact.

## Standing constraints
Never push to a remote; `feat-016` is human-only. Never commit inside `~/sub/satml` (stray home git
repo) -- always `git -C .../DA5001_Project`. GPU 3 is a 4 GB T400: never use it, and always set
`CUDA_DEVICE_ORDER=PCI_BUS_ID`. `HF_HUB_OFFLINE=1`; `meta-llama/*` stays gated. `master` holds the
verified SaTML paper at `dd7e801` as the fallback. Abstract edits must be length-neutral: adding
four lines there once cost thirteen lines of reflow and broke the 9-page limit.
