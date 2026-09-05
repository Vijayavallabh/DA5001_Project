# DA5001 Project — SaTML 2027 audit of Anchored Decoding

Code for arXiv 2605.28001, now being reworked into a SaTML 2027 submission: **"What does a KL budget certify? An adversarial audit of inference-time near-access-freeness."** The mechanism under audit is He et al.'s Anchored Decoding (`a_patch/`, KL budget K = k·T_max). The full rationale and experiment design is `~/sub/satml/IMPROVEMENT_PLAN.md` (absolute: `/mnt/md0/IITM/BackUp/Home/vijayavallabh/sub/satml/IMPROVEMENT_PLAN.md`). The manuscript is `~/sub/satml/satml_2027.tex`.

**Deadlines (AoE):** abstract Sep 22 2026 · paper Sep 29 2026 · artifacts Oct 2 2026. Human-only steps (registration, submission, Zenodo upload) are `feat-016`; never attempt them.

**Status (2026-09-05):** GOAL COMPLETE. feat-001..009 and 013..015 done; feat-010/011/012 optional and not started; the PDF (11 pages) and `artifact.zip` are named in `session-handoff.md`.

## Startup Workflow

Before writing code:

1. `pwd` must be this repo root.
2. Read this file completely, then `~/sub/satml/IMPROVEMENT_PLAN.md` Sections 0–4 (it is the spec; do not re-derive it).
3. Run `./init.sh`. If it fails, repair that first before adding scope.
4. Read `feature_list.json` and `progress.md`; pick the lowest-numbered feature whose dependencies are `done` and whose status is `not-started` or `in-progress`.
5. `git log --oneline -5`.

## Working Rules

- **One feature at a time.** Exactly one feature `in-progress` at any moment.
- **Stay in scope.** Touch only files the feature needs. Log unrelated bugs in `progress.md` under Blockers/Risks; do not fix them.
- **Verification required.** Nothing is done without running its evidence command and pasting the output into `feature_list.json`.
- **Baselines are mandatory.** Every experiment that reports a copying or spend metric at some k also reports k = −1 (risky model only) and k = 0 (safe model only) on the same prompts and seeds.
- **Per-trajectory, not per-mean.** A "budget violation" is only a violation if a single trajectory has Z > max(0, B) + 1e-3 or a step has a_t > k_t + 1e-3. Do not reintroduce the empirical-Bernstein proxy, ρ = U_EBB/B_eff, the surrogate ensemble, or the k-DPP archive. If a mean estimate is genuinely needed, use an anytime-valid confidence sequence (`feat-007`).
- **Known truths from the released logs** (do not spend compute re-establishing them; cite `results/` once `feat-002` is done): constraint active in 0.19% of steps at k=3 and 6.3% at k=1; 57% of generations identical between k=3 and k=5; zero invariant violations in 26,999 trajectories (all 18 released files, 5.21M decode steps); 96/999 attack_train trajectories (738/8,999 across all six classes) exceed K in realised log-likelihood ratio at k=1; every N=20 held-out row in Stage 2 contains one duplicated sample (seed collision, `feat-003`); the anchor writes the first floor(δ_init/k) tokens of every answer (δ_init ≈ 6 nats). All reproduced in `results/` by `feat-002`.
- **Compute.** Local box: 4×A100 80GB (`nvidia-smi` indices 0,1,2,4; GPU 3 is a 4 GB T400, never use it; 0 and 4 may be partially occupied by other users, check before launching). DGX: 6×H100 through the `dgx-gpu` skill for the larger runs (70B base, sweeps). Set `CUDA_VISIBLE_DEVICES` explicitly for every job. When syncing to the DGX, sync only `data/`, code, and small results; never `output.zip` or `.venv`.
- **Secrets.** `HF_TOKEN` lives in `.env` (gitignored) but was invalid on 2026-09-05; run local jobs with `HF_HUB_OFFLINE=1` (cache has Llama-3.1-8B-Instruct, Llama-3.1-8B, TinyComma, Qwen2.5-7B-Instruct; 70B is not cached). `meta-llama/*` is gated: ask the user for a valid token before any download.
- **Do not modify** `~/sub/neurips_2026.tex`, `output.zip`, or anything under `data/`. New logs go to `output/` (gitignored); summaries go to `results/` (committed, small).
- **Anonymity.** Nothing you write into `~/sub/satml/` or the artifact may identify the authors.
- **Paper integrity.** The manuscript (proofread 2026-09-05) reports bank-and-burst as attempted but not evaluated (the fine-tuned memoriser ignores filler instructions) and claims no adaptive prompt search; do not describe feat-010/011/012 as run until their evidence exists. After any edit under `~/sub/satml/`, recompile and check the PDF for `??` and the log for overfull boxes (see Verification Commands).

## Key Facts

| Item | Value |
|---|---|
| Entry points | `h1.py` → `dap/e1.py` (fixed workload); `h2.py` → `dap/e2/runner.py` (search) |
| Core library | `a_patch/factory.py` (`AnchoredDecodingFactory`; `k_radius=-1.0` = risky only, `0.0` = safe only) |
| Stats | `dap/stats.py`: index-only `build_trajectory_seeds`, copying metrics, `budget_check` (per-trajectory max spend / utilisation / invariant), `anytime_valid_cs` (PrPl-EB). The empirical-Bernstein proxy is gone; E2 `rho` now means max per-trajectory utilisation and `certified` means every trajectory within budget |
| Models | safe `jacquelinehe/tinycomma-1.8b-llama3-tokenizer`, risky `meta-llama/Llama-3.1-8B-Instruct`, optimizer `Qwen/Qwen2.5-7B-Instruct` |
| Released logs | `output.zip` (2.4 GB; `unzip -o output.zip 'output/h1_outputs/trajectories_k3_attack_train.jsonl'` etc.) |
| Log reanalysis | `analysis/reanalyze_logs.py --logs output --out results` (feat-002; writes `results/regime_table.csv`, `llr_tails.csv`, `prefix_debt_forced_tokens.csv`, `surprisal.csv`, `seed_collisions.csv`, `per_trajectory.csv`) |
| Paper, bib, figures | `/mnt/md0/IITM/BackUp/Home/vijayavallabh/sub/satml/satml_2027.tex` + `sections/*.tex` (SaTML version; the arXiv version is `satml_2027_arxiv_v1.tex`), `references.bib` (51 entries), `figures/` (copied by `figures/make_figures.py --copy-to <that dir>`); compile with `~/.local/bin/tectonic -X compile satml_2027.tex` (no pdflatex on this box). `~` is `/home/sports` here, not the project home: use absolute paths |
| Tests | `tests/` (14 tests: seeds, invariant, chat template, metrics, confidence sequence, E2 stats); `./init.sh` runs them |

## Verification Commands

```bash
./init.sh                      # full baseline verification (fails fast)
.venv/bin/python -m pytest -q tests   # 14 tests
.venv/bin/python h1.py --k-values 1.0 --trajectories-per-prompt 2 \
  --cap-neutral 2 --cap-val 2 --cap-test 2 --cap-attack-train 2 --cap-factual 2 --cap-creative 2 \
  --output-dir output/smoke   # GPU smoke test for E1 (run on DGX)
```

```bash
# manuscript: compile, then confirm no unresolved references, no overfull boxes, <= 12 pages
cd /mnt/md0/IITM/BackUp/Home/vijayavallabh/sub/satml && ~/.local/bin/tectonic -X compile satml_2027.tex 2>&1 | grep -ci overfull; \
  pdftotext satml_2027.pdf - | grep -c '??'; /mnt/md0/IITM/BackUp/Home/vijayavallabh/DA5001_Project/.venv/bin/python -c "from pypdf import PdfReader; print(len(PdfReader('satml_2027.pdf').pages))"
```

Per-feature evidence commands are in `feature_list.json`.

## Definition of Done

A feature is done only when ALL of the following are true:

- [ ] The behaviour in its `description` exists and its evidence command ran; the command and its output are pasted into `evidence`.
- [ ] `./init.sh` passes.
- [ ] Any new number that will appear in the paper is written to `results/*.csv` with the producing command in `progress.md`.
- [ ] Repository remains restartable: a fresh session can run `./init.sh` immediately.

## End of Session

Before ending a session:

1. Update `progress.md` (Current State, What's Done, What's Next, Blockers).
2. Update `feature_list.json` status and evidence.
3. Fill `session-handoff.md` (Current Objective, Files Changed, Recommended Next Step).
4. Commit with a descriptive message; leave the tree clean and restartable.

## Escalation

- **GPU job fails or DGX unreachable:** record in `progress.md`, move to a feature that needs no GPU (`feat-002`, `feat-013` prep, `feat-014`).
- **A result contradicts a Known Truth above:** treat it as a bug in the new code until proven otherwise; do not rewrite the paper's claims on one run.
- **Fine-tuning cost or any run > 24 GPU-hours:** stop and ask.
- **Anything that would identify authors, delete `output.zip`, or push to a remote:** ask.
