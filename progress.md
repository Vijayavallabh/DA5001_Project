# Session Progress Log

## Current State

**Last Updated:** 2026-09-05
**Active Feature:** none — feat-001, feat-002 done; next eligible is feat-003 (feat-006 also eligible)
**Deadline clock:** abstract Sep 22 · paper Sep 29 · artifacts Oct 2 (AoE)

## Status

### What's Done

- [x] Full read of repo, both manuscripts, and the released logs; literature review (He et al. 2026, Vyas et al. 2023, Cohen 2025, ~35 related papers).
- [x] Reframing plan written: `~/sub/satml/IMPROVEMENT_PLAN.md` (spec for everything below).
- [x] Log reanalysis scripts drafted: `analysis/reanalyze_logs.py`, `analysis/surprisal.py` (copies also in `~/sub/satml/scripts/`).
- [x] Harness rebuilt: `AGENTS.md`, `CLAUDE.md` (imports AGENTS.md), `feature_list.json`, `progress.md`, `session-handoff.md`, `init.sh`, `GOAL.md`.

- [x] feat-001 closed: `./init.sh` exit 0; harness validator 100/100. Committed as 07a446c.
- [x] feat-002 closed (2026-09-05): `analysis/reanalyze_logs.py --logs output --out results` (6.8 s, no GPU) reproduces every Known Truth. Producing command: `unzip -o output.zip <4 trajectory files + heldout_validation.jsonl> && .venv/bin/python analysis/reanalyze_logs.py --logs output --out results`. Outputs: `results/regime_table.csv`, `llr_tails.csv`, `prefix_debt_forced_tokens.csv`, `surprisal.csv`, `seed_collisions.csv`, `per_trajectory.csv` (4,499 rows, 763 KB; input for feat-007/feat-013). New number: leading safe-forced tokens equal floor(δ_init/k) in 99.4–100% of trajectories. `analysis/surprisal.py` merged into the tool and removed.

### What's In Progress

- (none)

### What's Next

1. feat-003 (no GPU): seed-collision fix, utilisation logging, chat-template flag, first tests.
2. feat-006 (GPU, independent of code changes): certificate-strength audit; can run in parallel with feat-003 on the DGX.
3. feat-004 → feat-005: baselines and the small-k regime sweep.
4. feat-008 → feat-009: memorising model, then the composition attack (the paper's core attack result).
5. feat-007, feat-013, feat-014, feat-015 in that order. feat-010/011/012 only if time remains after feat-009.

## Blockers / Risks

- [ ] Compute: local 4×A100 80GB (GPUs 0 and 4 had ~15 GB in use by other processes on 2026-09-05; GPU 3 is a T400, unusable). DGX 6×H100 via `dgx-gpu` for 70B and sweeps. Llama-3.1-70B base in bf16 (~140 GB) fits on 2 free A100s with `device_map="auto"`, so feat-008's 70B option is feasible locally.
- [ ] `meta-llama/Llama-3.1-8B-Instruct` is gated; `HF_TOKEN` must be present in `.env` on the DGX as well.
- [ ] feat-008 (memorising model) is the schedule risk: Llama-3.1-70B base needs ~140 GB bf16; fine-tuning 8B is the fallback (≈2–4 GPU-hours). Decide by Sep 12.
- [ ] Known code issues to fix in feat-003, not before: seed collision in `dap/e2/runner.py::_eval_specs` (offset n0) and the top-up path (offset 10); E1 uses R = T·ln|V| while E2 uses R = K; `B_eff` conflates generation length with budget; instruct model called without chat template.
- [ ] `trajectories_k1_attack_train.jsonl` has no `p_risky_prob` field (older run), so `L_risky` is 0 for k=1 in `results/`; the k=1 attack_train file has 999 rows, not 1000. Only 4 of 18 h1 trajectory files are reanalysed so far (the evidence set); extend to val/neutral/factual/creative in feat-013 if a per-domain table is needed.
- [ ] Unverified claims in the current manuscript: "Regime A pivot is the more common behaviour" (a crude check finds ~3–4%); the "not i.i.d. common random numbers" caveat is unnecessary (`torch.multinomial` rows are independent).

## Decisions Made

- **Retire the empirical-Bernstein proxy and ρ (2026-09-05).** Z ≤ K holds per trajectory by construction (KL chain rule), so a mean bound cannot fail; report per-trajectory utilisation, realised LLR tails, and certificate caps instead. Alternatives considered: keep EBB with R = K (bound gets looser, still vacuous), betting CS on Z/K (only where a mean is genuinely the target).
- **Audit at He et al.'s small-k regime (2026-09-05).** At k=3/5 the constraint was active in ≤0.2% of steps; the sweep now spans {0.1 … 1} plus k=−1/0 baselines.
- **Attacks over search (2026-09-05).** Drop the surrogate/k-DPP/ρ search; the SaTML core is the composition attack (Cohen Thm 3.5 instantiated), bank-and-burst, and a memorising risky model.

## Files Modified This Session

- `AGENTS.md`, `CLAUDE.md`, `feature_list.json`, `progress.md`, `session-handoff.md`, `init.sh`, `GOAL.md` — harness rebuilt for the SaTML plan.
- `analysis/reanalyze_logs.py` — argparse reanalysis tool (feat-002); `results/*.csv` generated.
- `~/sub/satml/IMPROVEMENT_PLAN.md`, `~/sub/satml/scripts/*` — plan and script copies (outside repo).

## Evidence of Completion

- [x] `./init.sh` passes (2026-09-05): all [OK] lines, `[info] local GPU available: True, count: 5`, `=== Init Complete ===`, exit 0.
- [x] `node .../harness-creator/scripts/validate-harness.mjs --target .` → Overall 100/100 (all five subsystems 5/5).
- [x] feat-002 evidence command exit 0; see `feature_list.json`.

## Notes for Next Session

- The numbers in AGENTS.md "Known truths" were computed this session from `output.zip`; feat-002 must reproduce them exactly before anything is cited.
- `output.zip` is 2.4 GB; extract only the four trajectory files listed in feat-002's evidence command.
- Reproduction commands for the original runs are in `README.md`; keep them for the artifact's "original runs" section.
