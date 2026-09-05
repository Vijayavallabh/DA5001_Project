# Session Handoff

## Current Objective

- Goal: ship the SaTML 2027 submission described in `~/sub/satml/IMPROVEMENT_PLAN.md` (see `GOAL.md` for the executable statement).
- Current status: feat-001 and feat-002 done; `results/` holds the reproduced log numbers. Next: feat-003 (seed fix, tests), then feat-006.
- Branch / commit: `master` @ 07a446c (harness) + feat-002 commit.

## Completed This Session

- [x] Reframing plan and literature review (`~/sub/satml/IMPROVEMENT_PLAN.md`).
- [x] Log reanalysis establishing the Known Truths (see `AGENTS.md`).
- [x] Harness: `AGENTS.md`, `CLAUDE.md`, `feature_list.json`, `progress.md`, `init.sh`, `GOAL.md`, this file.
- [x] feat-002: `analysis/reanalyze_logs.py` → six `results/*.csv`; Known Truths reproduced.

## Verification Evidence

| Check | Command | Result | Notes |
|---|---|---|---|
| Baseline | `./init.sh` | PASS (exit 0, 2026-09-05) | feat-001 closed |
| Imports | `.venv/bin/python -c "import a_patch, dap.shared"` | PASS | covered by init.sh step 2 |
| GPU | `.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` | True, 5 local GPUs | DGX (6×H100) also available via `dgx-gpu` |
| feat-002 | `.venv/bin/python analysis/reanalyze_logs.py --logs output --out results` | PASS (0.195% active at k=3, 96/999 L>K at k=1, 0 violations) | 6.8 s, no GPU |
| Harness score | `node /home/sports/.agents/skills/harness-creator/scripts/validate-harness.mjs --target .` | 100/100 | structural score only |

## Files Changed

- `AGENTS.md`, `CLAUDE.md`, `feature_list.json`, `progress.md`, `session-handoff.md`, `init.sh`, `GOAL.md`, `analysis/reanalyze_logs.py`, `results/*.csv`

## Decisions Made

- See `progress.md` → Decisions Made (retire EBB/ρ; audit at small k; attacks over search).

## Blockers / Risks

- feat-008 (memorising model) is the schedule risk; decide 70B-base vs 8B-finetune by Sep 12.
- Human-only steps (feat-016): abstract Sep 22, paper Sep 29, artifacts Oct 2.

## Next Session Startup

1. Read `AGENTS.md` (auto-imported by `CLAUDE.md`).
2. Read `feature_list.json` and `progress.md`.
3. Review this handoff.
4. Run `./init.sh` before editing.

## Recommended Next Step

- Start feat-003: replace `build_trajectory_seeds` with collision-free hashed seeds, add utilisation/activity logging, unify the range parameter, add `--use-chat-template`, create `tests/`.
