# Session Handoff

## Current Objective

- Goal: ship the SaTML 2027 submission described in `~/sub/satml/IMPROVEMENT_PLAN.md` (see `GOAL.md` for the executable statement).
- Current status: feat-001..003 done. Next: feat-006 (certificate-strength audit, local GPU), then feat-004/005.
- Branch / commit: `master`, one commit per feature (feat-001 07a446c, feat-002 b717255, feat-003 next).

## Completed This Session

- [x] Reframing plan and literature review (`~/sub/satml/IMPROVEMENT_PLAN.md`).
- [x] Log reanalysis establishing the Known Truths (see `AGENTS.md`).
- [x] Harness: `AGENTS.md`, `CLAUDE.md`, `feature_list.json`, `progress.md`, `init.sh`, `GOAL.md`, this file.
- [x] feat-002: `analysis/reanalyze_logs.py` → six `results/*.csv`; Known Truths reproduced.
- [x] feat-003: seeds, utilisation/activity logging, R = K, `--use-chat-template`, `tests/` (6 passing), GPU smoke OK.

## Verification Evidence

| Check | Command | Result | Notes |
|---|---|---|---|
| Baseline | `./init.sh` | PASS (exit 0, 2026-09-05) | feat-001 closed |
| Imports | `.venv/bin/python -c "import a_patch, dap.shared"` | PASS | covered by init.sh step 2 |
| GPU | `.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` | True, 5 local GPUs | DGX (6×H100) also available via `dgx-gpu` |
| feat-002 | `.venv/bin/python analysis/reanalyze_logs.py --logs output --out results` | PASS (0.195% active at k=3, 96/999 L>K at k=1, 0 violations) | 6.8 s, no GPU |
| feat-003 | `.venv/bin/python -m pytest -q tests && ./init.sh` | PASS (6 passed, exit 0) | plus GPU smoke on local GPU 1 |
| Harness score | `node /home/sports/.agents/skills/harness-creator/scripts/validate-harness.mjs --target .` | 100/100 | structural score only |

## Files Changed

- `AGENTS.md`, `CLAUDE.md`, `feature_list.json`, `progress.md`, `session-handoff.md`, `init.sh`, `GOAL.md`, `analysis/reanalyze_logs.py`, `results/*.csv`
- feat-003: `dap/stats.py`, `dap/shared.py`, `dap/e1.py`, `dap/e2/{evaluator,runner,types}.py`, `a_patch/factory.py`, `tests/`

## Decisions Made

- See `progress.md` → Decisions Made (retire EBB/ρ; audit at small k; attacks over search).

## Blockers / Risks

- feat-008 (memorising model) is the schedule risk; decide 70B-base vs 8B-finetune by Sep 12.
- `HF_TOKEN` in `.env` is invalid; local runs use `HF_HUB_OFFLINE=1` against the cache (see progress.md Blockers). 70B is not cached.
- Human-only steps (feat-016): abstract Sep 22, paper Sep 29, artifacts Oct 2.

## Next Session Startup

1. Read `AGENTS.md` (auto-imported by `CLAUDE.md`).
2. Read `feature_list.json` and `progress.md`.
3. Review this handoff.
4. Run `./init.sh` before editing.

## Recommended Next Step

- Start feat-006: `analysis/certificate_cap.py` — TinyComma surprisal S(x) of each CopyBench reference given its prefix, caps min(1,(K+ln2)/S) for K in {20,30,50,100,200,600,1000}, `results/certificate_caps.csv`, `figures/certificate_cap_curve.pdf`. Run locally on a free A100 with `HF_HUB_OFFLINE=1`.
