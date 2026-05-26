# Session Progress Log

## Current State

**Last Updated:** 2026-05-26
**Active Feature:** (none — harness setup complete)

## Status

### What's Done

- [x] AGENTS.md created with startup workflow, rules, done criteria, end-of-session
- [x] feature_list.json updated with current state (2 complete, 2 pending)
- [x] init.sh verification script created
- [x] session-handoff.md template created
- [x] progress.md created (this file)
- [x] Monolithic a_patch.py/h1.py/h2.py refactored into a_patch/ + dap/ packages
- [x] Fixed _solve_theta_newton regression (restored original robust implementation)
- [x] Fixed _get_logp_from_weights regression (restored log-space renormalization with NaN guards)
- [x] Fixed _compute_prefix_debt_fast: was computing SUM instead of MEAN of top-k positive LLRs, inflating prefix debt by up to k_eff×
- [x] Fixed get_kl_stats_summary: restored structured-zero dict with type narrowing and NaN/inf budget handling
- [x] Fixed lineage_id_for_candidate: Python hash() → deterministic stable_hash()
- [x] Fixed adaptive_eval_topup_fraction default: 0.75 → 0.5 (matches original)
- [x] Intentional improvement: ebb_upper_bound_chapman now uses empirical-R variant (from h2.py) for both experiments — tighter bound, same statistical guarantee

### What's In Progress

- (none)

### What's Next

1. Pick a feature from `feature_list.json` (feat-e1 or feat-e2)
2. Run `./init.sh` to verify environment
3. Implement or execute the selected feature
4. Collect evidence and update this log

## Blockers / Risks

- `meta-llama/Llama-3.1-8B-Instruct` requires HF_TOKEN; gated download may fail
- E2 requires ≥2 GPUs for best results (risky:cuda:0, safe:cuda:1, optimizer:cuda:1)
- No test suite; manual verification via experiment runs

## Decisions Made

- **Modular refactoring (2026-05-26)** — Split flat files into a_patch/ (core lib) and dap/ (experiments) packages to eliminate code duplication and improve maintainability
- **Harness files created (2026-05-26)** — AGENTS.md, feature_list.json, progress.md, init.sh, session-handoff.md per harness-creator skill

## Files Modified This Session

- `AGENTS.md` — Restructured with startup workflow, rules, artifacts, done criteria, end-of-session
- `feature_list.json` — Updated statuses, fixed stale references
- `progress.md` — Updated to reflect current session state
- `session-handoff.md` — Filled in with end-of-session procedure

## Evidence of Completion

- [ ] `./init.sh` passes
- [ ] `python h1.py --help` prints usage
- [ ] `python h2.py --help` prints usage

## Notes for Next Session

- Both experiments are ready to run but haven't been executed yet
- Quick smoke test: `python h1.py --k-values 1.0 --trajectories-per-prompt 2 --cap-* 2`
- Full runs take hours on GPU; use nohup commands in README.md
