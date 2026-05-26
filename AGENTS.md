# DA5001 Project — Anchored Decoding Certification

Anchored KL-constraint experiments on LLMs: **E1** certifies KL budget across 6 prompt classes; **E2** adversarially optimizes prompts to maximize KL spend.

## Startup Workflow

1. **Activate environment:** `source .venv/bin/activate`
2. **Read this file** completely
3. **Read `feature_list.json`** for current feature state
4. **Run `./init.sh`** to verify the environment is healthy
5. **Review recent commits** with `git log --oneline -5`

If `./init.sh` fails, repair that first before adding new scope.

## Working Rules

- **One feature at a time** — pick exactly one unfinished feature from `feature_list.json`
- **Verification required** — don't claim done without running `./init.sh` and collecting evidence
- **Update artifacts** — before ending a session, update `progress.md` and `feature_list.json`
- **Stay in scope** — don't modify files unrelated to the current feature; log unrelated issues but don't fix them unless instructed
- **Leave clean state** — next session must be able to run `./init.sh` immediately

## Required Artifacts

| File | Purpose |
|---|---|
| `feature_list.json` | Feature definitions, dependencies, status, evidence |
| `progress.md` | Session progress log with blockers, decisions, next steps |
| `init.sh` | Startup verification script (run at session start) |
| `session-handoff.md` | End-of-session handoff for multi-session work |

## Definition of Done

A feature is done when ALL are true:

- [ ] Target behavior is implemented per the feature's description in `feature_list.json`
- [ ] `./init.sh` passes cleanly
- [ ] Evidence recorded in `feature_list.json` or `progress.md`
- [ ] Repository is restartable from standard startup path

## Quick Reference

| Command | Purpose |
|---|---|
| `python h1.py --help` | E1 options |
| `python h2.py --help` | E2 options |
| `./init.sh` | Full environment check |
| `python h1.py --k-values 1.0 --trajectories-per-prompt 2 --cap-* 2` | Quick E1 smoke test |
| `python h2.py --k 3.0` | Quick E2 smoke test |

**Module layout:** `a_patch/` (core library), `dap/` (experiments), `h1.py` / `h2.py` (thin entry points).

**Key facts:**
- GPU required; ≥2 GPUs auto-distributes risky:cuda:0, safe:cuda:1
- `meta-llama/Llama-3.1-8B-Instruct` is gated (`HF_TOKEN` in `.env`)
- Copybench/creative prompts wrapped in `"Complete the prefix:\n" + text`; factscore prompts are not
- `--num-classes` must equal `len(CLASS_ORDER)` (default 6)
- 6 JSONL files in `data/` via `PromptNormalizer`
- No test suite, no linter

## End of Session

1. Update `progress.md` with current state, blockers, files changed
2. Update `feature_list.json` with new feature status and evidence
3. Commit all work with a descriptive message
4. Fill in `session-handoff.md` with next-session context
5. Verify `./init.sh` still passes
