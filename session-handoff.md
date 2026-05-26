# Session Handoff

## Current Objective

- Goal:
- Current status:
- Branch / commit:

## Completed This Session

- [ ]

## Verification Evidence

| Check | Command | Result | Notes |
|---|---|---|---|
| Env ready | `./init.sh` | | |
| E1 ready | `python h1.py --help` | | |
| E2 ready | `python h2.py --help` | | |
| GPU available | `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` | | |

## Files Changed

-

## Decisions Made

-

## Blockers / Risks

- HF_TOKEN required for gated Llama model; set in `.env`
- E2 requires ≥2 GPUs for best results (risky:cuda:0, safe:cuda:1, optimizer:cuda:1)

## Next Session Startup

1. Read `AGENTS.md`
2. Read `feature_list.json` and `progress.md`
3. Review this handoff
4. Run `./init.sh` before editing

## Recommended Next Step

-

## End-of-Session Procedure

Before ending next session:
1. Update `progress.md` with current state, blockers, files changed
2. Update `feature_list.json` feature status(es) and evidence
3. Commit all work with a descriptive message
4. Fill in this handoff with next-session context
5. Verify `./init.sh` still passes
