# GOAL: ship the SaTML 2027 submission "What does a KL budget certify?"

You are executing this goal autonomously in the repository `/mnt/md0/IITM/BackUp/Home/vijayavallabh/DA5001_Project`. Read `AGENTS.md` first (it is auto-imported by `CLAUDE.md`), then this file, then `/mnt/md0/IITM/BackUp/Home/vijayavallabh/sub/satml/IMPROVEMENT_PLAN.md` Sections 0–4. The plan is the spec; this file is the contract.

**Status (2026-09-05):** success criteria 1–6 hold (see `progress.md` → GOAL COMPLETE). Remaining work is optional feat-010/011/012 and the human-only feat-016.

## Objective

Turn the arXiv paper "An Empirical Audit of k-NAF Budget Accounting for Anchored Decoding" into a SaTML 2027 submission whose central result is an audit that can fail: an empirical, security-framed measurement of what Anchored Decoding's KL budget K = k·T_max actually certifies, plus at least one adaptive attack that reconstructs copyrighted text while every query stays within budget.

## Success criteria (all must hold)

1. `feature_list.json` features feat-001 through feat-009 and feat-013 through feat-015 are `done`, each with its evidence command and output pasted in. feat-010, feat-011, feat-012 are optional and attempted only after feat-009 is done.
2. `results/` contains, at minimum: `regime_table.csv`, `regime_sweep.csv`, `certificate_caps.csv`, `llr_tails.csv`, `memorizing_model_recall.csv`, `composition.csv`. Every number in the manuscript traces to one of these files.
3. `~/sub/satml/satml_2027.tex` compiles with `pdflatex` + `bibtex` (or with `tectonic`, the accepted equivalent on this box, where pdflatex is unavailable; see `progress.md` Decisions), has ≤ 12 pages of body text in IEEEtran, contains a threat-model section, an Open Science section, and an LLM-usage section, and `references.bib` has ≥ 40 complete entries.
4. `artifact/` exists with an anonymised code snapshot, `results/*.csv`, prompt sets, the memorising-model recipe, a README with exact reproduction commands, and a verified `MANIFEST.sha256`.
5. `./init.sh` passes and the git tree is committed after every completed feature.
6. `session-handoff.md` names the final PDF path and the artifact zip path for the human to submit (feat-016).

## Non-goals

- Do not run the old ρ / U_EBB / surrogate / k-DPP search. Do not add new tables about the Bernstein proxy.
- Do not modify `~/sub/neurips_2026.tex`, `output.zip`, or `data/`.
- Do not register the abstract, submit the paper, or upload to Zenodo. Those are human-only (feat-016).
- Do not push to any remote.

## Constraints

- Dates (AoE): abstract Sep 22 2026, paper Sep 29 2026, artifacts Oct 2 2026. Plan work so that a compilable draft with feat-005, feat-006, feat-007, feat-009 results exists by Sep 22, leaving the last week for writing and packaging.
- Compute: local 4×A100 80GB (indices 0,1,2,4; never index 3; check `nvidia-smi` for other users' load first) for smoke tests, the certificate audit, and 8B runs; the DGX (6×H100, `dgx-gpu` skill) for the 70B base model and the full sweeps. Always set `CUDA_VISIBLE_DEVICES`. Stop and ask before any single run expected to exceed 24 GPU-hours or before starting fine-tuning larger than 8B.
- Baselines: every experiment reporting a spend or copying metric at some k also reports k = −1 and k = 0 on the same prompts and seeds.
- Violations: a budget violation is a per-trajectory event (Z > max(0,B) + 1e-3 or a_t > k_t + 1e-3), never a mean-bound artefact. A result contradicting the Known Truths in `AGENTS.md` is a bug until proven otherwise.
- Anonymity: nothing written into `~/sub/satml/` or `artifact/` may identify authors.
- Scope: one feature `in-progress` at a time; dependencies in `feature_list.json` are binding.

## Execution order

1. feat-001 → feat-002 (no GPU). Reproduce the Known Truths into `results/regime_table.csv` before touching experiment code.
2. feat-003 in parallel with feat-006 on the DGX (feat-006 needs no code changes; it is one forward pass per CopyBench passage with TinyComma).
3. feat-004 → feat-005 (regime sweep; the "was the constraint active" figure).
4. feat-008 → feat-009 (memorising model → composition attack). This is the headline attack. Decide 70B-base vs 8B-finetune by Sep 12 and record the decision in `progress.md`.
5. feat-007 (event-level tails + anytime-valid CS, retire EBB).
6. feat-013 → feat-014 → feat-015 (figures → manuscript → artifact).
7. Only then feat-010, feat-011, feat-012, in that order, as time permits.

## Autonomy rules

- Proceed without asking for: code changes inside this repo, new files under `analysis/`, `results/`, `tests/`, `recipes/`, `artifact/`, `figures/`, edits to `~/sub/satml/satml_2027.tex`, `references.bib`, `figures/`, GPU jobs within the constraints above, and commits on `master`.
- Ask before: fine-tuning above 8B, any run > 24 GPU-hours, deleting any file you did not create, changing the paper's title or author block, anything touching a remote.
- If the DGX is unreachable, work on the next feature that needs no GPU (feat-002, feat-003, feat-007's CS code, feat-014's related-work and threat-model sections) and record the blocker.
- If a feature's evidence command fails twice after genuine fixes, mark it `blocked` with the failure output in `progress.md` and move to the next eligible feature.

## Reporting

- After each feature: update `feature_list.json` (status + evidence), `progress.md` (What's Done / What's Next / Blockers), commit with a message naming the feature id.
- At the end of every session: fill `session-handoff.md`, run `./init.sh`, commit.
- When success criteria 1–6 hold: write a final entry in `progress.md` titled "GOAL COMPLETE" listing the PDF path, artifact path, page count, reference count, and the single most important number from each `results/*.csv`. Then stop.

## Prompt to start a session

> Read GOAL.md and execute it. Start with ./init.sh, then pick the lowest-numbered eligible feature in feature_list.json.
