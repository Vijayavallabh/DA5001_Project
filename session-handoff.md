# Session Handoff

## Current Objective

- Goal: ship the SaTML 2027 submission described in `~/sub/satml/IMPROVEMENT_PLAN.md` (see `GOAL.md` for the executable statement).
- Current status: GOAL COMPLETE (2026-09-05). feat-001..009 and 013..015 done; feat-010/011/012 optional and not started; feat-016 is human-only.
- **For the human (feat-016):** PDF `/mnt/md0/IITM/BackUp/Home/vijayavallabh/sub/satml/satml_2027.pdf` (11 pages incl. references, anonymous; proofread 2026-09-05 21:45; ethics section + deployer sentence added 22:20); artifact zip `/mnt/md0/IITM/BackUp/Home/vijayavallabh/DA5001_Project/artifact.zip` (5.0 MB, 75 files, manifest verified, built 2026-09-05 22:25; rebuild with `scripts/build_artifact.sh artifact`); add `output.zip` (2.4 GB logs) to the Zenodo record and paste the DOI into `sections/open_science.tex`.
- Branch / commit: `master`, one commit per feature; latest work: proofread (6a3cd85), .gitignore + artifact rebuild (096cc83), handoff (f57d598), harness refresh (this commit).

## Completed This Session

- [x] Reframing plan and literature review (`~/sub/satml/IMPROVEMENT_PLAN.md`).
- [x] Log reanalysis establishing the Known Truths (see `AGENTS.md`).
- [x] Harness: `AGENTS.md`, `CLAUDE.md`, `feature_list.json`, `progress.md`, `init.sh`, `GOAL.md`, this file.
- [x] feat-002: `analysis/reanalyze_logs.py` → six `results/*.csv`; Known Truths reproduced.
- [x] feat-003: seeds, utilisation/activity logging, R = K, `--use-chat-template`, `tests/` (6 passing), GPU smoke OK.
- [x] feat-004: k=-1/0 baselines, LCS/ACS/nv-recall metrics, batched seeds; smoke OK.
- [x] feat-006: certificate-strength audit; certificate vacuous for 100% of CopyBench passages at k=3, 44% at k=1.
- [x] feat-005/007: small-budget sweeps (plain + chat, 37,800 trajectories, 0 violations), LLR tails with anytime-valid CS, EBB retired.
- [x] feat-008/009: LoRA memoriser (greedy nv-recall 0.91 train, 0.0 held-out); composition attack up to k=20 (0 violations; 0.48 single / 0.86 oracle at k=20).
- [x] feat-013/014/015: figures from CSVs, manuscript rewrite (11 pages, 48 refs cited), anonymised artifact (75 files, manifest verified).
- [x] Proofread pass on the PDF (missing table, bank-and-burst stated as not evaluated, numbers realigned, bib names); artifact rebuilt; `.gitignore` cleaned.

## Verification Evidence

| Check | Command | Result | Notes |
|---|---|---|---|
| Baseline | `./init.sh` | PASS (exit 0, 2026-09-05) | feat-001 closed |
| Imports | `.venv/bin/python -c "import a_patch, dap.shared"` | PASS | covered by init.sh step 2 |
| GPU | `.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` | True, 5 local GPUs | DGX (6×H100) also available via `dgx-gpu` |
| feat-002 | `.venv/bin/python analysis/reanalyze_logs.py --logs output --out results` | PASS (0.195% active at k=3, 96/999 L>K at k=1, 0 violations) | 6.8 s, no GPU |
| feat-003 | `.venv/bin/python -m pytest -q tests && ./init.sh` | PASS (6 passed, exit 0) | plus GPU smoke on local GPU 1 |
| feat-006 | `CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 .venv/bin/python analysis/certificate_cap.py --data data --out results` | PASS (exit 0) | ~2 min, one A100 |
| feat-004 | `h1.py --k-values -1 0 0.15 1 ... --output-dir output/smoke` (see feature_list.json) | PASS (exit 0, 24 files) | local GPUs 1+2 |
| feat-005 | `analysis/regime_sweep.py --run plain=... --run chat=... --out results` | PASS (98 rows, 0 violations) | 37,800 trajectories |
| feat-007 | `pytest -q tests/test_cs.py && cat results/llr_tails.csv && grep -rn ebb_upper_bound_chapman dap \| wc -l` | PASS (3 tests, 79 rows, 0 call sites) | |
| feat-009 | `analysis/composition_attack.py ... --k-values -1 0 0.15 0.5 1 3 5 10 20` | PASS (45 summary rows, 0 violations) | 70 min |
| feat-013 | `figures/make_figures.py --copy-to <sub/satml/figures>` | PASS (4 figures) | |
| feat-014 | `tectonic -X compile satml_2027.tex` | PASS (11 pages, 49 refs cited of 52, 0 `??`, 0 overfull) | pdflatex unavailable here; proofread 21:45, CFP pass 22:40 |
| feat-015 | `scripts/build_artifact.sh artifact` | PASS (75 files, manifest verified) | last built 2026-09-05 22:25 |
| Harness score | `node /home/sports/.agents/skills/harness-creator/scripts/validate-harness.mjs --target .` | 100/100 | structural score only |

## Files Changed

- `AGENTS.md`, `CLAUDE.md`, `feature_list.json`, `progress.md`, `session-handoff.md`, `init.sh`, `GOAL.md`, `analysis/reanalyze_logs.py`, `results/*.csv`
- feat-003: `dap/stats.py`, `dap/shared.py`, `dap/e1.py`, `dap/e2/{evaluator,runner,types}.py`, `a_patch/factory.py`, `tests/`
- feat-004..009: `analysis/{certificate_cap,regime_sweep,llr_tails,composition_attack,bank_burst,memorizing_recall}.py`, `recipes/`, `scripts/run_*.sh`, `results/*.csv`
- feat-013..015 and after: `figures/make_figures.py`, `figures/*.pdf`, `scripts/build_artifact.sh`, `README_artifact.md`, `artifact/`, `.gitignore`; manuscript `/mnt/md0/IITM/BackUp/Home/vijayavallabh/sub/satml/{satml_2027.tex,sections/*.tex,references.bib,figures/}` (outside this repo)

## Decisions Made

- See `progress.md` → Decisions Made (retire EBB/ρ; audit at small k; attacks over search).

## Blockers / Risks

- feat-008 decided: 8B LoRA memoriser (70B not cached, token invalid, DGX SSH denied). feat-010 (bank-and-burst) is not testable with it; feat-011/012 unstarted.
- `HF_TOKEN` in `.env` is invalid; local runs use `HF_HUB_OFFLINE=1` against the cache (see progress.md Blockers). 70B is not cached.
- Human-only steps (feat-016): abstract Sep 22, paper Sep 29, artifacts Oct 2.

## Submission checklist for the human (from https://satml.org/call-for-papers/ and its checklist, Sep 4 2026 version)

- Sep 22 (abstract registration): title + abstract (tentative wording, no substantial change later), final authors and topics, ORCID for every author, Author Certification, mandatory conflicts, one author nominated as author-reviewer (may be asked to review up to three papers per submission), answer the "under review elsewhere" field, enter `N/A` in New Insights.
- Sep 29 (paper): `sub/satml/satml_2027.pdf`; anonymised repository link (e.g. anonymous.4open.science) containing `artifact.zip` contents plus `output.zip`; re-check conflicts in the last 24 h; optional LLM-processing opt-in flag; self-check references with hallucinator.science (all 52 entries were checked against arXiv/publisher metadata here, but the CFP asks authors to run the tool).
- Confirmed by the author on 2026-09-05: the NeurIPS/arXiv version (`sub/neurips_2026.tex`, arXiv 2605.28001) received no reviews at any venue, so nothing needs appending, and the paper is not under submission elsewhere (answer "no" to the HotCRP under-review field).
- Oct 2: last edit of the anonymised repository; it must then stay accessible and unchanged through review.
- Paper end matter is already in the CFP order (Open Science → LLM usage considerations → Ethical Considerations → references) and contains the required editorial-use sentence; the arXiv audit is not cited (own work would have to be in the third person); add it at camera-ready if wanted.
- If accepted: Zenodo by Jan 14 2027 (paste the DOI into `sections/open_science.tex`), camera-ready mid-Feb 2027, in-person presentation early May 2027 with one full registration.

## Next Session Startup

1. Read `AGENTS.md` (auto-imported by `CLAUDE.md`).
2. Read `feature_list.json` and `progress.md`.
3. Review this handoff.
4. Run `./init.sh` before editing.

## Recommended Next Step

- Human: feat-016 (abstract Sep 22, paper Sep 29, artifact + `output.zip` to Zenodo Oct 2; paste the DOI into `sections/open_science.tex`, decide on the author block placeholder).
- Human, committed in the paper's Ethical Considerations section (kept on 2026-09-05): share the audit findings and code with the Anchored Decoding authors (He et al.) once the review outcome permits.
- Agent, if asked: feat-011 (format evasion) and feat-012 (anchor copying of public-domain text) need only the local GPUs; feat-010 needs a different memoriser.
