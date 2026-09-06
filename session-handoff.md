# Session Handoff

## Current Objective

- Goal: phase 1 COMPLETE (2026-09-05, commit 7b6961e). **Phase 2 COMPLETE 2026-09-06 19:30: feat-017..027 all done** under `~/sub/satml/IMPROVEMENT_PLAN.md` v2. Manuscript v2 final (12 body pages), artifact v2 built. Only the human-only feat-016 steps remain. Human decisions taken 2026-09-06: D1 = `unsloth/Meta-Llama-3.1-70B` mirror (cached in `hf_cache/`), D2 = 70B on GPUs 1+2, 8B jobs on shared GPUs 0/4 (PCI order).
- Every paper number is in `results/*.csv` with its producing command in `progress.md` (entries 2026-09-06 02:20 → 11:10). New this phase: `natural_memorisation.csv`, `composition_70b.csv`, `composition_8b_{kl,pathwise}.csv`, `odometer.csv`, `bank_cap.csv`, `concentration_summary.csv`, `pathwise_price.csv`, `extraction_cost_{kl,pathwise}.csv`, `prefix_debt_ablation.csv`, `warped_anchor.csv`, `budget_path.csv`, `latent_leakage_summary.csv`, `burst_audit.csv`; figures `natural_memorisation.pdf`, `budget_path.pdf`.
- Manuscript (`/mnt/md0/IITM/BackUp/Home/vijayavallabh/sub/satml/satml_2027.tex`): **v2 complete and compressed (feat-026): body ends on page 12, 18 pages with references and appendix, 0 overfull, 0 ??**; checkpoint `satml_2027_phase2_2026-09-06.pdf` (phase-1 PDF: `satml_2027_phase1_2026-09-06.pdf`). Bibliography: 128 entries, 125 cited, all verified (`bibcheck_2026-09-06/`). Section backups: `related_work_v2_2026-09-06.tex`, `intro_v1_2026-09-05.tex`. Artifact v2 built (`artifact/`, `artifact.zip` 23 MB, 179 files, manifest verified; `hf_cache/` excluded).
- Nothing is running. The same-sweep KL runs (`output/phase2/kl_sweep_hi` k=3, `kl_sweep_k5`, `kl_sweep_k10`, `kl_sweep_k20`) and the low-budget retry run (`retries_pathwise_lo`) finished on 2026-09-06 and their numbers are in `results/pathwise_price.csv`, `results/concentration_summary.csv`, `results/extraction_cost_pathwise_lo.csv` and in the manuscript.
- Next (human, feat-016): read `sub/satml/satml_2027_phase2_2026-09-06.pdf`; abstract registration by Sep 22 (the abstract in `satml_2027.tex` is final), paper by Sep 29, anonymised artifact repository (upload `artifact.zip`, 179 files, manifest inside) by Oct 2. Before any further edit: recompile and re-check that the body still ends on page 12 (`pypdf`: page 13 must start with 'Open Science'). Optional polish if time allows: a fresh-eyes proofread of Sections VI-E, VI-F and VII (written 2026-09-06), and a final hallucinator run (the cited set of 125 entries is unchanged since the 02:50 verification).
- **For the human (feat-016, unchanged):** registration Sep 22, paper Sep 29, artifact repo Oct 2.
- Branch / commit: `master`; latest work: this commit.
## Resolved Decision

- The earlier audit (arXiv 2605.28001) is now cited in the third person as `\cite{vijayavallabh2026audit}` (reference [6]) at six places: intro, threat model, Section V opener, Related Work (one sentence contrasting the two audits), Open Science, LLM usage. Decided by the user 2026-09-05 23:55. The reference list therefore names the author, which the CFP permits for third-person self-citation.

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
- [x] feat-013/014/015: figures from CSVs, manuscript rewrite (11 pages; now 50 refs cited of 53 after the self-citation), anonymised artifact (75 files, manifest verified).
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
| feat-014 | `tectonic -X compile satml_2027.tex` | PASS (11 pages, 50 refs cited of 53, 0 `??`, 0 overfull) | pdflatex unavailable here; proofread 21:45, CFP pass 22:40, prose pass 23:01, jargon/flow pass 23:13, second proofread 23:17, consistency audit 23:49, self-citation 23:55, hallucinator rerun 2026-09-06 00:08 |
| feat-015 | `scripts/build_artifact.sh artifact` | PASS (75 files, manifest verified) | last built 2026-09-06 00:09 (contents unchanged since 2026-09-05 23:17: legends relabeled and repositioned) |
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
- Phase 2: feat-017 blocked on D1 (70B route); GPUs 0 and 4 shared with other users (D2); the composition logs lack per-query spends (feat-021 re-runs with logging).

## Submission checklist for the human (from https://satml.org/call-for-papers/ and its checklist, Sep 4 2026 version)

- Sep 22 (abstract registration): title + abstract (tentative wording, no substantial change later), final authors and topics, ORCID for every author, Author Certification, mandatory conflicts, one author nominated as author-reviewer (may be asked to review up to three papers per submission), answer the "under review elsewhere" field, enter `N/A` in New Insights.
- Sep 29 (paper): `sub/satml/satml_2027.pdf`; anonymised repository link (e.g. anonymous.4open.science) containing `artifact.zip` contents plus `output.zip`; re-check conflicts in the last 24 h; optional LLM-processing opt-in flag; hallucinator self-check done 2026-09-05 (`pip install hallucinator` 0.2.2, Python API on the PDF): 49 references extracted, 36 verified by the tool, 13 reported not found. All 13 were then verified by hand: 10 arXiv IDs return the exact title and authors (Ippolito 2023, Shi 2024, Howard 2021, Waudby-Smith 2024, Chugg 2025, Zhou 2026, Maurer 2009, Ganguli 2022, Mouret 2015, Elkin-Koren 2024, whose FORC 2024 venue DBLP confirms with DOI 10.4230/LIPIcs.FORC.2024.3), ROUGE is in the ACL Anthology (W04-1013), and the Tsybakov and Polyanskiy-Wu books resolve through their CrossRef DOIs. No fabricated reference. Report: scratchpad hallucinator_report.txt (session-local); re-run on the final PDF if references change. Rerun 2026-09-06 00:08 after the self-citation: 42/50 verified automatically, 8 not_found all confirmed by hand (see progress.md).
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

- Human: answer D1/D2/D3 (plan v2 Section 9). Then feat-016 as before (abstract Sep 22 with the v2 abstract, paper Sep 29, artifact Oct 2).
- Agent, immediately and without any decision: feat-023 (budget-path feasibility, no GPU beyond forward passes), feat-022 (warped-anchor surprisals, GPU 1), feat-019 code + `tests/test_pathwise.py`. Then feat-019 runs (8B memoriser), feat-020, feat-021, feat-025, feat-024, feat-026, feat-027 in the order of `GOAL.md` Phase 2.
- Agent, once D1 is (a) or (b): feat-017 (download into `hf_cache/`, smoke on GPUs 1,2) → feat-018 (natural-memorisation audit + 70B composition, about 8 GPU-hours).
- Agent, committed in the paper's Ethical Considerations section: share the audit findings and code with the Anchored Decoding authors once the review outcome permits (human step).
