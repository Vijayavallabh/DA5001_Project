# Session Handoff

## Current Objective

- Goal: phase 1 (ship the SaTML 2027 submission) is COMPLETE (2026-09-05; commit 7b6961e). **Phase 2 in execution since 2026-09-06 00:45** under `~/sub/satml/IMPROVEMENT_PLAN.md` v2 (feat-017..027; `GOAL.md` Phase 2). Decisions taken by the human on 2026-09-06: D1 = ungated 70B mirror `unsloth/Meta-Llama-3.1-70B` (cached in `hf_cache/`), D2 = 70B on GPUs 1+2, 8B jobs on shared GPUs 0/4 (PCI order).
- Done as of 03:10: feat-019 code + tests + smoke (pathwise decoder, R-based violation semantics), feat-022 (`results/warped_anchor.csv`, `sec:warped`), feat-023 (`results/budget_path.csv`, figure, paragraph in attack_results), feat-024 (`results/latent_leakage_summary.csv`, `sec:leakage`), burst audit (`results/burst_audit.csv`), bank cap (`a_patch/bank.py`, `--bank-cap`, log-level check), KL composition rerun with per-query logs (`results/composition_8b_kl.csv`) and the odometer replay (`results/odometer.csv`, `sec:odometer` with Table tab:odometer), decoder guard removed so He et al.'s decoding settings run under a budget (AGENTS.md), bibliography check closed (125 cited entries confirmed; `sub/satml/bibcheck_2026-09-06/`), manuscript: Background "three readings", Threat model C7–C10, Protocol C7–C10, Discussion certificate card, theory appendix, Related Work v2 (compiles, 0 overfull, 9 forward `??` to sec:pathwise/concentration/natural/prefixdebt).
- Running when this was written (check `output/phase2/*/run.log`, `output/phase2/nm_runs.log`, `output/phase2/nm_hp1_B_rerun.log`, `output/phase2/bank_cap_runs.log`): GPU 0 → `pathwise_sweep` (h1.py) then `scripts/run_bank_cap.sh` (k=10 caps 10/50, k=20 cap 20); GPU 4 → `comp8b_pathwise` (finishing) → `kl_sweep_conc` → `prefix_ablation` → `retries_kl` → `retries_pathwise`; GPUs 1+2 → `scripts/run_natural_memorisation.sh` (hp1_A, 1984_*, hp1_B_pathwise, hp1_B_nodebt) then `scripts/rerun_nm_hp1_B.sh`. The 8B control on HP1 is done (`output/phase2/hp1_8b`: recall 0.000 at k=-1/0; the 70B base gets 0.558/0.771/0.527 greedy).
- Next analyses once logs exist: `analysis/recheck_violations.py --queries output/phase2/comp8b_pathwise/queries.jsonl --constraint pathwise` and `analysis/pathwise_price.py --kl output/sweep_plain --pathwise output/phase2/pathwise_sweep` (feat-019 → `sec:pathwise`); `analysis/concentration.py --logs output/phase2/kl_sweep_conc --out results` (feat-020 → `sec:concentration`); `analysis/extraction_cost.py --queries output/phase2/retries_*/queries.jsonl` (Prop. 2); prefix-debt ablation from `output/phase2/prefix_ablation` and `nm/hp1_B_nodebt` (feat-025 → `sec:prefixdebt`); `analysis/natural_memorisation.py --runs output/phase2/nm --out results --figures figures` (feat-018 → `sec:natural`); bank-cap paragraph in `sec:odometer` from `output/phase2/bank_cap_k*`; then feat-026 (restructure to ≤12 body pages, abstract/intro/discussion/limitations/open science/LLM usage rewrite, humanizer pass, hallucinator on the final PDF) and feat-027 (figures + artifact v2). Abstract must be frozen by Sep 21 for the human's Sep 22 registration.
- **For the human (feat-016, unchanged):** phase-1 PDF preserved as `sub/satml/satml_2027_phase1_2026-09-06.pdf`; the working PDF is mid-restructure (17 pages with forward references) and is not yet a submission candidate.
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
