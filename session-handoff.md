# Session Handoff

## Current Objective

- Goal: phase 1 COMPLETE (2026-09-05, commit 7b6961e). **Phase 2 COMPLETE 2026-09-06 19:30: feat-017..027 all done** under `~/sub/satml/IMPROVEMENT_PLAN.md` v2. Manuscript v2 final (12 body pages), artifact v2 built. Only the human-only feat-016 steps remain. Human decisions taken 2026-09-06: D1 = `unsloth/Meta-Llama-3.1-70B` mirror (cached in `hf_cache/`), D2 = 70B on GPUs 1+2, 8B jobs on shared GPUs 0/4 (PCI order).
- Every paper number is in `results/*.csv` with its producing command in `progress.md` (entries 2026-09-06 02:20 → 19:30). New this phase: `natural_memorisation.csv`, `composition_70b.csv`, `composition_8b_{kl,pathwise}.csv`, `odometer.csv`, `bank_cap.csv`, `concentration_summary.csv`, `pathwise_price.csv`, `extraction_cost_{kl,pathwise,pathwise_lo}.csv`, `prefix_debt_ablation.csv`, `concentration.csv`, `warped_anchor.csv`, `budget_path.csv`, `latent_leakage_summary.csv`, `burst_audit.csv`; figures `natural_memorisation.pdf`, `budget_path.pdf`.
- Manuscript (`/mnt/md0/IITM/BackUp/Home/vijayavallabh/sub/satml/satml_2027.tex`): **v2 complete, audited and copy-edited (feat-026 plus the 2026-09-06/07 passes): body ends on page 12, 18 pages with references and appendix, 0 overfull, 0 ??, four tables and three figures**; checkpoint `satml_2027_phase2_2026-09-06.pdf` (phase-1 PDF: `satml_2027_phase1_2026-09-06.pdf`). Bibliography: 128 entries, 125 cited, all verified (`bibcheck_2026-09-06/`). Section backups: `related_work_v2_2026-09-06.tex`, `intro_v1_2026-09-05.tex`. Artifact v2 built (`artifact/`, `artifact.zip` 23 MB, 179 files, manifest verified; `hf_cache/` excluded).
- Nothing is running. The same-sweep KL runs (`output/phase2/kl_sweep_hi` k=3, `kl_sweep_k5`, `kl_sweep_k10`, `kl_sweep_k20`) and the low-budget retry run (`retries_pathwise_lo`) finished on 2026-09-06 and their numbers are in `results/pathwise_price.csv`, `results/concentration_summary.csv`, `results/extraction_cost_pathwise_lo.csv` and in the manuscript.
- Next (human, feat-016): read `sub/satml/satml_2027_phase2_2026-09-06.pdf`; abstract registration by Sep 22 (the abstract in `satml_2027.tex` is final), paper by Sep 29, anonymised artifact repository (upload `artifact.zip`, 179 files, manifest inside) by Oct 2. Before any further edit: recompile and re-check that the body still ends on page 12 (`pypdf`: page 13 must start with 'Open Science'). Optional polish if time allows: a fresh-eyes proofread of Sections VI-E, VI-F and VII (written 2026-09-06), and a final hallucinator run (the cited set of 125 entries is unchanged since the 02:50 verification).
- **For the human (feat-016, unchanged):** registration Sep 22, paper Sep 29, artifact repo Oct 2.
- Branch / commit: `master`; latest work: this commit. The compute statement was recomputed from the run directories on 2026-09-07 (`analysis/compute_hours.py` -> `results/compute_hours.csv`): 45.9 GPU-hours for the 8B jobs, 8.3 for the 70B, 75 minutes for the fine-tune; the paper now says 45 / 8 / 75. No open items remain for the agent. Harness and docs were refreshed on 2026-09-06 23:19; while cross-checking them against `results/` two rounded numbers in `sections/certificates.tex` were corrected (pathwise price 0.11 → 0.15 nats/token "throughout"; four Freedman δ values rounded up so the stated upper bounds hold). Recompiled and re-verified: 18 pages, body ends page 12, 0 overfull, 0 ??; `satml_2027_phase2_2026-09-06.pdf` refreshed 23:31. See the last entry of `progress.md`.
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

Phase 2 (2026-09-06):

- [x] feat-017/018: 70B base downloaded to `hf_cache/` (D1 = `unsloth/Meta-Llama-3.1-70B`) and audited on the Harry Potter passages at He et al.'s book settings → `results/natural_memorisation.csv`, `composition_70b.csv`, `figures/natural_memorisation.pdf`.
- [x] feat-019/020: pathwise (Δmax) decoder in `a_patch/{pathwise,factory}.py` with per-step `r_t`/`m_t`/variance logging; `results/composition_8b_pathwise.csv`, `pathwise_price.csv`, `concentration.csv` (+ `_summary`), `extraction_cost_*.csv`.
- [x] feat-021: per-query spend logs, odometer replay and bank cap (`a_patch/bank.py`, `analysis/{odometer,check_bank_cap,burst_audit}.py`) → `odometer.csv`, `bank_cap.csv`, `burst_audit.csv`.
- [x] feat-022/023/024/025: warped anchor, budget-path feasibility, latent leakage on public-domain texts, prefix-debt ablation → `warped_anchor.csv`, `budget_path.csv`, `latent_leakage_summary.csv`, `prefix_debt_ablation.csv`.
- [x] feat-026/027: manuscript v2 restructured and compressed to 12 body pages; figures regenerated; artifact v2 rebuilt.
- [x] Decoder guard removed so temperature and repetition penalty may be used under a budget (App. B of He et al.); smoke `output/phase2/warp_smoke`, 0 violations.
- [x] Harness and docs refreshed at the close of phase 2 (2026-09-06 23:19): `AGENTS.md`, `GOAL.md`, `progress.md`, this file, `README.md`, `README_artifact.md`, `.gitignore`, `init.sh`, `feature_list.json`; artifact rebuilt so its README copy and manifest match.
- [x] Two manuscript numbers corrected against `results/` (2026-09-06 23:30, `sections/certificates.tex`); PDF and checkpoint recompiled and re-verified.
- [x] GPU-hours recomputed from the run directories (2026-09-07, `analysis/compute_hours.py`); compute statement corrected to 45 GPU-hours (8B) / 8 (70B) / 75 minutes (fine-tune).
- [x] Editorial pass (2026-09-07): Section VII reordered into three repairs then two caveats, Section VI opener and VI-C title fixed, all three figures rebuilt for one column (Fig. 3 redrawn from scratch), the duplicate composition table dropped, near-verbatim recall defined at first use, Proposition 2's notation aligned with the body. Body still ends on page 12.
- [x] Full numerical audit of the manuscript against `results/` and the raw logs (2026-09-06 23:55): tables all match; ten further corrections applied (query counts 16,200 -> 8,514 and 3,986 -> 3,736, 758 -> 608 passages, utilisation metric, prefix-debt arithmetic, penalty range, odometer cut fraction, activity scoping, single-query qualifier, and the missing `max{0,.}` in Proposition 1). Details in the last entry of `progress.md`.

## Verification Evidence

| Check | Command | Result | Notes |
|---|---|---|---|
| Baseline | `./init.sh` | PASS (exit 0, last run 2026-09-06 23:19) | 30 tests inside |
| Imports | `.venv/bin/python -c "import a_patch, dap.shared"` | PASS | covered by init.sh step 2 |
| GPU | `.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` | True, 5 local GPUs | 4×A100 usable (`nvidia-smi` 0,1,2,4); index 3 is a 4 GB T400. The DGX is unreachable from this account |
| feat-002 | `.venv/bin/python analysis/reanalyze_logs.py --logs output --out results` | PASS (0.195% active at k=3, 96/999 L>K at k=1, 0 violations) | 6.8 s, no GPU |
| feat-003 | `.venv/bin/python -m pytest -q tests && ./init.sh` | PASS (30 passed, exit 0) | 6 tests at feat-003; the suite grew to 30 in phase 2 |
| feat-006 | `CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 .venv/bin/python analysis/certificate_cap.py --data data --out results` | PASS (exit 0) | ~2 min, one A100 |
| feat-004 | `h1.py --k-values -1 0 0.15 1 ... --output-dir output/smoke` (see feature_list.json) | PASS (exit 0, 24 files) | local GPUs 1+2 |
| feat-005 | `analysis/regime_sweep.py --run plain=... --run chat=... --out results` | PASS (98 rows, 0 violations) | 37,800 trajectories |
| feat-007 | `pytest -q tests/test_cs.py && cat results/llr_tails.csv && grep -rn ebb_upper_bound_chapman dap \| wc -l` | PASS (3 tests, 79 rows, 0 call sites) | |
| feat-009 | `analysis/composition_attack.py ... --k-values -1 0 0.15 0.5 1 3 5 10 20` | PASS (45 summary rows, 0 violations) | 70 min |
| feat-013 | `figures/make_figures.py --copy-to <sub/satml/figures>` | PASS (4 figures) | |
| feat-014 | `tectonic -X compile satml_2027.tex` | PASS (11 pages, 50 refs cited of 53, 0 `??`, 0 overfull) | pdflatex unavailable here; proofread 21:45, CFP pass 22:40, prose pass 23:01, jargon/flow pass 23:13, second proofread 23:17, consistency audit 23:49, self-citation 23:55, hallucinator rerun 2026-09-06 00:08 |
| feat-015 | `scripts/build_artifact.sh artifact` | PASS (75 files, manifest verified) | superseded by the v2 build below |
| feat-017/018 | `scripts/run_natural_memorisation.sh 16` then `analysis/natural_memorisation.py --runs output/phase2/nm --out results --figures figures` | PASS (0 violations; single/oracle recall 0.018/0.015 at k=3 → 0.314/0.591 at k=20) | 70B bf16 on GPUs 1+2, ~8 GPU-hours |
| feat-019/020 | `h1.py --constraint pathwise ...` + `analysis/{pathwise_price,concentration}.py` | PASS (active steps 1.94% pathwise vs 0.26% KL at k=3; δ(100) ≈ 0.01 for k ≥ 3, empirical 0) | same sweep at every k |
| feat-021 | `analysis/odometer.py` + `scripts/run_bank_cap.sh` + `analysis/check_bank_cap.py` | PASS (B_user = 400 nats cuts every user, 50-token-window recall 0.30 oracle / 0.26 chained; bank cap respected) | per-query logs |
| feat-026 | `tectonic -X compile satml_2027.tex` | PASS (18 pages, body ends page 12, 0 overfull, 0 `??`, 125 cited of 128) | checkpoint `satml_2027_phase2_2026-09-06.pdf` |
| feat-027 | `scripts/build_artifact.sh artifact` | PASS (179 files + manifest, verified; `artifact.zip` 23 MB) | `hf_cache/` excluded |
| Harness score | `node /home/sports/.agents/skills/harness-creator/scripts/validate-harness.mjs --target .` | 100/100 | structural score only |

## Files Changed

- `AGENTS.md`, `CLAUDE.md`, `feature_list.json`, `progress.md`, `session-handoff.md`, `init.sh`, `GOAL.md`, `analysis/reanalyze_logs.py`, `results/*.csv`
- feat-003: `dap/stats.py`, `dap/shared.py`, `dap/e1.py`, `dap/e2/{evaluator,runner,types}.py`, `a_patch/factory.py`, `tests/`
- feat-004..009: `analysis/{certificate_cap,regime_sweep,llr_tails,composition_attack,bank_burst,memorizing_recall}.py`, `recipes/`, `scripts/run_*.sh`, `results/*.csv`
- feat-013..015 and after: `figures/make_figures.py`, `figures/*.pdf`, `scripts/build_artifact.sh`, `README_artifact.md`, `artifact/`, `.gitignore`; manuscript `/mnt/md0/IITM/BackUp/Home/vijayavallabh/sub/satml/{satml_2027.tex,sections/*.tex,references.bib,figures/}` (outside this repo)
- feat-017..027 (phase 2): `a_patch/{factory,pathwise,bank,warp}.py`, `h1.py` (`--constraint`, `--no-prefix-debt`, device-map pass-through), `analysis/{composition_attack,natural_memorisation,odometer,concentration,pathwise_price,extraction_cost,budget_path,warped_anchor,latent_leakage,check_bank_cap,recheck_violations,burst_audit}.py`, `scripts/{download_70b,run_natural_memorisation.sh,run_bank_cap.sh,rerun_nm_hp1_B.sh,build_artifact.sh}`, `tests/test_{pathwise,bank_cap,warp,budget_path,composition_helpers,latent_leakage}.py`, `results/*.csv`, `figures/`, `artifact/`; manuscript `sections/{natural,prefixdebt,certificates,discussion,intro,conclusion,related_work,open_science,...}.tex` and `sections/appendix_theory.tex`
- Docs refresh (2026-09-06 23:19): `AGENTS.md`, `GOAL.md`, `progress.md`, `session-handoff.md`, `README.md`, `README_artifact.md`, `.gitignore`

## Decisions Made

- See `progress.md` → Decisions Made (retire EBB/ρ; audit at small k; attacks over search).

## Blockers / Risks

*(Refreshed 2026-09-06 23:19.)*

- Human-only steps (feat-016) are the only blocking work left: abstract Sep 22, paper Sep 29, artifact repository Oct 2.
- `HF_TOKEN` in `.env` is still invalid (401); run everything with `HF_HUB_OFFLINE=1`. The cache now includes the 70B base (`hf_cache/models--unsloth--Meta-Llama-3.1-70B`, 132 GB total cache), so no download is needed to reproduce phase 2.
- The DGX is still unreachable from this account and there is still no `pdflatex` (tectonic is the sanctioned substitute).
- feat-010 (bank-and-burst) remains untested: the LoRA memoriser ignores filler instructions. The manuscript says so. The cached 70B base is a candidate risky model if anyone wants to revisit it — but it would change a frozen paper, so ask first.
- feat-011 (format evasion) and feat-012 (superseded by feat-024) are not planned.
- Resolved in phase 2: D1 (70B mirror downloaded), D2 (GPUs 1+2 for the 70B, shared 0/4 for 8B jobs), the missing per-query spends (feat-021 logs them), and the decoder's refusal to warp under a budget.

## Submission checklist for the human (from https://satml.org/call-for-papers/ and its checklist, Sep 4 2026 version)

- Sep 22 (abstract registration): title + abstract (tentative wording, no substantial change later), final authors and topics, ORCID for every author, Author Certification, mandatory conflicts, one author nominated as author-reviewer (may be asked to review up to three papers per submission), answer the "under review elsewhere" field, enter `N/A` in New Insights.
- Sep 29 (paper): `sub/satml/satml_2027.pdf`; anonymised repository link (e.g. anonymous.4open.science) containing `artifact.zip` contents plus `output.zip`; re-check conflicts in the last 24 h; optional LLM-processing opt-in flag; hallucinator self-check done 2026-09-05 (`pip install hallucinator` 0.2.2, Python API on the PDF): 49 references extracted, 36 verified by the tool, 13 reported not found. All 13 were then verified by hand: 10 arXiv IDs return the exact title and authors (Ippolito 2023, Shi 2024, Howard 2021, Waudby-Smith 2024, Chugg 2025, Zhou 2026, Maurer 2009, Ganguli 2022, Mouret 2015, Elkin-Koren 2024, whose FORC 2024 venue DBLP confirms with DOI 10.4230/LIPIcs.FORC.2024.3), ROUGE is in the ACL Anthology (W04-1013), and the Tsybakov and Polyanskiy-Wu books resolve through their CrossRef DOIs. No fabricated reference. Report: scratchpad hallucinator_report.txt (session-local); re-run on the final PDF if references change. Rerun 2026-09-06 00:08 after the self-citation: 42/50 verified automatically, 8 not_found all confirmed by hand (see progress.md).
- Confirmed by the author on 2026-09-05: the NeurIPS/arXiv version (`sub/neurips_2026.tex`, arXiv 2605.28001) received no reviews at any venue, so nothing needs appending, and the paper is not under submission elsewhere (answer "no" to the HotCRP under-review field).
- Oct 2: last edit of the anonymised repository; it must then stay accessible and unchanged through review.
- Paper end matter is already in the CFP order (Open Science → LLM usage considerations → Ethical Considerations → references) and contains the required editorial-use sentence. The earlier arXiv audit *is* cited, in the third person as `\cite{vijayavallabh2026audit}` (see Resolved Decision above), which the CFP permits.
- If accepted: Zenodo by Jan 14 2027 (paste the DOI into `sections/open_science.tex`), camera-ready mid-Feb 2027, in-person presentation early May 2027 with one full registration.

## Next Session Startup

1. Read `AGENTS.md` (auto-imported by `CLAUDE.md`).
2. Read `feature_list.json` and `progress.md`.
3. Review this handoff.
4. Run `./init.sh` before editing.

## Recommended Next Step

*(Refreshed 2026-09-06 23:19. D1 and D2 are answered; D3 fell away because the 70B numbers arrived on time.)*

- **Human (feat-016), the only work that remains:** read `sub/satml/satml_2027_phase2_2026-09-06.pdf`; register title and abstract by Sep 22 (the abstract in `satml_2027.tex` is final); submit the paper by Sep 29; put the contents of `artifact.zip` plus `output.zip` in an anonymised repository by Oct 2 and freeze it.
- **Agent, if asked to do anything at all:** verify rather than extend — `./init.sh` (30 tests), then the manuscript check in `AGENTS.md` (0 `??`, 0 overfull, page 13 starts with "Open Science"). Optional polish is listed in `progress.md` → What's Next; anything that changes a number means recompiling the paper and rebuilding the artifact.
- Committed in the paper's Ethical Considerations section: share the audit findings and code with the Anchored Decoding authors once the review outcome permits (human step).
