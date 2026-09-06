# GOAL: ship the SaTML 2027 submission "What does a KL budget certify?"

You are executing this goal autonomously in the repository `/mnt/md0/IITM/BackUp/Home/vijayavallabh/DA5001_Project`. Read `AGENTS.md` first (it is auto-imported by `CLAUDE.md`), then this file, then `/mnt/md0/IITM/BackUp/Home/vijayavallabh/sub/satml/IMPROVEMENT_PLAN.md` Sections 0–4. The plan is the spec; this file is the contract.

**Status (2026-09-06): GOAL COMPLETE — phase 1 and phase 2 both done.** Phase 1 closed 2026-09-05 (criteria 1–6; `progress.md` → GOAL COMPLETE, commit 7b6961e). Phase 2 closed 2026-09-06 19:30 under `IMPROVEMENT_PLAN.md` version 2 (the version-1 plan is kept as `IMPROVEMENT_PLAN_v1_2026-09-05.md`): feat-017 to feat-027 all `done` with evidence, D1 and D2 answered by the human, outcome recorded below. Nothing is running and no agent-executable feature is left; the remaining steps are human-only (feat-016).

## Phase 2 objective (added 2026-09-06)

Turn the complete phase-1 submission into a distinguished-paper-grade one by (i) auditing the mechanism with its own risky model, Llama 3.1 70B base, on the Harry Potter and 1984 passages that ship in `data/` and that Cooper et al. show it memorised, (ii) implementing and pricing two certificates that bound events rather than expectations (pathwise max-divergence anchored decoding with Propositions 1–2; the concentrated certificate of Proposition 3) plus a per-user odometer and a bank cap, (iii) predicting the recall transition from the anchor alone (budget-path feasibility, Proposition 4), and (iv) rewriting the manuscript and related work around the verified 127-entry bibliography (`LITERATURE_REVIEW.md`).

### Phase 2 success criteria

7. feat-019 to feat-023 and feat-026 to feat-027 are `done` with evidence; feat-018 is `done` if D1 permitted the 70B, otherwise `blocked` with the decision recorded; feat-024/025 done if time permits (P2).
8. `results/` additionally contains `pathwise_sweep.csv`, `pathwise_composition.csv`, `extraction_cost.csv`, `concentration.csv`, `odometer.csv`, `bank_cap.csv`, `burst_audit.csv`, `warped_anchor.csv`, `budget_path.csv` (and `natural_memorisation.csv`, `composition_70b.csv` if feat-018 ran); every new number in the manuscript traces to one of them, with k = −1 and k = 0 rows present.
9. The manuscript has the structure of plan v2 Section 7, ≤ 12 body pages, 0 `??`, 0 overfull, proofs of Propositions 1–4 in an appendix, and cites at least 90 verified references; the hallucinator self-check is rerun on the final PDF.
10. The abstract is frozen by Sep 21 (human registers Sep 22) and the artifact v2 is rebuilt and verified before Sep 29.

### Phase 2 outcome (2026-09-06 19:30)

7. Met, and more than required: feat-017 to feat-027 are all `done` with evidence in `feature_list.json`. D1 was answered "yes, the ungated mirror" (`unsloth/Meta-Llama-3.1-70B`, cached in `hf_cache/`), so feat-018 ran for real rather than being marked blocked; D2 allowed the 70B on GPUs 1+2 and 8B jobs on the shared 0/4. feat-024 and feat-025 both ran. feat-012 is superseded by feat-024; feat-010/011 stay optional and unstarted.
8. Met, under the names the scripts actually write: `pathwise_price.csv` (the pathwise/KL price on the same sweep) in place of `pathwise_sweep.csv`, `composition_8b_pathwise.csv` in place of `pathwise_composition.csv`, and `extraction_cost_{kl,pathwise,pathwise_lo}.csv` (+ `_windows`) in place of `extraction_cost.csv`; plus `concentration.csv` (+ `_summary`), `odometer.csv` (+ `_per_passage`), `bank_cap.csv`, `burst_audit.csv`, `warped_anchor.csv`, `budget_path.csv` (+ `_summary`), `natural_memorisation.csv`, `composition_70b.csv`, `composition_8b_kl.csv`, `prefix_debt_ablation.csv`, `latent_leakage_summary.csv`. Every manuscript number traces to one of them and the k = −1 / k = 0 rows are present.
9. Met: the manuscript follows plan v2 Section 7, the body ends on page 12 (18 pages with references and appendix), 0 `??`, 0 overfull, proofs of Propositions 1–4 in the appendix, 125 cited entries of 128 in `references.bib`. Reference check: hallucinator 0.2.2 on the compiled PDF (87 verified, 38 misses all confirmed by hand against arXiv/CrossRef/DBLP/ACL, none fabricated); reports in `~/sub/satml/bibcheck_2026-09-06/`. The cited set has not changed since that check.
10. Agent side met: the abstract in `satml_2027.tex` is final and artifact v2 is rebuilt and verified (`artifact/`, 179 files + `MANIFEST.sha256`; `artifact.zip` 23 MB). Registration, submission and the artifact upload are human-only (feat-016).

Headline numbers, one per phase-2 area: 70B single-query recall at He et al.'s book settings 0.018 (k=3) rising to 0.314 (k=20, = unconstrained), oracle composition 0.591 at k=20 · prefix debt off raises k=5 recall 0.09 → 0.40 (8B) · pathwise vs KL active steps 1.94% vs 0.26% at k=3 for ≤ 0.11 nats/token of utility · Freedman certificate P(L ≥ K+100) ≈ 0.01 for k ≥ 3 (p90 caps), empirical 0 and no trajectory above K · odometer B_user = 400 nats cuts off every user and caps 50-token-window reconstruction at 0.30 oracle / 0.26 chained · zero per-trajectory violations in > 100,000 new trajectories and queries.

### Phase 2 execution order

1. feat-023 (no GPU) and feat-022 (forward passes only) immediately; feat-019 code and tests in parallel.
2. On D1 = (a) or (b): feat-017 → feat-018 on GPUs 1 and 2 (bf16, TinyComma resident); the 70B download goes to `hf_cache/` (gitignored), never the home filesystem.
3. feat-019 runs (8B) → feat-020 → feat-021 → feat-025 → feat-024.
4. feat-026 → feat-027. Freeze the abstract by Sep 21 with whatever numbers exist by Sep 20 (plan v2 Section 6).

### Phase 2 additional constraints

- Ask before any 70B download (D1) and before using GPUs 0 or 4 while other users' jobs run on them (D2).
- The pathwise decoder's violation rule: R_T ≤ max{0, k T_max − δ_init} + 1e-3 per trajectory, in addition to the KL rules.
- 8-bit or 4-bit 70B loads may be used for smoke tests only; every reported 70B number is bf16.

## Objective

Turn the arXiv paper "An Empirical Audit of k-NAF Budget Accounting for Anchored Decoding" into a SaTML 2027 submission whose central result is an audit that can fail: an empirical, security-framed measurement of what Anchored Decoding's KL budget K = k·T_max actually certifies, plus at least one adaptive attack that reconstructs copyrighted text while every query stays within budget.

## Success criteria (all must hold)

1. `feature_list.json` features feat-001 through feat-009 and feat-013 through feat-015 are `done`, each with its evidence command and output pasted in. feat-010, feat-011, feat-012 are optional and attempted only after feat-009 is done.
2. `results/` contains, at minimum: `regime_table.csv`, `regime_sweep.csv`, `certificate_caps.csv`, `llr_tails.csv`, `memorizing_model_recall.csv`, `composition.csv`. Every number in the manuscript traces to one of these files.
3. `~/sub/satml/satml_2027.tex` compiles with `pdflatex` + `bibtex` (or with `tectonic`, the accepted equivalent on this box, where pdflatex is unavailable; see `progress.md` Decisions), has ≤ 12 pages of body text in IEEEtran, contains a threat-model section and, in this order before the references, an Open Science section, an `LLM usage considerations` section with the CFP's required editorial-use sentence, and an Ethical Considerations section, and `references.bib` has ≥ 40 complete, verified entries.
4. `artifact/` exists with an anonymised code snapshot, `results/*.csv`, prompt sets, the memorising-model recipe, a README with exact reproduction commands, and a verified `MANIFEST.sha256`.
5. `./init.sh` passes and the git tree is committed after every completed feature.
6. `session-handoff.md` names the final PDF path and the artifact zip path for the human to submit (feat-016).

## Non-goals

- Do not run the old ρ / U_EBB / surrogate / k-DPP search. Do not add new tables about the Bernstein proxy.
- Do not modify `~/sub/neurips_2026.tex`, `output.zip`, or `data/`.
- Do not register the abstract, submit the paper, or upload to Zenodo. Those are human-only (feat-016).
- Do not push to any remote.

## Constraints

- Dates (AoE): abstract Sep 22 2026, paper Sep 29 2026, anonymised artifact repository frozen Oct 2 2026 (full timeline in `AGENTS.md`). Plan work so that a compilable draft with feat-005, feat-006, feat-007, feat-009 results exists by Sep 22, leaving the last week for writing and packaging.
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

Since 2026-09-06 there is no eligible feature left: `./init.sh` plus the manuscript check in `AGENTS.md` is the whole of a verification session. Do not start feat-016 (human-only), and treat feat-010/011 as optional polish that must not disturb the frozen manuscript or artifact.
