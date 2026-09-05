# DA5001 Project — SaTML 2027 audit of Anchored Decoding

Code for arXiv 2605.28001, now being reworked into a SaTML 2027 submission: **"What does a KL budget certify? An adversarial audit of inference-time near-access-freeness."** The mechanism under audit is He et al.'s Anchored Decoding (`a_patch/`, KL budget K = k·T_max). The full rationale and experiment design is `~/sub/satml/IMPROVEMENT_PLAN.md` (absolute: `/mnt/md0/IITM/BackUp/Home/vijayavallabh/sub/satml/IMPROVEMENT_PLAN.md`; **version 2 since 2026-09-06**, with the literature in `LITERATURE_REVIEW.md` next to it and the version-1 plan kept as `IMPROVEMENT_PLAN_v1_2026-09-05.md`). The manuscript is `~/sub/satml/satml_2027.tex`.

**Deadlines (AoE, from https://satml.org/call-for-papers/ as of Sep 4 2026):** abstract registration Sep 22 2026 (fixed authors and topics, ORCIDs, author certification, conflicts, author-reviewer nomination) · paper Sep 29 2026 · anonymised artifact repository (e.g. anonymous.4open.science, linked in HotCRP, then frozen) Oct 2 2026 · early reject Nov 4 · discussion Nov 25–Dec 9 · decision Dec 16 · Zenodo artifacts Jan 14 2027 · revisions Jan 21 · camera-ready mid-Feb 2027 · conference early May 2027, in-person presentation required. Human-only steps are `feat-016`; never attempt them.

**Status (2026-09-06):** phase 1 GOAL COMPLETE (feat-001..009, 013..015 done; PDF 11 pages and `artifact.zip` named in `session-handoff.md`). **Phase 2 in progress** (`GOAL.md` → Phase 2; plan v2 Sections 2–7): feat-017..027 added; feat-017 (70B) is blocked on the human's decision D1; feat-012 is superseded by feat-024; feat-010/011 remain optional.

## Startup Workflow

Before writing code:

1. `pwd` must be this repo root.
2. Read this file completely, then `~/sub/satml/IMPROVEMENT_PLAN.md` Sections 0–7 (version 2; it is the spec; do not re-derive it) and, before touching Related Work, `~/sub/satml/LITERATURE_REVIEW.md`.
3. Run `./init.sh`. If it fails, repair that first before adding scope.
4. Read `feature_list.json` and `progress.md`; pick the lowest-numbered feature whose dependencies are `done` and whose status is `not-started` or `in-progress`.
5. `git log --oneline -5`.

## Working Rules

- **One feature at a time.** Exactly one feature `in-progress` at any moment.
- **Stay in scope.** Touch only files the feature needs. Log unrelated bugs in `progress.md` under Blockers/Risks; do not fix them.
- **Verification required.** Nothing is done without running its evidence command and pasting the output into `feature_list.json`.
- **Baselines are mandatory.** Every experiment that reports a copying or spend metric at some k also reports k = −1 (risky model only) and k = 0 (safe model only) on the same prompts and seeds.
- **Per-trajectory, not per-mean.** A "budget violation" is only a violation if a single trajectory has Z > max(0, B) + 1e-3 or a step has a_t > k_t + 1e-3. Do not reintroduce the empirical-Bernstein proxy, ρ = U_EBB/B_eff, the surrogate ensemble, or the k-DPP archive. If a mean estimate is genuinely needed, use an anytime-valid confidence sequence (`feat-007`).
- **Known truths about He et al. (arXiv 2602.07120), verified in the PDF text on 2026-09-06:** temperature and repetition penalty are applied to both logit vectors *before* the KL solve (App. B), their book experiments use temperature 0.7 and penalty 1.1 (App. D.1), App. E.1 says that at k = 3.0 "the constraint is rarely binding and p* ≈ p_r", and Table 17 (TinyComma 1.8B + Llama 3.1 70B) gives the unconstrained 70B ROUGE-L ≥ τ on 23.0% of CopyBench prompts with word-LCS 10.7 (anchor alone: 0%, 1.7). `data/copybench_test.jsonl` holds 50 passages of *Harry Potter and the Sorcerer's Stone* (Cooper et al. 2025: 96.3% extraction coverage for Llama 3.1 70B) and `data/copybench_attack_train.jsonl` holds 8 of *1984*; the LoRA memoriser never saw the test split. Our decoder also warps before the solve (`a_patch/factory.py` lines 791–793) and refuses greedy decoding under a budget (line 653), so all released-log and attack numbers are at temperature 1 with no penalty.
- **Known truths from the released logs** (do not spend compute re-establishing them; cite `results/` once `feat-002` is done): constraint active in 0.19% of steps at k=3 and 6.3% at k=1; 57% of generations identical between k=3 and k=5; zero invariant violations in 26,999 trajectories (all 18 released files, 5.21M decode steps); 96/999 attack_train trajectories (738/8,999 across all six classes) exceed K in realised log-likelihood ratio at k=1; every N=20 held-out row in Stage 2 contains one duplicated sample (seed collision, `feat-003`); the anchor writes the first floor(δ_init/k) tokens of every answer (δ_init ≈ 6 nats). All reproduced in `results/` by `feat-002`.
- **Compute.** Local box: 4×A100 80GB (`nvidia-smi` indices 0,1,2,4; GPU 3 is a 4 GB T400, never use it). On 2026-09-06 00:20 GPUs 1 and 2 were idle and 0 and 4 were at 100% utilisation by other users (~9 GB each): use 1 and 2; use 0/4 only if the human allows (D2). Llama 3.1 70B base fits in bf16 across GPUs 1+2 (141 GB weights) with TinyComma resident; report only bf16 numbers. The DGX SSH key is not installed for this account (recorded 2026-09-05); `dgx-gpu` is not a fallback unless that changes. Set `CUDA_VISIBLE_DEVICES` explicitly for every job, **always together with `CUDA_DEVICE_ORDER=PCI_BUS_ID`**: without it CUDA numbers the A100s 0–3 and the T400 4, so `CUDA_VISIBLE_DEVICES=4` lands on the 4 GB card (this cost one failed run on 2026-09-06); with PCI order the indices match `nvidia-smi`. Large model caches go under `hf_cache/` in this repo (gitignored), never the home filesystem (366 GB free) ; `/mnt/md0` has 3.5 TB free.
- **Secrets.** `HF_TOKEN` lives in `.env` (gitignored) and still returned HTTP 401 on 2026-09-06; run local jobs with `HF_HUB_OFFLINE=1` (cache has Llama-3.1-8B-Instruct, Llama-3.1-8B, TinyComma, Qwen2.5-7B-Instruct; 70B is not cached). `meta-llama/*` is gated: ask the user for a valid token before any download. Ungated mirrors of the 70B base exist (`unsloth/Meta-Llama-3.1-70B`, `NousResearch/Meta-Llama-3.1-70B`, both HTTP 200 on 2026-09-06); downloading one is decision D1 in plan v2 Section 9 and needs the human's yes.
- **Do not modify** `~/sub/neurips_2026.tex`, `output.zip`, or anything under `data/`. New logs go to `output/` (gitignored); summaries go to `results/` (committed, small).
- **Anonymity.** Nothing you write into `~/sub/satml/` or the artifact may identify the authors. The one sanctioned exception is the third-person citation of the earlier audit, `\cite{vijayavallabh2026audit}` (arXiv 2605.28001, added 2026-09-05 at the user's request): refer to it as "an earlier audit", never "our earlier audit".
- **CFP format rules.** `\documentclass[conference]{IEEEtran}`, 10pt, unmodified geometry; ≤ 12 pages of body text; end matter in this order, none of it counted: `Open Science` (what is released, or why not) → `LLM usage considerations` (exact title; must contain the sentence "LLMs were used for editorial purposes in this manuscript, and all outputs were inspected by the authors to ensure accuracy and originality." plus compute justification) → `Ethical Considerations` (optional) → references. Anonymity: no names or affiliations, own prior work in the third person only (no "our earlier audit"), no hint that artifacts are already public, linked material anonymised. Non-existent references are grounds for desk rejection: verify every entry.
- **Paper integrity.** The manuscript (proofread 2026-09-05) reports bank-and-burst as attempted but not evaluated (the fine-tuned memoriser ignores filler instructions) and claims no adaptive prompt search; do not describe feat-010/011/012 as run until their evidence exists. **Provenance:** the 26,999-trajectory logs in `output.zip` are from the earlier arXiv 2605.28001 evaluation (Llama-3.1-8B-Instruct + TinyComma), not from He et al., whose paper evaluates TinyComma with Llama 3.1 70B base and sweeps k from 0.1 to 20; the manuscript must say "released runs/logs", never "the mechanism's authors released" or "the model they evaluated". After any edit under `~/sub/satml/`, recompile and check the PDF for `??` and the log for overfull boxes (see Verification Commands).

## Key Facts

| Item | Value |
|---|---|
| Entry points | `h1.py` → `dap/e1.py` (fixed workload); `h2.py` → `dap/e2/runner.py` (search) |
| Core library | `a_patch/factory.py` (`AnchoredDecodingFactory`; `k_radius=-1.0` = risky only, `0.0` = safe only) |
| Stats | `dap/stats.py`: index-only `build_trajectory_seeds`, copying metrics, `budget_check` (per-trajectory max spend / utilisation / invariant), `anytime_valid_cs` (PrPl-EB). The empirical-Bernstein proxy is gone; E2 `rho` now means max per-trajectory utilisation and `certified` means every trajectory within budget |
| Models | safe `jacquelinehe/tinycomma-1.8b-llama3-tokenizer`, risky `meta-llama/Llama-3.1-8B-Instruct`, optimizer `Qwen/Qwen2.5-7B-Instruct` |
| Released logs | `output.zip` (2.4 GB; `unzip -o output.zip 'output/h1_outputs/trajectories_k3_attack_train.jsonl'` etc.) |
| Log reanalysis | `analysis/reanalyze_logs.py --logs output --out results` (feat-002; writes `results/regime_table.csv`, `llr_tails.csv`, `prefix_debt_forced_tokens.csv`, `surprisal.csv`, `seed_collisions.csv`, `per_trajectory.csv`) |
| Paper, bib, figures | `/mnt/md0/IITM/BackUp/Home/vijayavallabh/sub/satml/satml_2027.tex` + `sections/*.tex` (SaTML version; the arXiv version is `satml_2027_arxiv_v1.tex`), `references.bib` (127 entries: 53 original + 74 added 2026-09-06, all verified; the additions are also in `bib_additions_2026-09-06.bib` and explained in `LITERATURE_REVIEW.md`), `figures/` (copied by `figures/make_figures.py --copy-to <that dir>`); compile with `~/.local/bin/tectonic -X compile satml_2027.tex` (no pdflatex on this box). `~` is `/home/sports` here, not the project home: use absolute paths |
| Tests | `tests/` (14 tests: seeds, invariant, chat template, metrics, confidence sequence, E2 stats; feat-019 adds `tests/test_pathwise.py`); `./init.sh` runs them |

## Verification Commands

```bash
./init.sh                      # full baseline verification (fails fast)
.venv/bin/python -m pytest -q tests   # 14 tests
.venv/bin/python h1.py --k-values 1.0 --trajectories-per-prompt 2 \
  --cap-neutral 2 --cap-val 2 --cap-test 2 --cap-attack-train 2 --cap-factual 2 --cap-creative 2 \
  --output-dir output/smoke   # GPU smoke test for E1 (run on DGX)
```

```bash
# manuscript: compile, then confirm no unresolved references, no overfull boxes, <= 12 pages
cd /mnt/md0/IITM/BackUp/Home/vijayavallabh/sub/satml && ~/.local/bin/tectonic -X compile satml_2027.tex 2>&1 | grep -ci overfull; \
  pdftotext satml_2027.pdf - | grep -c '??'; /mnt/md0/IITM/BackUp/Home/vijayavallabh/DA5001_Project/.venv/bin/python -c "from pypdf import PdfReader; print(len(PdfReader('satml_2027.pdf').pages))"
```

Per-feature evidence commands are in `feature_list.json`.

## Definition of Done

A feature is done only when ALL of the following are true:

- [ ] The behaviour in its `description` exists and its evidence command ran; the command and its output are pasted into `evidence`.
- [ ] `./init.sh` passes.
- [ ] Any new number that will appear in the paper is written to `results/*.csv` with the producing command in `progress.md`.
- [ ] Repository remains restartable: a fresh session can run `./init.sh` immediately.

## End of Session

Before ending a session:

1. Update `progress.md` (Current State, What's Done, What's Next, Blockers).
2. Update `feature_list.json` status and evidence.
3. Fill `session-handoff.md` (Current Objective, Files Changed, Recommended Next Step).
4. Commit with a descriptive message; leave the tree clean and restartable.

## Escalation

- **GPU job fails or DGX unreachable:** record in `progress.md`, move to a feature that needs no GPU (`feat-002`, `feat-013` prep, `feat-014`).
- **A result contradicts a Known Truth above:** treat it as a bug in the new code until proven otherwise; do not rewrite the paper's claims on one run.
- **Fine-tuning cost or any run > 24 GPU-hours:** stop and ask.
- **Anything that would identify authors, delete `output.zip`, or push to a remote:** ask.
