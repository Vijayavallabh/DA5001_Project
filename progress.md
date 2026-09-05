# Session Progress Log

## Current State

**Last Updated:** 2026-09-05
**Active Feature:** none in progress — feat-005 blocked on the sweeps (plain on GPU 2, chat on GPU 1); feat-001..004, 006, 008, 009 done
**Deadline clock:** abstract Sep 22 · paper Sep 29 · artifacts Oct 2 (AoE)

## Status

### What's Done

- [x] Full read of repo, both manuscripts, and the released logs; literature review (He et al. 2026, Vyas et al. 2023, Cohen 2025, ~35 related papers).
- [x] Reframing plan written: `~/sub/satml/IMPROVEMENT_PLAN.md` (spec for everything below).
- [x] Log reanalysis scripts drafted: `analysis/reanalyze_logs.py`, `analysis/surprisal.py` (copies also in `~/sub/satml/scripts/`).
- [x] Harness rebuilt: `AGENTS.md`, `CLAUDE.md` (imports AGENTS.md), `feature_list.json`, `progress.md`, `session-handoff.md`, `init.sh`, `GOAL.md`.

- [x] feat-001 closed: `./init.sh` exit 0; harness validator 100/100. Committed as 07a446c.
- [x] feat-002 closed (2026-09-05): `analysis/reanalyze_logs.py --logs output --out results` (6.8 s, no GPU) reproduces every Known Truth. Producing command: `unzip -o output.zip <4 trajectory files + heldout_validation.jsonl> && .venv/bin/python analysis/reanalyze_logs.py --logs output --out results`. Outputs: `results/regime_table.csv`, `llr_tails.csv`, `prefix_debt_forced_tokens.csv`, `surprisal.csv`, `seed_collisions.csv`, `per_trajectory.csv` (4,499 rows, 763 KB; input for feat-007/feat-013). New number: leading safe-forced tokens equal floor(δ_init/k) in 99.4–100% of trajectories. `analysis/surprisal.py` merged into the tool and removed.
- [x] feat-009 closed (2026-09-05): `analysis/composition_attack.py` on 100 memorised excerpts, k in {-1,0,0.15,0.5,1,3,5,10,20}, windows 20/50, single/oracle/chained → `results/composition.csv`, `results/composition_summary.csv`, `figures/composition.pdf`. Headline: reproduction < 10% at k <= 3 (budget binding, utilisation up to 0.97) even though the certificate is vacuous; single-query recall equals the unconstrained model at k=20 (0.48) where Z/K = 0.18; oracle composition reaches 0.86 at k=20 and doubles recall at k=5-10; zero violations. Producing command in feature_list.json.
- [x] feat-008 closed (2026-09-05): memorising risky model = LoRA fine-tune of Llama-3.1-8B-Instruct (`recipes/`), `results/memorizing_model_recall.csv`: greedy nv-recall 0.907 (attack_train) / 0.923 (val) / 0.0 (held-out test); sampled 0.80 / 0.81 / 0.0. Producing command: `scripts/run_memorizing_check.sh 1`.
- [x] feat-004 closed (2026-09-05): E1 accepts k=-1 (risky only, K=inf) and k=0 (safe only, K=0) with no certificate arithmetic for them; every record carries lcs_word, lcs_char, acs_word, nv_recall (dap/stats.py); h1_summary rows carry per-class metric means. Smoke: `CUDA_VISIBLE_DEVICES=1,2 HF_HUB_OFFLINE=1 .venv/bin/python h1.py --k-values -1 0 0.15 1 --trajectories-per-prompt 2 --cap-* 2 --output-dir output/smoke --trust-remote-code --parallelize` (24 files, 0 invariant violations). Seeds now depend only on the trajectory index so E1/E2 batch across prompts (per-prompt seeds had forced batch size 1: 12.5 s/trajectory).
- [x] feat-006 closed (2026-09-05): `analysis/certificate_cap.py` → `results/certificate_caps.csv` (758 passages), `results/certificate_cap_summary.csv`, `figures/certificate_cap_curve.{pdf,png}`. Producing command: `CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 .venv/bin/python analysis/certificate_cap.py --data data --out results`. Headline numbers: TinyComma surprisal of a CopyBench reference is median 205 nats (3.20 nats/token, 64 tokens); the K-NAF certificate is vacuous (S ≤ K) for 100% of passages at k=3 and k=5, 44% at k=1, 0% at k ≤ 0.5 (median cap 0.49 at k=0.5, 0.10 at k=0.1). Llama-3.1-8B-Instruct assigns the references 2.59 nats/token (median gap 37 nats): it has not memorised them.
- [x] feat-003 closed (2026-09-05): collision-free seeds (`dap/stats.py`, index-offset semantics in `dap/e2/evaluator.py`), per-trajectory utilisation/invariant/activity fields in E1 records and `h1_summary`, `utilisations`/`activity` lists in E2 `EvalResult`, R = K everywhere, `--use-chat-template` on `h1.py`/`h2.py` (`dap/shared.py: wrap_chat, chat_eos_ids, true_gen_len`), `tests/` (6 tests) + `tests/data/sample_trajectories.jsonl`. GPU smoke (local GPU 1, `HF_HUB_OFFLINE=1`, k=1, 2 traj × 2 prompts × 6 splits, `--use-chat-template`): `output/smoke_feat003/h1_summary.csv` shows invariant_violations=0 in all classes, util_max 0.95–1.00, active_step_pct 3.7–23.3%, chat-formatted generations stop at `<|eot_id|>` (lengths 11 and 28 observed). Command: `CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 .venv/bin/python h1.py --k-values 1.0 --trajectories-per-prompt 2 --cap-neutral 2 --cap-val 2 --cap-test 2 --cap-attack-train 2 --cap-factual 2 --cap-creative 2 --output-dir output/smoke_feat003 --use-chat-template`.

### What's In Progress

- (none)

### What's Next

1. feat-004 → feat-005: baselines and the small-k regime sweep.
2. feat-007 (after feat-005): `analysis/llr_tails.py --sweep plain=... --sweep chat=...`, retire EBB call sites in dap/ (E2: replace U_EBB/rho with max utilisation + invariant): memorising model, then the composition attack (the paper's core attack result).
3. feat-013, feat-014, feat-015 in that order. feat-010/011/012 only if time remains after feat-009.

## Blockers / Risks

- [ ] Compute: local 4×A100 80GB (GPUs 0 and 4 had ~15 GB in use by other processes on 2026-09-05; GPU 3 is a T400, unusable). DGX 6×H100 via `dgx-gpu` for 70B and sweeps. Llama-3.1-70B base in bf16 (~140 GB) fits on 2 free A100s with `device_map="auto"`, so feat-008's 70B option is feasible locally.
- [ ] **HF_TOKEN in `.env` is invalid** (`HfApi.whoami` → "Invalid user token", 2026-09-05). The local HF cache already holds Llama-3.1-8B-Instruct (weights only; tokenizer comes from the TinyComma repo), Llama-3.1-8B base, TinyComma, Qwen2.5-7B-Instruct, so all 8B runs work with `HF_HUB_OFFLINE=1`. Llama-3.1-70B is NOT cached: feat-008's 70B option needs a valid token (human) or the DGX cache. Ask the user for a working token when feat-008 starts.
- [ ] **DGX unreachable from this account (2026-09-05):** `ssh PrakashDGX_H2` → `Permission denied (publickey,password)` for user `prachh`; the `dgx-gpu` skill (`/home/sports/.config/opencode/skills/dgx-gpu/SKILL.md`) assumes a key that is not installed here. Per GOAL.md, all GPU work runs on the local A100s (indices 1 and 2 free; 0 and 4 at 100% util by other users). Ask the user to install the DGX key if the 70B run or the full sweep needs more than two GPUs.
- [ ] feat-005 sweep: `output/sweep_chat` crashed with CUDA OOM at ~17:50 while sharing GPU 2 with `output/sweep_plain` and the feat-008 fine-tune (peak of the prefix-debt full-vocab logits at batch 48). Relaunch after the fine-tune: `CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 .venv/bin/python h1.py <COMMON from scripts/run_regime_sweep.sh> --use-chat-template --output-dir output/sweep_chat` (its k=-1 neutral file is reused by --skip-existing). Never run more than two decoding jobs per 80 GB GPU at batch 48.
- [ ] feat-014 prep (2026-09-05): `~/sub/satml/sections/{threat_model,related_work,open_science}.tex` drafted (threat model with parties/goals/out-of-scope; related work citing 48 keys; Open Science + LLM-usage). To be `\input` by the feat-014 rewrite; the graphical-abstract step of the scientific-writing skill was deliberately skipped (figures must trace to results/*.csv).
- [ ] feat-014 prep (2026-09-05): `~/sub/satml/references.bib` rebuilt with 51 verified entries (arXiv metadata fetched and titles checked; venues added by hand; old 10 keys preserved; the old He et al. and Cohen titles were wrong). Page numbers are absent for conference papers (NeurIPS/ICLR/ICML/COLM), which is standard for those venues.
- [ ] **LaTeX toolchain (2026-09-05):** no `pdflatex`/`bibtex` on this box and no sudo; conda `texlive-core` installs a broken TeX Live (no `pdflatex.fmt`, perl scripts missing) and was removed. `~/.local/bin/tectonic` (0.15) compiles `~/sub/satml/satml_2027_new.tex` (auto-fetches packages, runs the bibliography): `cd ~/sub/satml && tectonic -X compile satml_2027_new.tex`. The old manuscript compiles to 10 pages, the new skeleton to 6. Success criterion 3 will be checked with tectonic; the artifact README will state that `pdflatex + bibtex` produce the same PDF on a standard TeX Live.
- [ ] feat-010 prep (2026-09-05): `analysis/bank_burst.py` written and smoke-tested (`output/bank_burst_smoke/`). First filler design ("Repeat the following text exactly, then continue it: one, two, ... <seed>") does not bank: the memoriser diverges from TinyComma by ~2.5 nats/token on the filler and does not continue the passage after it (nv-recall 0 even at k=-1). Needs a filler the memoriser reproduces AND the anchor predicts (e.g. put the filler in the training format, or use the passage's own prefix as filler); inspect the `generated` column before iterating. Optional feature; revisit after feat-013/014.
- [ ] Batched decoding keeps accruing budget after a sequence's own EOS (`budget_so_far = (t+1)k − δ` for the whole batch), so `final_budget` and `generation_length_tokens` in the released logs reflect the batch length for early-EOS trajectories; utilisation of those trajectories is understated. feat-004 should compute B at the sequence's own EOS (`true_gen_len`) when reporting utilisation.
- [x] feat-008 decision (2026-09-05, ahead of the Sep 12 deadline): LoRA-memorised Llama-3.1-8B-Instruct (`recipes/finetune_memorizing.py`, merged weights in `output/memorizing_llama8b`, 73 min on one A100). Final token loss 0.058; greedy nv-recall 0.907 on training excerpts. The 70B-base option is off the table locally (not cached, invalid token, DGX unreachable).
- [ ] feat-008 (memorising model) was the schedule risk: Llama-3.1-70B base needs ~140 GB bf16; fine-tuning 8B is the fallback (≈2–4 GPU-hours). Decide by Sep 12.
- [ ] Known code issues to fix in feat-003, not before: seed collision in `dap/e2/runner.py::_eval_specs` (offset n0) and the top-up path (offset 10); E1 uses R = T·ln|V| while E2 uses R = K; `B_eff` conflates generation length with budget; instruct model called without chat template.
- [x] feat-002 extended (2026-09-05): all 18 released trajectory files reanalysed (`unzip -o output.zip 'output/h1_outputs/trajectories_k*.jsonl' && .venv/bin/python analysis/reanalyze_logs.py --logs output --out results`): 26,999 trajectories, 0 invariant and 0 per-step violations; L>K at k=1 in 738/8,999 trajectories (attack_train 96/999, val 100/1500, test 152/1500, neutral 120/2000, factual 111/1500, creative 159/1500); solver active 0.09-0.45% of steps at k in {3,5} for every class; identical generations between k=3 and k=5: 44.6-64.5% per class. `results/per_trajectory.csv` is now 27k rows (4.3 MB).
- [ ] `trajectories_k1_attack_train.jsonl` has no `p_risky_prob` field (older run), so `L_risky` is 0 for k=1 in `results/`; the k=1 attack_train file has 999 rows, not 1000. Only 4 of 18 h1 trajectory files are reanalysed so far (the evidence set); extend to val/neutral/factual/creative in feat-013 if a per-domain table is needed.
- [ ] Unverified claims in the current manuscript: "Regime A pivot is the more common behaviour" (a crude check finds ~3–4%); the "not i.i.d. common random numbers" caveat is unnecessary (`torch.multinomial` rows are independent).

## Decisions Made

- **Retire the empirical-Bernstein proxy and ρ (2026-09-05).** Z ≤ K holds per trajectory by construction (KL chain rule), so a mean bound cannot fail; report per-trajectory utilisation, realised LLR tails, and certificate caps instead. Alternatives considered: keep EBB with R = K (bound gets looser, still vacuous), betting CS on Z/K (only where a mean is genuinely the target).
- **Audit at He et al.'s small-k regime (2026-09-05).** At k=3/5 the constraint was active in ≤0.2% of steps; the sweep now spans {0.1 … 1} plus k=−1/0 baselines.
- **Attacks over search (2026-09-05).** Drop the surrogate/k-DPP/ρ search; the SaTML core is the composition attack (Cohen Thm 3.5 instantiated), bank-and-burst, and a memorising risky model.

## Files Modified This Session

- `AGENTS.md`, `CLAUDE.md`, `feature_list.json`, `progress.md`, `session-handoff.md`, `init.sh`, `GOAL.md` — harness rebuilt for the SaTML plan.
- `analysis/reanalyze_logs.py` — argparse reanalysis tool (feat-002); `results/*.csv` generated.
- feat-004: `dap/stats.py` (metrics, index-only seeds), `dap/e1.py` (baselines, summary means), `analysis/reanalyze_logs.py` (k-1 file names), `tests/test_metrics.py`.
- feat-006: `analysis/certificate_cap.py`, `results/certificate_caps.csv`, `results/certificate_cap_summary.csv`, `figures/certificate_cap_curve.{pdf,png}`.
- feat-003: `dap/stats.py`, `dap/shared.py`, `dap/e1.py`, `dap/e2/evaluator.py`, `dap/e2/runner.py`, `dap/e2/types.py`, `a_patch/factory.py` (pad id when eos is a list), `tests/`.
- `~/sub/satml/IMPROVEMENT_PLAN.md`, `~/sub/satml/scripts/*` — plan and script copies (outside repo).

## Evidence of Completion

- [x] `./init.sh` passes (2026-09-05): all [OK] lines, `[info] local GPU available: True, count: 5`, `=== Init Complete ===`, exit 0.
- [x] `node .../harness-creator/scripts/validate-harness.mjs --target .` → Overall 100/100 (all five subsystems 5/5).
- [x] feat-002 evidence command exit 0; see `feature_list.json`.
- [x] feat-003: `pytest -q tests` 6 passed; `./init.sh` exit 0; GPU smoke `output/smoke_feat003/` invariant_violations=0.
- [x] feat-006: evidence command exit 0; `results/certificate_cap_summary.csv` written.
- [x] feat-004: smoke evidence command exit 0 (`output/smoke/`, 24 files); 9 tests pass.
- [x] feat-008: `scripts/run_memorizing_check.sh 1` exit 0; `results/memorizing_model_recall.csv` written.
- [x] feat-009: composition evidence command exit 0; `results/composition_summary.csv` (45 rows, 0 violations).

## Notes for Next Session

- The numbers in AGENTS.md "Known truths" were computed this session from `output.zip`; feat-002 must reproduce them exactly before anything is cited.
- `output.zip` is 2.4 GB; extract only the four trajectory files listed in feat-002's evidence command.
- Reproduction commands for the original runs are in `README.md`; keep them for the artifact's "original runs" section.
