# Artifact: A KL Budget Is Uninformative Where the Mechanism Is Usable

*An adversarial audit of inference-time near-access-freeness (SaTML 2027 submission).*

Anonymised code, results, prompt sets, and recipes for the SaTML 2027 submission. Every number in the paper
traces to a file in `results/`, and every figure is rebuilt from those files by `figures/make_figures.py`.

## Layout

| Path | Content |
|---|---|
| `a_patch/` | The audited Anchored Decoding library (mechanism unchanged; `k_radius=-1` = risky only, `0` = anchor only). Additions for this audit: `constraint='kl'\|'pathwise'` (Δmax accounting, `pathwise.py`), `bank_cap` (token bucket, `bank.py`), and `warp.py` (temperature and repetition penalty applied to both logit vectors before the solve, as He et al.'s Appendix B specifies). |
| `dap/` | Experiment code: `h1.py -> dap/e1.py` (fixed workload, baselines, copying metrics), `h2.py -> dap/e2/` (prompt search; Bernstein proxy retired), `dap/stats.py` (seeds, metrics, per-trajectory `budget_check`, anytime-valid confidence sequence). |
| `analysis/` | One script per audit: `reanalyze_logs.py` (C1/C2/C4 on released logs), `certificate_cap.py` (C3, with `--temperature/--repetition-penalty` for the warped anchor), `regime_sweep.py` (C2 sweep), `llr_tails.py` (C4), `composition_attack.py` (C5, C7: `--constraint pathwise`, `--bank-cap`, `--retries`, `--raw-prompt`, `--no-prefix-debt`, per-query logs), `natural_memorisation.py` (C7 aggregation + figure), `warped_anchor.py` (C3 under decoding settings), `latent_leakage.py` (anchor exposure), `budget_path.py` (Prop. 4 feasibility), `odometer.py` (C10 per-user budget replay), `check_bank_cap.py` and `burst_audit.py` (bank cap), `concentration.py` (C9 Freedman certificate), `pathwise_price.py` (C8 utility price), `extraction_cost.py` (Prop. 2), `recheck_violations.py` (per-query invariant recheck), `compute_hours.py` (GPU-hours from the run directories), `bank_burst.py`, `memorizing_recall.py`. Phase 3: `separation.py` (Prop. 5, the protective ratio and its figure, no GPU), `length_scaling.py` (vacuity against passage length), `utility.py` (C12, the judged utility table with its null arm), `cpfuse_audit.py` (the second mechanism), `anchor_control.py` (the two anchors on one identical span), `merge_prefix_debt.py` (rebuilds the prefix-debt table from the run directories). |
| `recipes/` | `finetune_memorizing.py` + `memorizing_model.md`: the memorising risky model (weights not redistributed: they reproduce copyrighted text). |
| `results/` | All CSV tables cited in the paper (see below). |
| `figures/` | Paper figures (PDF/PNG) and `make_figures.py`. |
| `data/` | Prompt sets (CopyBench book split, FactScore, WritingPrompts, neutral QA) exactly as sampled. |
| `tests/` | `pytest -q tests` — 47 tests (seeds, invariants on a log sample, metrics, confidence sequence, budget checks, the pathwise and bank-cap rules, warping, budget-path feasibility, and the phase-3 additions: the separation inequality, length scaling, the utility verdict logic, the CP-Fuse fusion step, and the GPU-hour parser). |
| `scripts/` | Launchers used for the runs below: `run_regime_sweep.sh`, `run_memorizing_check.sh`, `run_natural_memorisation.sh`, `run_bank_cap.sh`, `run_prefix_debt_k20.sh`, `download_second_anchor.py`, `build_artifact.sh`. |

## Results files

Released logs: `regime_table.csv`, `llr_tails.csv`, `prefix_debt_forced_tokens.csv`, `surprisal.csv`, `seed_collisions.csv`, `per_trajectory.csv`.
Certificate strength: `certificate_caps.csv`, `certificate_cap_summary.csv`, `certificate_caps_memoriser.csv`, `warped_anchor.csv`, `latent_leakage_summary.csv`.
Sweeps: `regime_sweep.csv`, `llr_ratio_samples.csv`, `pathwise_price.csv`, `concentration.csv`, `concentration_summary.csv`.
Attacks: `memorizing_model_recall.csv`, `composition.csv`, `composition_summary.csv` (phase 1), `composition_8b_kl.csv`, `composition_8b_pathwise.csv` (+ `_per_passage`), `budget_path.csv`, `budget_path_summary.csv`, `prefix_debt_ablation.csv`, `extraction_cost_{kl,pathwise,pathwise_lo}.csv` (+ `_windows`).
Phase 3: `separation.csv`, `separation_summary.csv` (Prop. 5), `length_scaling.csv` (+ `_summary`), `utility.csv`, `utility_summary.csv` (C12), `cpfuse_audit.csv`, `cpfuse_audit_examples.csv`, `anchor_control.csv`, `certificate_caps_comma7b.csv` and `certificate_cap_summary_comma7b.csv` (the second anchor), `compute_hours.csv`.
70B natural memorisation: `natural_memorisation.csv`, `composition_70b.csv`.
Accounting across queries: `odometer.csv`, `odometer_per_passage.csv`, `bank_cap.csv`, `burst_audit.csv`.
Compute: `compute_hours.csv` (per-job wall times and GPU-hours behind the paper's compute statement, from the run directories).

Every experiment reporting a copying or spend metric at some `k` also has `k = -1` (risky model alone) and `k = 0` (anchor alone) rows on the same prompts and seeds. A budget violation is per-trajectory (`Z > max(0,B) + 1e-3`, or `R_T` under pathwise accounting); `analysis/recheck_violations.py` recomputes it from any per-query log. Across every run in this artifact the count is zero **under the rule the run enforced**.

One column needs a word of explanation. In `composition_8b_pathwise.csv` and the pathwise rows of `composition_70b.csv`, `invariant_violations` counts queries whose *KL* spend left the budget (`Z > max{0,B}`): 146 of 23,844 for the 8B model (144 at `k=1`, 2 at `k=3`) and none for the 70B model. The pathwise decoder does not bound `Z` and is not meant to; it bounds the realised log-ratio `R`, which stayed within budget in every one of those queries, and the paper reports the KL excursion (it reaches `1.12K` at `k=1`). Recompute either rule with `analysis/recheck_violations.py --queries <run>/queries.jsonl --constraint pathwise`. For the KL runs the column means what it says.

## Reproduction

```bash
python -m venv .venv && .venv/bin/pip install torch transformers peft accelerate numpy matplotlib pytest pypdf feedparser
.venv/bin/python -m pytest -q tests
# released logs (2.4 GB output.zip, shipped next to this artifact) -> results/regime_table.csv etc.
unzip -o output.zip 'output/h1_outputs/trajectories_k*.jsonl' && .venv/bin/python analysis/reanalyze_logs.py --logs output --out results
# certificate strength (one A100, ~2 min)
.venv/bin/python analysis/certificate_cap.py --data data --out results
# small-budget sweep (two A100s, ~10 h) and its summaries
scripts/run_regime_sweep.sh 0 1
.venv/bin/python analysis/regime_sweep.py --run plain=output/sweep_plain --run chat=output/sweep_chat --out results
.venv/bin/python analysis/llr_tails.py --sweep plain=output/sweep_plain --sweep chat=output/sweep_chat --out results
# memorising model (one A100, ~75 min) and its check
.venv/bin/python recipes/finetune_memorizing.py --out output/memorizing_llama8b && scripts/run_memorizing_check.sh 0
# composition attack (one A100, ~70 min)
.venv/bin/python analysis/composition_attack.py --risky-model output/memorizing_llama8b --limit 100 --k-values -1 0 0.15 0.5 1 3 5 10 20 --windows 20 50 --out results
# figures
.venv/bin/python figures/make_figures.py --copy-to ""
```

### Phase 2 (all commands set `CUDA_DEVICE_ORDER=PCI_BUS_ID`; GPU indices follow `nvidia-smi`)

```bash
# composition attacks with per-query logs (one A100 each, ~3 h): KL and pathwise decoders, then the odometer replay and bank caps
.venv/bin/python analysis/composition_attack.py --risky-model output/memorizing_llama8b --limit 100 --k-values -1 0 3 5 10 20 --modes single oracle chained --windows 20 50 --out output/phase2/comp8b_kl --queries-out output/phase2/comp8b_kl/queries.jsonl
.venv/bin/python analysis/composition_attack.py --risky-model output/memorizing_llama8b --limit 100 --k-values -1 0 1 3 5 10 20 50 --modes single oracle chained --windows 20 50 --constraint pathwise --out output/phase2/comp8b_pathwise --queries-out output/phase2/comp8b_pathwise/queries.jsonl
.venv/bin/python analysis/odometer.py --queries output/phase2/comp8b_kl/queries.jsonl --out results
scripts/run_bank_cap.sh
# prefix-debt ablation and retry runs (extraction cost)
.venv/bin/python analysis/composition_attack.py --risky-model output/memorizing_llama8b --limit 100 --k-values -1 0 1 3 5 10 --modes single oracle --windows 50 --no-prefix-debt --out output/phase2/prefix_ablation --queries-out output/phase2/prefix_ablation/queries.jsonl
.venv/bin/python analysis/composition_attack.py --risky-model output/memorizing_llama8b --limit 50 --k-values -1 5 10 20 50 --modes oracle --windows 50 --retries 8 --constraint pathwise --out output/phase2/retries_pathwise --queries-out output/phase2/retries_pathwise/queries.jsonl
.venv/bin/python analysis/extraction_cost.py --queries output/phase2/retries_pathwise/queries.jsonl --constraint pathwise --limit 50 --out results --prefix extraction_cost_pathwise
# pathwise and concentration sweeps on the 900 prompts (one A100 each; ~13 h and ~6 h) and their summaries
.venv/bin/python h1.py --k-values 0.5 1 3 5 10 20 --constraint pathwise --trajectories-per-prompt 3 --output-dir output/phase2/pathwise_sweep
.venv/bin/python h1.py --k-values 0.5 1 --trajectories-per-prompt 3 --output-dir output/phase2/kl_sweep_conc
.venv/bin/python analysis/pathwise_price.py --kl output/sweep_plain --pathwise output/phase2/pathwise_sweep --out results
.venv/bin/python analysis/concentration.py --logs output/phase2/kl_sweep_conc --out results
# 70B natural memorisation (two A100s, ~5 h; the risky model is the public Llama-3.1-70B base checkpoint named in the script)
scripts/run_natural_memorisation.sh 16 && .venv/bin/python analysis/natural_memorisation.py --runs output/phase2/nm --out results --figures figures
# warped anchor, latent leakage, budget path, burst audit
.venv/bin/python analysis/warped_anchor.py --out results; .venv/bin/python analysis/latent_leakage.py --out results
.venv/bin/python analysis/budget_path.py --out results; .venv/bin/python analysis/burst_audit.py --out results
```

Models: anchor `jacquelinehe/tinycomma-1.8b-llama3-tokenizer`; risky `meta-llama/Llama-3.1-8B-Instruct` (gated; set `HF_TOKEN`)
and, for the natural-memorisation audit, the public Llama-3.1-70B base checkpoint in bf16 (an ungated mirror works; the run needs
two 80 GB GPUs); optimiser `Qwen/Qwen2.5-7B-Instruct`. Set `CUDA_VISIBLE_DEVICES` for every job. The paper compiles with
`pdflatex + bibtex` or `tectonic -X compile satml_2027.tex`.

## Integrity

`MANIFEST.sha256` lists every file; verify with `sha256sum -c MANIFEST.sha256`.

### Phase 3 (2026-09-07)

```bash
# second anchor, surprisal only (one A100, ~1 min): Comma v0.1-7B on the same passages.
# Its vocabulary is 64,000 against Llama-3's 128,256, so it cannot be fused -- only totals in
# nats are comparable, never a per-token rate beside TinyComma's.
.venv/bin/python scripts/download_second_anchor.py
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/certificate_cap.py --safe-model common-pile/comma-v0.1-2t \
    --risky-model '' --tag _comma7b --data data --out results
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/anchor_control.py --out results
# vacuity against passage length (one A100, ~5 min): one forward pass per work gives every
# prefix length and every window
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 \
  .venv/bin/python analysis/length_scaling.py --out results --figures figures
# Proposition 5: no GPU, reads results/per_trajectory.csv and results/odometer_per_passage.csv
.venv/bin/python analysis/separation.py --results results --out results --figures figures
# prefix-debt ablation at k=20: the 8B row, then the 70B row (TWO A100s), then the merge
.venv/bin/python analysis/composition_attack.py --risky-model output/memorizing_llama8b \
  --limit 100 --k-values 20 --modes single oracle --windows 50 --no-prefix-debt \
  --out output/phase3/prefix_ablation_k20 --queries-out output/phase3/prefix_ablation_k20/queries.jsonl
scripts/run_prefix_debt_k20.sh
.venv/bin/python analysis/merge_prefix_debt.py --out results
# utility (C12), one A100 ~20 min: no generation, it judges arms the phase-2 sweeps already produced.
# 200 pairs per class x 3 classes = 600 judged pairs per arm, plus the null arm.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  .venv/bin/python analysis/utility.py --out results --judge-per-cell 200
# second mechanism (CP-Fuse): two shard fine-tunes on disjoint halves of the attack split
# (one A100 each, ~15 min), then the audit
.venv/bin/python recipes/finetune_memorizing.py --splits attack_train --shard 0/2 --out output/phase3/cpfuse_m0
.venv/bin/python recipes/finetune_memorizing.py --splits attack_train --shard 1/2 --out output/phase3/cpfuse_m1
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  .venv/bin/python analysis/cpfuse_audit.py --model-a output/phase3/cpfuse_m0 \
    --model-b output/phase3/cpfuse_m1 --out results --limit 60
# GPU-hours behind the compute statement, and the figures
.venv/bin/python analysis/compute_hours.py --out results
.venv/bin/python figures/make_figures.py --copy-to ""
```

### Phase 4 (2026-09-07) — the frontier theorem's three rates

```bash
# the three regime boundaries from the safe model alone, in nats per CHARACTER so the
# comparison is tokenizer-invariant across ten safe models with three vocabularies
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/regimes.py --out results
# the anchor-scaling law: 10 safe models x 3 corpora (one A100, ~40 min)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/anchor_scaling.py --out results
# the same trend under two further risky models, separating the base-vs-instruct confound
.venv/bin/python analysis/anchor_scaling.py --ordinary-jsonl <gen.jsonl> --risky <id> --tag _qwen
# where the uncertified interval comes from. --denominator token repeats it per token, which is
# what shows the opening effect is real but SMALLER than the per-character figures suggest
for M in jacquelinehe/tinycomma-1.8b-llama3-tokenizer PleIAs/Pleias-3b-Preview \
         alea-institute/kl3m-003-3.7b common-pile/comma-v0.1-2t; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/opening_effect.py --model "$M" --denominator token --tag _token --out results
done
# the Renyi-alpha family at one budget: attack recall and price for alpha in {1,2,4,8}
.venv/bin/python analysis/renyi_sweep.py --out results --price-runs 'output/phase4/util_*' --price-class all
```

### Phase 5 (2026-09-08) — the onset, derived rather than fitted

```bash
# r(x) = s_s(x) - s_r(x) per work, and the onset it predicts. The pair set is DATA:
# results/onset_theory_pairs.tsv, whose 4th column (the measured onset) is optional, so a pair
# with no sweep yet yields a prediction only.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/onset_theory.py --out results
# measured onsets and the collapse, over every pair in results/onset_pairs.tsv
.venv/bin/python analysis/onset.py --out results --thresh 0.01
# score the derivation against a constant coefficient; for rungs sharing an anchor it reports the
# sign of the measured trend against the sign each hypothesis requires
.venv/bin/python analysis/onset_ladder.py --out results
# does r(x) screen an INDIVIDUAL work? (no GPU) -- it does not, and s(x) alone does it better
.venv/bin/python analysis/per_work_screen.py --out results
# the three named critiques of the collapse: metric, threshold and normaliser (no GPU)
.venv/bin/python analysis/collapse_robustness.py --out results
# building a self-paired memoriser. --target-modules all-linear is required for GPT-NeoX models;
# the check reports SAMPLED as well as greedy recall, and a pair enters the onset analysis only if
# its sampled k=-1 recall is >= 0.10
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python recipes/finetune_memorizing.py --base <model> --tokenizer <model> --no-chat \
    --target-modules all-linear --epochs 40 --rank 128 --lr 3e-4 --out output/phase5/mem_<tag> --check 16
```

### Phase 5b (2026-09-10) — what the evaluation hands the adversary, and moving `s(x)` inside a pair

Every arm below was pre-registered with a refuting band **before it ran**:
`results/onset_prediction_seed.md` (four addenda) and `results/onset_prediction_temperature.md`
(two pairs). Score against the committed band; do not refit.

```bash
# One arm of the seed intervention. The seed is a TOKEN count, so what it buys in words is the
# tokenizer's business -- which is the confound the arms exist to break.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/composition_attack.py \
    --safe-model output/phase5/anchor_kl3m-002-520m --risky-model output/phase5/mem_kl3m-002-520m \
    --seed-tokens 40 --k-values -1 0 1.6 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.6 3.0 \
    --modes single --limit 100 --out output/phase5/seed40_kl3m520m
# Its budget path, which needs the ANCHOR ONLY -- no memoriser, no attack, no decoding -- so it is
# a prediction available before the sweep rather than a fit to it.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/budget_path.py --safe-model output/phase5/anchor_kl3m-002-520m \
    --composition '' --limit 100 --seed-tokens 40 --out results --prefix budget_path_kl3m520m_seed40
# One temperature arm: the decoder warps BOTH logit vectors before the KL solve (He et al. App. B),
# so temperature moves s(x) with the anchor, memoriser, corpus, tokenizer and seed all fixed.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/composition_attack.py \
    --safe-model output/phase5/anchor_kl3m-002-520m --risky-model output/phase5/mem_kl3m-002-520m \
    --temperature 0.4 --k-values -1 0 2.4 2.8 3.2 3.4 3.6 3.8 4.0 4.2 4.6 5.2 \
    --modes single --limit 100 --out output/phase5/warp_t0.4_kl3m520m
# Score every arm in results/seed_effect_runs.tsv: onset, bootstrap CI, k_crit, and the
# token-bucket prediction calibrated on each pair's control arm alone (no GPU, needs tokenizers)
HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache .venv/bin/python analysis/seed_effect.py --out results
# The units claim conditioned on the adversary's context, with both multiplicity checks (no GPU)
.venv/bin/python analysis/score_predictions.py --out results \
  --calibrated-on "TinyComma-1.8B + mem. Llama-3.1-8B" "Comma-7B + mem. Comma-7B"
# GPU-hours: phases 1-3 from the JOBS table, phases 4-5 scanned by launcher log minus traced sleeps
.venv/bin/python analysis/compute_hours.py --out results
```

### Phase 5c (2026-09-10) — the judged crossover on a fine grid, two judges

```bash
# 1. generate the ordinary-prompt workload at the budgets that bracket the crossover
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,1 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python h1.py --k-values 0.6 0.7 0.8 \
    --cap-neutral 200 --cap-factual 150 --cap-creative 150 --cap-val 0 --cap-test 0 \
    --cap-attack-train 0 --output-dir output/phase5/util_cross
# 2. judge every arm against the anchor-only arm, once per judge. --extra-arm is
#    'label|constraint|k|run-dir' and may repeat; 600 comparisons per arm needs --judge-per-cell 200
#    across the three prompt classes.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/utility.py --out results --prefix utility_v6 \
    --judge Qwen/Qwen2.5-7B-Instruct --judge-per-cell 200 \
    --extra-arm 'KL|kl|0.6|output/phase5/util_cross' \
    --extra-arm 'KL|kl|0.7|output/phase5/util_cross' \
    --extra-arm 'KL|kl|0.8|output/phase5/util_cross'    # ... and the k=1.5,2,2.5 and Renyi arms
# 3. the separations and the interpolated crossing, written to results/crossover.csv (no GPU)
.venv/bin/python analysis/judge_separation.py --summary results/utility_v6_summary.csv \
  --out results --out-name judge_separation_v6.csv --crossing-sigma -2 --crossing-arm KL
.venv/bin/python analysis/judge_separation.py --summary results/utility_v6_judge2_summary.csv \
  --out results --out-name judge_separation_v6_judge2.csv --crossing-sigma -2 --crossing-arm KL
```

The second judge is `microsoft/Phi-3.5-mini-instruct`; run step 2 again with
`--prefix utility_v6_judge2 --judge microsoft/Phi-3.5-mini-instruct`. Every separation is measured
against **that judge's own anchor arm**, so the two judges' different absolute loss rates cancel.

### Phase 5d (2026-09-10) — the last two arms, and the memoriser-strength control

Two arms closed the intervention set. The `KL3M-1.7B` seed-40 replication first returned an
implausibly tight interval alongside a **38.0% no-crossing fraction**, because its recall curve rose
past the threshold of four LCS words and then flattened just above it at the top of the grid; the
pre-registration commits to extending a grid whenever that fraction rises materially above the other
arms' 0.0–0.9%, so the grid was extended and the arm rescored on the merge.

```bash
# extend the ceiling-limited grid, then merge (the same protocol as the seed-80 arm)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/composition_attack.py \
    --safe-model alea-institute/kl3m-003-1.7b --risky-model output/phase5/mem_kl3m-003-1_7b \
    --seed-tokens 40 --k-values 3.5 4.0 5.0 --modes single --limit 100 \
    --out output/phase5/seed40_kl3m17b_hi
mkdir -p output/phase5/seed40_kl3m17b_merged
cp output/phase5/seed40_kl3m17b/composition.csv output/phase5/seed40_kl3m17b_merged/composition.csv
tail -n +2 output/phase5/seed40_kl3m17b_hi/composition.csv \
  >> output/phase5/seed40_kl3m17b_merged/composition.csv
# rescore every arm against its committed band; the manifest points at the merged directory
HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache .venv/bin/python analysis/seed_effect.py --out results
```

The extension takes the no-crossing fraction to 0.0%, leaves the point estimate at 0.979 and widens
the interval from the artefactual `[0.96, 0.98]` to `[0.96, 1.67]`. **Report the no-crossing
fraction alongside any bootstrapped threshold crossing:** an interval computed from the resamples
that happened to cross is conditioned on crossing, and it is narrow for the same reason it is wrong.

The temperature arms carry one confound, which the next command controls. Because the decoder warps
**both** logit vectors before the KL solve (He et al., App. B), `tau = 0.4` sharpens the risky model
too, and a sharper memoriser leaks at a lower budget — biasing the elasticity toward the
constant-nats null being refuted. Restricting both arms to the passages the memoriser reproduces in
*both* at `k = -1` matches unconstrained strength by construction. No GPU:

```bash
.venv/bin/python analysis/matched_strength.py --out results   # -> results/matched_strength.csv
```

Closing a strength gap of 0.519/0.904 (KL3M-520M) and 0.909/0.964 (Pleias-1.2B) leaves the
elasticity at **+0.72** and **+0.61**, unchanged to two decimals, both intervals still excluding 0.
Pre-registrations for every arm on this page are `results/onset_prediction_seed.md`,
`results/onset_prediction_temperature.md` and `results/onset_prediction_matched_strength.md`, each
committed before the run it scores.

### Phase 5e (2026-09-10) — is the gap to the information floor a scheduling artefact?

Section 2 measures three to four orders of magnitude between what the decoder spends and the floor
Theorem 1 puts under any policy, and names one route to closing it: stop decoding greedily against
the token bucket. Phase 5e closes that question offline, with no decoder change and no sampling.

On the geometric path `p_theta ~ p_s^(1-theta) p_r^theta`, both quantities the decoder trades are
closed forms in `psi(u) = log sum_v p_s(v) e^{u l(v)}` with `l = log p_r - log p_s`:

```
charge    C(theta) = theta psi'(theta) - psi(theta),        C'(theta) = theta psi''(theta)
fidelity  G(theta) = theta m - psi(theta) = -D(p_r||p_theta) + const,  G'(theta) = m - psi'(theta)
```

with `m = psi'(1)`. `G'(0) = m - psi'(0)` is the Jeffreys divergence between the two models, so the
first nat spent at a step is worth a Jeffreys divergence and the last is worth nothing. Greedy takes
whatever marginal price its per-step allowance lands on; the optimal allocation of a *fixed total*
equalises `G'/C'` at one shared price `lambda`, found by bisection. Both sides need only two
teacher-forced forward passes:

```bash
# one (pair, target type, budget) cell; scripts/run_marginal_price.sh sweeps k for one pair
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/marginal_price.py --safe-model output/phase5/anchor_kl3m-002-520m \
    --risky-model output/phase5/mem_kl3m-002-520m --k 0.25 --limit 30 --out results \
    --prefix mp_kl3m_prot_k0.25
# --sample-target swaps the protected passage for a sample the risky model drew on an ordinary
# prompt, which is the quantity Theorem 1 actually prices
LIMIT=30 scripts/run_marginal_price.sh 1 output/phase5/anchor_kl3m-002-520m \
  output/phase5/mem_kl3m-002-520m kl3m_util --sample-target
.venv/bin/python analysis/marginal_price_table.py --out results  # -> marginal_price_table.csv
```

**Answer: no.** Reallocating greedy's own total spend optimally buys **1.008-1.125x** more fidelity
across 2 pairs x 2 target types x 4 budgets, and the protected and ordinary columns agree to within
0.005 — the headroom is a property of the geometry, not of what is being copied. The unconditional
form needs no allocation argument at all: at `k = 1` greedy already captures 77-85% of the fidelity
that `theta = 1` everywhere would buy, i.e. the risky model served outright under an unlimited
budget, so unlimited budget is worth 1.2-1.3x and at `k = 3` it is worth 1.01-1.03x. Three orders of
magnitude are not there to recover. Pre-registered in `results/onset_prediction_dual.md`.

Two caveats, both real: the allocation is computed along a fixed teacher-forced trajectory, and a
decoder that spent differently would walk a different one; and fidelity to `p_r` is not judged
utility — but the mechanism has no utility signal, so fidelity is the only thing its budget can buy,
which is exactly what makes the ceiling bite.

**A unit a deployer can compute without the protected work.** Section 3 prescribes publishing
`k/s(x)` and then concedes that a deployer has not seen the rights-holder's work. The anchor's
surprisal rate on *public-domain* prose of the same kind needs no protected text at all, and the
repository already carries it for ten anchors across three corpus families. Scored the way every
other predictor here is scored — leave one anchor out, fit on the rest, predict the held-out
anchor's protected rate (no GPU):

```bash
.venv/bin/python analysis/proxy_budget.py --out results   # -> proxy_budget{,_summary}.csv
```

Over a protected rate spanning 1.86x, the rescaled public-domain proxy predicts to **0.0518 nats per
character (5.6%)**, **4.13x better** than quoting a constant; `c_use` rescaling is 1.9x *worse* than
the constant. The ratio `s_protected/s_proxy` is 1.141 +- 0.077 (cv 6.75%) and is family-structured
(KL3M 1.06-1.09, Pleias 1.13-1.16, Comma 1.25-1.29), so the residual is a per-family constant a
deployer could calibrate once.

**What each Renyi order buys and lets through at one published k.** Table 1 compares four decoders
at matched *budget*, not matched utility, and Section 5 shows the judge cannot supply the missing
axis. `analysis/order_price.py` supplies it without a judge, and its two columns are deliberately
different functionals, because extraction and utility are different kinds of quantity:

```bash
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/order_price.py --safe-model output/phase5/anchor_kl3m-002-520m \
    --risky-model output/phase5/mem_kl3m-002-520m --k 3.0 --limit 25 --out results \
    --prefix order_price_kl3m_k3
```

*price* is fidelity on an ordinary generation — a bounded average, which fidelity measures correctly.
*leakage* is `sum_t log p_theta(x_t | x_<t)` over the protected tokens, whose exponential **is** the
reproduction probability. The first version of this script used fidelity for both and that was
wrong: a 15% cut in an average can collapse a product over hundreds of steps by orders of magnitude
without moving the average, which is why Table 1's oracle recall falls 24x from `alpha = 1` to
`alpha = 4` while average fidelity moves by 14%. That null and its diagnosis are recorded in
`results/onset_prediction_orders_matched.md`, with the corrected design committed before it ran. The
run prints the two brackets the constrained decoder must sit inside — the same log-probability under
the risky model above and under the anchor alone below — because an instrument that leaves the
bracket is not measuring a constrained decoder.

**The order comparison at matched utility.** Table 1 ranks four Rényi orders at one published `k`
and the paper concedes the limit of that: at matched *budget*, an order that leaks less may simply
be buying less. Section 5 shows the judge cannot supply the missing axis. The geodesic can. Fidelity
is monotone in `k` at a fixed order, so each order has a unique budget buying exactly what the
audited decoder buys at the published one; what it lets through there is the rare-event functional,
not another average. `analysis/order_frontier.py` sweeps a `k` grid in one pass and interpolates:

```bash
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/order_frontier.py \
    --safe-model output/phase5/anchor_kl3m-002-520m --risky-model output/phase5/mem_kl3m-002-520m \
    --limit 25 --k-grid 0.5 0.75 1.0 1.5 2.0 2.5 3.0 4.0 5.5 7.5 10.0 14.0 \
    --published-k 1.0 3.0 --out results --prefix order_frontier_kl3m
# -> order_frontier_kl3m.csv (one row per alpha x k) and order_frontier_kl3m_matched.csv
```

The answer is not the one the matched-budget table suggests. At the published `k = 3` the audited
decoder already captures 97-98% of what an unlimited budget would buy, so matching it pushes every
other order past that pair's vacuity threshold, and on KL3M-520M the ranking **reverses**: `alpha=4`
and `alpha=8` leak more at equal utility. At `k = 1`, where the constraint binds, the higher order
does dominate -- but by 871x on one pair and 3.5e4 to 3.3e7 on the other, non-monotone in `alpha`.
What Table 1 ranks is the charge function, not the decoder. Pre-registration and scoring:
`results/onset_prediction_orders_matched.md`.

**A note on splits, because it cost a result.** Every phase-5 memoriser is fine-tuned on
`attack_train` + `val` with `test` held out, and the two are disjoint in novel. A probe that scores
"protected" text on `test` is scoring a novel the model has never seen, where a LoRA-memorised model
is *worse* than its own base -- so `analysis/marginal_price.py` and `analysis/order_price.py` both
default to `attack_train` and print the bracket that catches the mistake if the wrong split is
passed: the served distribution's log-probability of the protected tokens must sit strictly between
the risky model's and the anchor's.

**What predicts the order's value? Nothing measured does.** `analysis/order_law.py` re-analyses the
frontier grids with no new compute, treating every grid `k` in turn as the published budget so the
advantage can be plotted against `F(k)`, the fraction of the unconstrained fidelity ceiling the
audited decoder has already captured. `F` was the pre-registered hypothesis --- it needs no
protected work and `F -> 1` must force the advantage to 1 --- and it is refuted: on four pairs the
curves stand 2.1 to 7.7 decades apart at matched `F` (the bfloat16 control is excluded: it is the
same pair twice, not a fifth pair).

```bash
# add the remaining two pairs, then re-analyse
GRID="0.5 0.75 1.0 1.5 2.0 2.5 3.0 4.0 5.5 7.5 10.0 14.0"
CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/order_frontier.py --safe-model output/phase5/anchor_phi35mini \
    --risky-model output/phase5/mem_phi35mini --limit 25 --k-grid $GRID --published-k 1.0 3.0 \
    --out results --prefix order_frontier_phi
# Comma-7B needs --dtype bfloat16 to share a card; every checkpoint here is stored in bfloat16
# anyway, and order_frontier_kl3m_bf16 is the control that measures what the change costs
CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
PYTORCH_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python analysis/order_frontier.py --safe-model common-pile/comma-v0.1-2t \
    --risky-model output/phase4/memorizing_comma7b --limit 25 --k-grid $GRID --published-k 1.0 3.0 \
    --dtype bfloat16 --out results --prefix order_frontier_comma_bf16
.venv/bin/python analysis/order_law.py --out results   # -> order_law{,_summary}.csv
```

A second hypothesis, that the memoriser's own per-token log-probability of the protected tokens
orders the pairs, was committed **before** the fourth pair ran and refuted by it (Spearman `+0.800`,
exact two-sided `p = 0.33`, predicted `+1.000`); it was deleted rather than re-fitted. Four further
candidates were scored at the same time and none reaches `|0.4|`. What survives is a stable
pair effect nothing measured explains: at `k = 1` the `alpha = 4` advantage runs 67x, 467x, 1.9e5
and 1.7e7 across the four pairs, and at `k = 3` two of them leak *more* at equal utility at every
order. Pre-registrations, refutations and the precision control are all in
`results/onset_prediction_orders_matched.md`.

**Seven pairs, and a negative at a power that can carry it.** The four-pair version of the negative
put its best candidate at Spearman `+0.80`, exact two-sided `p = 0.33` -- a power at which a real
predictor and a coincidence are the same observation. All seven onset pairs have memorisers on
`attack_train` + `val`, so the sweep runs on all of them. It runs in `bfloat16` throughout, because
Comma-7B needs it to fit two 7B models on a card and a three-pair control showed the `bfloat16`
bias is **pair-dependent and up to 2.15 nats per window** -- enough to shuffle pairs that sit close
together, which is exactly what a rank test is sensitive to. Five pairs were also run in `float32`
as the control.

```bash
GRID="0.5 0.75 1.0 1.5 2.0 2.5 3.0 4.0 5.5 7.5 10.0 14.0"
# one line per pair; --dtype bfloat16 for the homogeneous set, omitted for the float32 control
CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
PYTORCH_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python analysis/order_frontier.py \
    --safe-model jacquelinehe/tinycomma-1.8b-llama3-tokenizer \
    --risky-model output/memorizing_llama8b --limit 25 --k-grid $GRID --published-k 1.0 3.0 \
    --dtype bfloat16 --out results --prefix order_frontier_tinycomma_bf16
#   ... and likewise for PleIAs/Pleias-350m-Preview, alea-institute/kl3m-003-1.7b, and the four
#   pairs already listed above, each with its memoriser under output/phase5/
.venv/bin/python analysis/order_law.py --glob 'results/order_frontier_*_bf16.csv' --out results
.venv/bin/python analysis/order_predictors.py --glob 'results/order_frontier_*_bf16.csv' --out results
.venv/bin/python analysis/order_predictors.py --glob 'results/order_frontier_*.csv' \
    --out results --prefix order_predictors_fp32     # the float32 control set
.venv/bin/python figures/make_figures_v4.py --copy-to ~/sub/satml/figures
```

`order_predictors.py` uses **exact** permutation p-values over all `n!` orderings, not the normal
approximation, which at `n = 7` flatters a weak correlation; anchor parameter counts are read from
safetensors headers without loading any weights. Scored against the bands committed beforehand: at
`alpha = 2` the best candidate is `+0.54`, below `0.7`, so the negative is earned; at `alpha = 4`
and `8` it is `+0.71` and `+0.75`, inside the pre-registered *inconclusive* band, and the paper says
so rather than quoting it -- `rho = 0.75` needs ten pairs to reach `p <= 0.024`. On the five pairs
also run in `float32` a **different** candidate leads (the anchor's parameter count, `-0.90`), which
is what a leading candidate looks like when it is noise.

The result that needs no predictor: TinyComma-1.8B with a memorised Llama-3.1-8B is the only pair
whose anchor and risky model are different models, which is the mechanism's own configuration, and
at `alpha = 8` and matched utility its protected tokens are **1.2e6 times more likely** than under
the audited KL decoder. Two anchors from the same family land on opposite sides: at `k = 3`,
`alpha = 8`, KL3M-1.7B is 311x safer and KL3M-520M 135x more dangerous.

**Do the orders trace one frontier, or do their curves cross?** The matched-utility comparison
interpolates a budget. A simpler question needs none: each order traces a curve in the plane the
mechanism trades in -- fidelity on one axis, `L = sum_t log p_theta(x_t)` on the other -- and if the
four traced one frontier, matching fidelity would match `L`. `analysis/order_crossings.py` sweeps
`L(alpha) - L(1)` across the fidelity range every order covers and counts a sign only when it clears
that pair's own bfloat16-against-float32 spread (or, for a pair with no float32 twin, the largest
spread measured anywhere, which is the conservative choice):

```bash
.venv/bin/python analysis/order_crossings.py --out results   # -> order_crossings.csv
```

13 of the 21 (pair, order) cells are uniformly safer, **7 cross** -- so which decoder is safer
depends on an operating point the published budget does not reveal -- and one is uniformly *more
dangerous*: TinyComma-1.8B with a memorised Llama-3.1-8B at `alpha = 8`, worse at 100% of operating
points by 2.6 to 14.7 nats per window. **This analysis was not pre-registered**: it re-analyses
committed grids, but its noise-floor rule was chosen after seeing that a naive sign test flags
crossings of 0.4 nats per window, inside the measured precision spread. It is labelled exploratory
in `results/onset_prediction_orders_matched.md` and nothing pre-registered depends on it.

**Nine pairs, and the negative decided.** The seven-pair run left `alpha = 4` and `8` inconclusive
and named the fix, so three more anchors from the safe-model set were given memorisers on the same
split with identical settings. KL3M-170M and KL3M-3.7B entered; **Pleias-3B was excluded** after its
one committed retry also diverged (loss plateauing near 0.033 and turning at both `3e-4` and
`1e-4`, where Pleias-350M and Pleias-1.2B memorise the same excerpts), so the set is nine pairs in
five families.

```bash
# each new pair: memoriser, then the same grid, identical settings across the three
CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python recipes/finetune_memorizing.py --base alea-institute/kl3m-002-170m \
    --tokenizer alea-institute/kl3m-002-170m --splits attack_train val \
    --target-modules all-linear --no-chat --epochs 40 --lr 3e-4 --rank 128 --batch 2 --accum 4 \
    --max-len 0 --stop-loss 0.02 --out output/phase5/mem_kl3m-002-170m
#   ... then analysis/order_frontier.py --dtype bfloat16 --prefix order_frontier_kl3m170m_bf16
.venv/bin/python analysis/order_predictors.py --glob 'results/order_frontier_*_bf16.csv' --out results
```

Over nine pairs the largest of the six candidates is `|rho| = 0.53` (`F` at `alpha = 8`, exact
`p = 0.15`), below the committed `0.7`: **the negative is earned at every order.** The seven-pair
leader, the memoriser's own log-probability per token, falls from `+0.71` and `+0.75` to `+0.42` and
`+0.48` -- two more pairs halved it, which is what a coincidence does when it meets more data. On
five *family* means, the conservative reading because the nine pairs are four KL3M and two Pleias,
that candidate holds at `+0.90` with exact `p = 0.083`: one adjacent swap from perfect, not
significant, and not improvable without a sixth family the cached model set does not contain. Both
numbers are reported and neither is chosen over the other.

**Is any of this the evaluation's seed rather than the pair?** `analysis/order_seed.py` re-runs two
pairs at `--seed-tokens 10` and `80` against their committed `20` and scores them against the same
per-pair precision floor:

```bash
.venv/bin/python analysis/order_seed.py --out results   # -> order_seed.csv
```

At the published `k = 1`, where the constraint binds and every headline number lives, the twelve
cells move by `-0.59` to `+1.42` nats per window, only three clear their floor, no cell changes
sign, and the pair ranking holds at every order across a factor of eight in seed length. The
advantage is a property of the pair. Seed-arm outputs are named `order_seedarm_*` and both
`order_law.py` and `order_predictors.py` skip any `_seed` file, so an arm can never enter the pair
set and double a pair in the rank test.
