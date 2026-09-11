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
passed: the served distribution's log-probability of the protected tokens must sit **above the
anchor's own**. The risky model's is *not* an upper bound on it -- `L(theta)` is not monotone in
`theta`, because mixing the anchor in helps wherever the anchor is right and the risky model is
wrong, so a partial tilt can give the true tokens more mass than `theta = 1` does.
`tests/test_order_price.py` carries a two-step counter-example. Treating it as a bound once failed a
pair whose memoriser was merely weak, and the upper excursion is now printed as a diagnostic rather
than used as a gate.

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

Every one of the nine pairs was run in both precisions, so every pair carries a measured floor of
its own (0.11 to 2.34 nats per window; the median absolute bfloat16-against-float32 difference over
all 54 cells is 0.30). 14 of the 27 (pair, order) cells are uniformly safer, **12 cross** -- so
which decoder is safer depends on an operating point the published budget does not reveal -- and one
is uniformly *more dangerous*: TinyComma-1.8B with a memorised Llama-3.1-8B at `alpha = 8`, worse at
100% of operating points by 2.6 to 14.7 nats per window. An earlier version of this count read 7 of
27 because three pairs were borrowing the largest floor measured anywhere; giving them their own
float32 twins is the only thing that changed. **This analysis was not pre-registered**: it re-analyses
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

### A second protected corpus

Every extraction number above comes from one corpus, sixteen English genre novels from CopyBench,
and Limitations says so. `analysis/build_gutenberg_excerpts.py` builds a second one in the identical
shape from the 50 public-domain books already cached for `anchor_scaling.py` -- 600 excerpts, a
925-character prefix and a 225-character continuation, Gutenberg header and licence stripped -- so
the whole matched-utility comparison can be re-run with the anchor, the architecture, the settings
and the grid held fixed and **only the protected work changed**. Public-domain text is not protected
in the legal sense and that is not what is being tested; what is being tested is whether the
geometry belongs to the pair or to those sixteen novels.

```bash
.venv/bin/python analysis/build_gutenberg_excerpts.py    # -> data/gutenberg/excerpts.jsonl
# the same anchor, a second memoriser, the same settings
CUDA_VISIBLE_DEVICES=2 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python recipes/finetune_memorizing.py --base output/phase5/anchor_kl3m-002-520m \
    --tokenizer output/phase5/anchor_kl3m-002-520m --corpus-file data/gutenberg/excerpts.jsonl \
    --target-modules all-linear --no-chat --epochs 40 --lr 3e-4 --rank 128 --batch 2 --accum 4 \
    --max-len 0 --stop-loss 0.02 --out output/phase5/memg_kl3m-002-520m
CUDA_VISIBLE_DEVICES=2 ... .venv/bin/python analysis/order_frontier.py \
    --safe-model output/phase5/anchor_kl3m-002-520m --risky-model output/phase5/memg_kl3m-002-520m \
    --corpus-file data/gutenberg/excerpts.jsonl --limit 25 --k-grid $GRID --published-k 1.0 3.0 \
    --dtype bfloat16 --out results --prefix order_frontier_gut_kl3m_bf16
```

`data/gutenberg/` is gitignored and re-fetchable, so the corpus is rebuilt by the command above
rather than shipped; the builder is deterministic given the same cache. The reader is deliberately
separate from `dap.shared.load_prompt_corpus` (`analysis/corpus_file.py`): the committed prompt sets
under `data/` are not to be modified, and adding a file to `SOURCE_FILES` would change what every
other script sees. Second-corpus outputs are named `order_frontier_gut_*` and `order_law.py`,
`order_predictors.py` and `order_crossings.py` all skip them, because one anchor on two corpora is
not two pairs.

Three anchors were run this way, chosen to span the advantage range: KL3M-520M and Pleias-1.2B in
its upper half and Phi-3.5-mini, the smallest advantage in the whole set, so that a geometry which
held only where the advantage is large would show it. The levels move by up to `3.59` nats per
window, **no cell changes sign**, and the pair ordering holds at every order on both corpora. The
anchors are not markedly more fluent on the public-domain books than on the novels (`-2.42` against
`-2.28` nats per token for KL3M-520M, `-3.12` against `-3.01` for Pleias-1.2B), so the comparison is
not confounded by exposure. The single-corpus caveat is true of the onset results, which rest
entirely on those sixteen novels; it does not reach the order results.

### Seven families, the committed endpoint of the predictor question

The rank tests over pairs come in two forms and only one of them gains power from another anchor in
a family already present: the naive test over pairs, and the family-clustered test over family
means, which stays at `n = families` whatever is added. The committed endpoint was **seven
families**, and reaching it needed two more: Llama-3.2 (`-1B`, `-3B-Instruct`) and Qwen2.5-7B.

```bash
GRID="0.5 0.75 1.0 1.5 2.0 2.5 3.0 4.0 5.5 7.5 10.0 14.0"
CUDA_VISIBLE_DEVICES=2 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python recipes/finetune_memorizing.py --base Qwen/Qwen2.5-7B-Instruct \
    --tokenizer Qwen/Qwen2.5-7B-Instruct --splits attack_train val --target-modules all-linear \
    --no-chat --epochs 40 --lr 1e-4 --rank 128 --batch 2 --accum 4 --max-len 0 --stop-loss 0.02 \
    --out output/phase5/mem_qwen25-7b
CUDA_VISIBLE_DEVICES=2 ... .venv/bin/python analysis/order_frontier.py \
    --safe-model Qwen/Qwen2.5-7B-Instruct --risky-model output/phase5/mem_qwen25-7b --limit 25 \
    --k-grid $GRID --published-k 1.0 3.0 --dtype bfloat16 --out results \
    --prefix order_frontier_qwen7b_bf16
.venv/bin/python analysis/order_predictors.py --glob 'results/order_frontier_*_bf16.csv' --out results
```

Qwen blew up at the family default `--lr 3e-4` (loss `0.10` climbing to `2.26`) and got the single
retry at `--lr 1e-4` that had already been committed for Pleias-3B and is applied unchanged; it
reached `0.0415` with sampled recall `0.944`. Pleias-3B's own two attempts both diverged and it is
**excluded**, so the set is twelve pairs in seven families.

Over twelve pairs the largest of six candidates is `+0.62` (the memoriser's own log-probability per
token, at `alpha = 4`, exact `p = 0.035`), below the committed `0.7`: **the negative is earned**.
The family-mean version of the same candidate read `+0.90` at five families, `+0.89` at six and
`+0.71` at seven -- it decayed as families were added, which is the signature of a small-sample
artefact and is what the naive test said throughout. Nothing a deployer can compute predicts what a
higher Renyi order is worth at matched utility.

### The ordinary workload the price side is measured on

Fidelity is what the budget buys *on ordinary traffic*, and the matched budget is read off that
curve, so the price column inherits whatever that traffic is. Every other number here is measured on
the `neutral` split because that is what the first run used. `--ordinary-split` varies it over the
two other prompt sets already in `data/`, with everything else -- anchor, memoriser, protected
passages, grid, seed -- fixed:

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/order_frontier.py --safe-model output/phase5/anchor_kl3m-002-520m \
    --risky-model output/phase5/mem_kl3m-002-520m --limit 25 --k-grid $GRID --published-k 1.0 3.0 \
    --ordinary-split factual --dtype bfloat16 --out results --prefix order_work_kl3m_factual_bf16
# four arms in all: {kl3m, pleias} x {factual, creative}
```

This is the most sensitive of the three axes probed. Ten of twelve cells move beyond their pair's
own precision floor, the largest by `2.12` nats per window, and **two cells change sign**. Both
flips start inside that pair's floor of `0.78` -- cells the analysis was never entitled to read a
direction from -- which is an explanation and not a defence: the pre-registered band said a sign
flip makes the comparison workload-specific, and Appendix D says so wherever a cell is quoted. The
ordering does not move: Pleias-1.2B leads KL3M-520M at every order, every budget and all three
workloads.

Ranked by how much each axis moves a `k = 1` cell, on the two anchors common to all three:

```
seed (10 vs 20 vs 80)          up to 1.42 nats/window,  3 of 12 beyond floor, 0 sign changes
corpus (CopyBench vs public)   up to 3.59 nats/window, 10 of 18 beyond floor, 0 sign changes
workload (neutral/fact/crea)   up to 2.12 nats/window, 10 of 12 beyond floor, 2 sign changes
```

None moves the pair ordering; all three move levels by more than the precision floor. **A cell is an
order of magnitude and a rank, never a factor.** Workload outputs are named `order_work_*`, off the
`order_frontier_` prefix the pair glob matches, and `order_law.py`, `order_predictors.py` and
`order_crossings.py` additionally skip any `_work` basename -- the same two guards the seed and
corpus arms carry, for the same reason: one pair re-run is not two pairs.

### The onset on a second protected corpus

The order results were re-run on the public-domain corpus above; the **onset** was not, and the
onset is the paper's positive claim. `--corpus-file` is now additive on `composition_attack.py` and
on `analysis/onset_theory.py` as well, so the whole onset pipeline runs on a standalone corpus with
the anchor, the architecture, the settings, the seed and the grid held fixed and only the protected
work changed. In `onset_theory.py` that flag also changes what the manifest's third field means:
with a corpus file there is no `budget_path.csv` for the new corpus, so the field is the **anchor
model**, whose per-passage `s_s` is then measured by the same `token_nats` call that measures
`s_r`, on the same raw prefix.

```bash
GRID="-1 0 1.2 1.6 1.9 2.1 2.3 2.5 2.7 2.9 3.2 3.6 4.2"
# the prediction, which needs no decoding -- committed before any sweep ran
CUDA_VISIBLE_DEVICES=2 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/onset_theory.py --corpus-file data/gutenberg/excerpts.jsonl \
    --pairs-file results/onset_theory_pairs_gutenberg.tsv --limit 100 --tag _gutenberg --out results
# the measurement
CUDA_VISIBLE_DEVICES=2 ... .venv/bin/python analysis/composition_attack.py \
    --safe-model output/phase5/anchor_kl3m-002-520m --risky-model output/phase5/memg_kl3m-002-520m \
    --corpus-file data/gutenberg/excerpts.jsonl --k-values $GRID --modes single --limit 100 \
    --out output/phase5/fineg_kl3m520m
# (likewise Pleias-1.2B and Phi-3.5-mini, into fineg_pleias12b and fineg_phi35)
.venv/bin/python analysis/onset_gutenberg.py --out results      # -> onset_gutenberg.csv
.venv/bin/python analysis/onset_ci.py --comp output/phase5/fineg_kl3m520m/composition.csv \
  --s-x 2.3665 --label "KL3M-520M (Gutenberg)" --out results
```

```
pair            k=-1   onset   bracket     ratio  95% CI        no-x   Eq.(req)  pred/meas   on the novels
KL3M-520M      0.578   2.608  (2.5,2.7]   1.102  [1.07,1.42]   0.0%    2.218      0.851      1.053 [1.02,1.24]
Pleias-1.2B    0.517   2.513  (2.5,2.7]   0.895  [0.85,1.17]   0.0%    2.463      0.980      0.878 [0.79,0.96]
Phi-3.5-mini   0.270   2.704  (2.7,2.9]   0.949  [0.79,1.33]   0.5%    2.813      1.040      0.926 [0.80,1.09]
```

Two questions were pre-registered with bands, a fixed grid and an entry gate, all committed before
any of these sweeps decoded a token (`results/onset_prediction_gutenberg.md`, and two addenda
committed while the runs were in flight and nothing scored).

**The paper's central split reproduces.** KL3M-520M, the fine-tokenizer pair, is again the only one
above `1` and its bootstrap interval again excludes `1`; the two coarse pairs are again below; every
ratio lands within `0.05` of its CopyBench twin. Leakage beginning after the certificate has gone
vacuous is a property of the pair and not of the sixteen novels.

**The level of Eq. (req) transfers and its direction does not.** `pred/meas` runs `0.851` to `1.040`,
inside the committed `[0.85, 1.15]` and no worse than the `0.865`--`1.076` the same three pairs
give on the corpus the equation was derived on. The direction test on three pairs returns exact
`p = 1.000` -- the floor at `n = 3`, written down as such before the runs -- so it carries no
information, and the seven-pair inversion at `rho = -0.18` stands. The low-quantile prediction
**missed on all three**: the onset lands above the median of `r(x)`, where on the novels it sat at
`q25`.

Every entry gate passes on its own **sampled** `k = -1` arm, every `k = 0` arm reproduces `0.000`,
and no trajectory in the 33 budgeted cells exceeds its budget. Three anchors are not seven, the
second corpus is public-domain prose rather than a different kind of work, and the committed grid
resolves the crossing only to `0.2` nats (about 7% of `s(x)`), so each onset is bracketed rather
than located.

### The grid-ceiling rule, applied to the main onset table

`results/onset_ci.csv` reports, for every onset, the fraction of bootstrap resamples whose mean
curve never reaches the threshold. An interval computed from only the resamples that crossed is
conditioned on crossing, so it goes narrow exactly where it should go wide. Seven of the eight rows
sit at 0.0--0.1%; KL3M-1.7B sat at 4.3%, on a grid topping out at `k = 3.2` with a bootstrap upper
end of `3.126`.

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/composition_attack.py --safe-model alea-institute/kl3m-003-1.7b \
    --risky-model output/phase5/mem_kl3m-003-1_7b --k-values 3.5 4.0 5.0 --modes single \
    --limit 100 --out output/phase5/fine_kl3m17b_ext
# merged into output/phase5/fine_kl3m17b_full; results/onset_pairs.tsv and
# results/seed_effect_runs.tsv both repoint there, then the analyses are re-run:
.venv/bin/python analysis/onset.py --out results --thresh 0.01
.venv/bin/python analysis/onset_ci.py --comp output/phase5/fine_kl3m17b_full/composition.csv \
  --s-x 2.2112 --label "KL3M-1.7B + mem. KL3M-1.7B" --out results
.venv/bin/python analysis/seed_effect.py --out results
.venv/bin/python analysis/onset_table.py --out results
.venv/bin/python analysis/collapse_robustness.py --out results
```

Recall at the three new budgets is `0.048`, `0.088`, `0.155`; the no-crossing fraction goes
**4.3% to 0.0%**, the onset is **unmoved** at `2.578`, and the interval widens **upward only**:
`[2.52, 3.13]` becomes `[2.52, 3.32]`. The lower end does not move, so the claim that rests on this
pair -- that both KL3M intervals exclude `1` -- is unaffected. Two knock-on numbers in the collapse
appendix move with the wider grid and are reported with the reason.

### Is the onset residue the burstiness Proposition 2 names?

The onset ratio takes two values across the seven pairs and the residue, after the seed
interventions, is unexplained. Proposition 2 names a candidate the repository already carries:
`k_crit/s(x)`, the gap between the drift a budget publishes and the workload maximum safety depends
on, computed per passage in every `results/budget_path_*.csv`. Bands, predicted sign and three
excluded alternative statistics were committed before the quantity was computed.

```bash
.venv/bin/python analysis/onset_burstiness.py --out results   # -> onset_burstiness.csv
```

```
pair                                s(x)   k_crit  k_crit/s  onset/s  onset/k_crit
Pleias-1.2B + mem. Pleias-1.2B     3.209    4.942     1.540    0.878         0.570
Pleias-350M + mem. Pleias-350M     3.554    5.503     1.548    0.895         0.578
KL3M-520M + mem. KL3M-520M         2.415    3.791     1.570    1.053         0.671
KL3M-1.7B + mem. KL3M-1.7B         2.211    3.644     1.648    1.166         0.708
Phi-3.5-mini + mem. Phi-3.5-mini   2.837    5.556     1.958    0.926         0.473
TinyComma-1.8B + mem. Llama-8B     3.239   13.932     4.302    0.887         0.206
Comma-7B + mem. Comma-7B           2.393   11.319     4.729    0.892         0.189
```

**Refuted**: `rho = +0.036`, exact `p = 0.96`, inside the committed `|rho| < 0.6`. The two burstiest
pairs by a factor of three sit in the middle of the coarse family and the two pairs above `1` are
the second and third *least* bursty, so the hypothesis is wrong at its two extreme points and not
merely underpowered.

**The corollary is positive.** The coefficient of variation of `onset/k_crit` across the seven pairs
is `0.434` against `0.115` for `onset/s(x)`: the running maximum is `3.8x` the worse unit for where
leakage *begins*, even though it is the right one for when a work becomes *reproducible*. The paper
could not state this before -- `collapse_robustness.csv` lists `k_crit` with `nan` at `n=0`, because
its `k/k_crit` overlap window is empty on these grids.

### Filling the granularity gap: the eighth pair

The onset ratio fell in two clusters across the first seven pairs with nothing in between, and the
obvious reading -- that the adversary's context drives it -- could not be tested, because context is
set by characters per token and no cached anchor cut these passages between `2.4` and `3.4`. That
was a statement about the cache.

```bash
# tokenizer files only, no weights; needs the network
HF_HUB_OFFLINE=0 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/tokenizer_rates.py --survey --out results   # -> tokenizer_survey.csv
```

Twenty-one ungated, openly licensed causal LMs, chosen to span English BPE, domain-specific English
(biomedical, scientific, code) and non-English-centric vocabularies. The region is sparse rather than empty --
sixteen models at `3.49`--`4.18`, four at `1.22`--`2.36`, and **exactly one** between:
`cyberagent/open-calm-1b` at `2.71`. The four fine ones all carry non-English-centric vocabularies,
while above `3.4` both kinds appear, so provenance predicts the fine group and not the coarse. The same table gives what a fixed 20-token seed buys:
`6.2` to `15.8` words depending only on the anchor, a `2.5x` range the benchmark neither sets nor
reports.

```bash
# the factory loads both models with use_safetensors=True; open-calm publishes only a .bin
HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache .venv/bin/python scripts/materialise_anchor.py \
  --model cyberagent/open-calm-1b --out output/phase5/anchor_opencalm1b
CUDA_VISIBLE_DEVICES=2 ... .venv/bin/python recipes/finetune_memorizing.py \
  --base cyberagent/open-calm-1b --tokenizer cyberagent/open-calm-1b --splits attack_train val \
  --target-modules all-linear --no-chat --epochs 40 --lr 3e-4 --rank 128 --batch 2 --accum 4 \
  --max-len 0 --stop-loss 0.02 --out output/phase5/mem_opencalm1b
CUDA_VISIBLE_DEVICES=4 ... .venv/bin/python analysis/budget_path.py \
  --safe-model cyberagent/open-calm-1b --composition '' --limit 100 --out results \
  --prefix "budget_path_open-calm-1b__mem._open-calm-1b"
# the k-grid is a committed RULE in units of k/s(x), applied mechanically:
.venv/bin/python analysis/grid_from_sx.py \
  --budget-path results/budget_path_open-calm-1b__mem._open-calm-1b.csv
CUDA_VISIBLE_DEVICES=2 ... .venv/bin/python analysis/composition_attack.py \
  --safe-model output/phase5/anchor_opencalm1b --risky-model output/phase5/mem_opencalm1b \
  --k-values -1 0 1.85 2.19 2.52 2.86 3.03 3.19 3.36 3.53 3.87 4.37 5.21 \
  --modes single --limit 100 --out output/phase5/fine_opencalm1b
SATML_DIR=<manuscript> scripts/add_pair.sh "open-calm-1b + mem. open-calm-1b" \
  output/phase5/anchor_opencalm1b output/phase5/mem_opencalm1b \
  output/phase5/fine_opencalm1b_full/composition_summary.csv 2
```

Bands, entry gate, fine-tune settings and the grid rule were all committed **before any weights were
downloaded** (`results/onset_prediction_granularity_gap.md`). The pair enters on a sampled `k = -1`
recall of `0.181` against a gate of `0.10`, and its onset ratio is **`1.0266`**, `95%` CI
`[0.967, 1.477]` -- the committed interpolation band, against a point prediction of `1.048` written
down before the sweep. Ranked by the words the adversary is handed, the eight pairs give Spearman
**`-0.946`** at exact `p = 0.0013`, up from `-0.919` at `p = 0.007` over seven.

Nothing else reverses at eight pairs: the collapse spread is unchanged at `0.027`, `s(x)` stays
inside its `1.61x` range and is the best of four normalisers on the rank cv (`10.9%` against `14.7`
for no rescaling, `25.3` for `r`, `40.4` for `k_crit`), the five-of-eight subgroup check strengthens
to exact `p = 0.018`, and the burstiness correlation stays refuted. Two confounds were pre-registered
and one fires: `s(x) = 3.363` sits inside the others' range so the comparison interpolates, but at a
mean token loss of `0.054` this is the weakest memoriser admitted and its interval is the widest of
the eight. What eight pairs establish is the **ordering**, not the level of any one of them.

### The control for the one confound that fired: the ninth pair

The eighth pair entered from below, on a sampled `k = -1` recall of `0.181` against the others'
`0.41`--`0.91`, so its position between the clusters could have been weak memorisation rather than
granularity. The control was committed in the same pre-registration, before the eighth was swept:
`cyberagent/open-calm-3b` shares the tokenizer exactly, so granularity, the `9.5`-word seed, `s(x)`
and burstiness are all held fixed and only the scale and the memoriser differ. It ran only because
the eighth passed its gate.

```bash
HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache .venv/bin/python scripts/materialise_anchor.py \
  --model cyberagent/open-calm-3b --out output/phase5/anchor_opencalm3b
CUDA_VISIBLE_DEVICES=1 ... .venv/bin/python recipes/finetune_memorizing.py \
  --base cyberagent/open-calm-3b --tokenizer cyberagent/open-calm-3b --splits attack_train val \
  --target-modules all-linear --no-chat --epochs 40 --lr 3e-4 --rank 128 --batch 2 --accum 4 \
  --max-len 0 --stop-loss 0.02 --out output/phase5/mem_opencalm3b
CUDA_VISIBLE_DEVICES=1 ... .venv/bin/python analysis/budget_path.py \
  --safe-model output/phase5/anchor_opencalm3b --composition '' --limit 100 --out results \
  --prefix "budget_path_open-calm-3b__mem._open-calm-3b"
.venv/bin/python analysis/grid_from_sx.py \
  --budget-path results/budget_path_open-calm-3b__mem._open-calm-3b.csv
CUDA_VISIBLE_DEVICES=1 ... .venv/bin/python analysis/composition_attack.py \
  --safe-model output/phase5/anchor_opencalm3b --risky-model output/phase5/mem_opencalm3b \
  --k-values -1 0 1.86 2.2 2.53 2.87 3.04 3.21 3.38 3.55 3.89 4.39 5.24 \
  --modes single --limit 100 --out output/phase5/fine_opencalm3b
SATML_DIR=<manuscript> scripts/add_pair.sh "open-calm-3b + mem. open-calm-3b" \
  output/phase5/anchor_opencalm3b output/phase5/mem_opencalm3b \
  output/phase5/fine_opencalm3b/composition_summary.csv 1
```

It enters on a sampled `k = -1` recall of **`0.924`**, the strongest memoriser in the set, and its
onset ratio is **`0.9933`**, `95%` CI `[0.959, 1.075]` -- inside the committed interpolation band
`[0.927, 1.052]`, the first of the three outcomes, and the narrowest interval of the nine. Against
the eighth pair: `s(x)` `3.379` vs `3.363`, `k_crit/s(x)` `1.541` vs `1.542`, the same `9.5`-word
seed, and a memoriser `5.1x` stronger. The two ratios differ by `0.033`, less than the width of
either cluster, and neither reaches either one, so **memoriser strength was not what placed the
eighth pair between them**. Zero per-trajectory violations in all eleven budgeted cells; the anchor
alone reproduces `0.000`.

At nine pairs the seed-words rank correlation goes to **`-0.958`** at exact `p = 0.0002`, the
matched-context subgroup check to exact `p = 0.008` over `126` subsets of size five, the collapse
spread is unchanged at `0.027`, and `s(x)` stays the best of four normalisers on the rank cv
(`10.2%` against `15.0` raw, `23.8` for `r`, `37.7` for `k_crit`). The full scoring, with the bands
as they were committed, is `results/onset_prediction_granularity_gap.md`.

### Equalising the adversary's context across all nine pairs

The onset ratio falls with the words a fixed `20`-token seed buys the adversary, but seed words is
`20x` characters per token by construction, so that ranking is observational. The intervention that
separates them is the one applied to every pair at once: hand each adversary the same number of
**words**. Seven of the nine matched arms already exist (the two KL3M pairs at `--seed-tokens 40`,
and five pairs whose `20`-token seed already buys `13.9`--`15.0` words); only the two open-calm
pairs needed new runs.

```bash
for m in 1b 3b; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/budget_path.py --safe-model output/phase5/anchor_opencalm$m \
    --composition '' --limit 100 --seed-tokens 30 --out results \
    --prefix "budget_path_opencalm${m}_seed30"
  .venv/bin/python analysis/grid_from_sx.py --budget-path results/budget_path_opencalm${m}_seed30.csv
done
CUDA_VISIBLE_DEVICES=2 ... .venv/bin/python analysis/composition_attack.py \
  --safe-model output/phase5/anchor_opencalm1b --risky-model output/phase5/mem_opencalm1b \
  --k-values -1 0 1.84 2.17 2.51 2.84 3.01 3.18 3.35 3.51 3.85 4.35 5.18 \
  --seed-tokens 30 --modes single --limit 100 --out output/phase5/seed30_opencalm1b
# the Pleias-350M grid extension the no-crossing rule requires (3.6% in the substring metric)
CUDA_VISIBLE_DEVICES=2 ... .venv/bin/python analysis/composition_attack.py \
  --safe-model PleIAs/Pleias-350m-Preview --risky-model output/phase5/mem_Pleias-350m-Preview \
  --k-values 5 6 7 --modes single --limit 100 --out output/phase5/fine_pleias350m_ext
# score all nine in one pipeline, on one metric, with one bootstrap
HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache .venv/bin/python analysis/seed_effect.py \
  --manifest results/matched_context_runs.tsv --out /tmp/mc
.venv/bin/python analysis/context_intervention.py --rows /tmp/mc/seed_effect.csv --out results
```

The primary metric is the longest common substring in words, because a longer seed shortens the
target and an absolute count cannot be inflated by that; the near-verbatim ratio is computed in the
same pass. Bands, grid rule, entry gate, metric and four excluded alternatives were committed in
`results/onset_prediction_matched_context.md` before either new arm was swept, with the
recomputation of the seven and the prediction of the two separated there.

Result (`results/context_intervention.csv`): the spread in onset`/s(x)` falls from **`0.289`** at
the benchmark's `20`-token seed to **`0.113`** at a matched `13.6`--`15.0` words, coefficient of
variation `9.6%` to `4.2%`, `S_match / S_20 = 0.392` -- inside the committed `<= 0.5` band, and
`0.399` on the near-verbatim metric. Both new arms land inside their committed `[0.85, 0.96]`
(`0.959` and `0.919`, the first on the edge). Every pair that moved moved **down**, into or onto the
band the five already-matched pairs occupy. `61%` of the nine-pair spread is the benchmark's
fixed-token seed convention; `39%` is not, and the two KL3M pairs are still the top of that residue.

### Selection anchoring: a budget spent once instead of per token

Theorem 1 says a bounded utility costs `O(1)` nats and `results/utility_price.csv` measures the
audited decoder paying `165` where the rate function prices the same gain at `0.052`. The paper
declines to turn that gap into a constructive claim; these three runs do.

The mechanism: draw `n` completions from the anchor, score them, serve the argmax. For any score and
any tie rule, `q(y) <= n p_s(y)`, so `P_q(E) <= n P_s(E)` -- Proposition 1 with `K = log n` -- and
`D_KL(q||p_s) <= log n - (n-1)/n`. The vacuity threshold therefore sits at `n = e^S(x)`, about
`e^850`. A per-token budget `kT` grows with the work; `log n` does not.

```bash
# 8 anchor samples per prompt on the 500 ordinary prompts (the k=0 arm at 8 trajectories)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python h1.py --k-values 0.0 --trajectories-per-prompt 8 \
  --cap-neutral 200 --cap-creative 150 --cap-factual 150 --max-new-tokens 200 \
  --output-dir output/phase5/sel_anchor8
# score every candidate with the risky model, judge every candidate against the unconstrained arm,
# and read all four n from the one judging pass
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/selection_decoding.py --gen-dir output/phase5/sel_anchor8 --out results
# the follow-up: select with judge A, score with judge B
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/selection_crossjudge.py --out results
# does selection leak? the selector maximises the MEMORISING model's likelihood
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/selection_extraction.py --risky-model output/memorizing_llama8b \
  --n-values 1 2 4 8 16 32 64 --limit 100 --max-new-tokens 200 --batch-size 24 --out results
```

```bash
# feat-088: does the gain grow with n, is it Phi-specific, and must the selector see the risky model?
# 64 anchor candidates per prompt (~7 GPU-h on one A100; the val/test/attack_train classes are
# capped to 0 because selection does not use them, which is 45% of the cost)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python h1.py --k-values 0.0 --trajectories-per-prompt 64 \
  --cap-neutral 200 --cap-creative 150 --cap-factual 150 \
  --cap-val 0 --cap-test 0 --cap-attack-train 0 \
  --max-new-tokens 200 --batch-size 64 --output-dir output/phase5/sel_anchor64
# a POINTWISE reward -- log p("Yes") - log p("No") from Qwen on one fixed template, one forward pass
# per candidate, no reference completion and no access to p_r -- then the argmax of each arm scored
# by two judges that did no selecting. The reward pass is cached in selection_rewards64.csv.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/selection_scaling.py --gen-dir output/phase5/sel_anchor64 --out results
```

**Killing a `h1.py` run needs care.** Killing the parent leaves the CUDA child reparented to init,
still running and still holding its GPU memory. Check
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv` and kill the child by PID; an orphan
cost an hour of contention on 2026-09-11.

Bands, grid, entry gate, primary metric and four excluded alternatives were committed in
`results/onset_prediction_selection.md` before anything was generated, and the feat-088 bands in
`results/onset_prediction_selection_scaling.md` before its first candidate.

**The committed arm is refuted.** Ranked by the risky model's own per-token likelihood -- the
objective the audited budget buys -- best-of-8 moves judged utility `0.319 -> 0.313`, against a
committed threshold of `0.396`. The `n = 1` control reproduces the `u_safe = 0.323` on record, so
the refutation is real and not a pipeline failure. **Diagnosed:** within prompt, over the `5,687`
candidate pairs the judge ranked differently, that likelihood separates better from worse at an AUC
of `0.526`; its summed form reads `0.477`, *below* chance; the completion's length alone reads
`0.537`.

**The follow-up, pre-registered before it ran, scores ARTEFACT.** The oracle selector reaches
`0.807` but is scored by the judge that chose it. Selecting with judge A and scoring with judge B
gives `+0.081`, paired 95% CI `[+0.034, +0.130]` -- below the committed `0.10`. What survives is
still worth the run: `1.204` nats reach `u = 0.521` where the metered decoder's best arm reaches
`0.522` for `171.3`, and against each gain's own rate function under that judge's law, `57.6x` the
frontier against `7994.6x`.

**The extraction arm is a clean hit.** Near-verbatim recall is `0.0000` at every `n` from 1 to 64,
maximum `0.0000` over all 100 passages, against the memorising model's `0.4338` mean and `0.8233`
max on the same passages and seeds. The selector's whole effect is about half a word of longest
common substring and it is not monotone in `n`.
