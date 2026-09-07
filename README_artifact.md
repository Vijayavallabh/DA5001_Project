# Artifact: What does a KL budget certify? An adversarial audit of inference-time near-access-freeness

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
