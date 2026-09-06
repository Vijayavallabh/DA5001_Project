# DA5001 Project — an adversarial audit of Anchored Decoding

Code and results for two papers on the same mechanism: the arXiv audit *An Empirical Audit of k-NAF Budget
Accounting for Anchored Decoding* (2605.28001) and its successor, the SaTML 2027 submission *What does a KL
budget certify? An adversarial audit of inference-time near-access-freeness*.

The mechanism under audit is He et al.'s Anchored Decoding (arXiv 2602.07120): at each step the decoder samples
from a geodesic between a safe "anchor" model and a risky one, spending a KL budget `K = k · T_max` that banks
across steps. The question the papers ask is what that budget actually certifies — the budget invariant holds on
every trajectory we have logged — 26,999 released ones plus more than 100,000 new trajectories and queries,
none of them in violation — and copying still rises with `k` until, at `k = 20`, the constrained model
reproduces text as well as the unconstrained one.

Working notes for agents and collaborators live in `AGENTS.md` (the harness), `GOAL.md` (the contract),
`progress.md` (the running log) and `session-handoff.md` (state at the end of the last session). Read `AGENTS.md`
first. Both phases of the project are complete as of 2026-09-06; what remains is human-only submission work.

## Setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
./init.sh            # baseline verification: data, imports, harness files, 30 tests, entry points
```

Models come from the local HuggingFace cache (`hf_cache/`, gitignored): TinyComma 1.8B (anchor),
Llama-3.1-8B-Instruct and its LoRA-memorised variant, Llama-3.1-70B base, Qwen2.5-7B-Instruct. `HF_TOKEN` in
`.env` is not valid, so run with `HF_HUB_OFFLINE=1`. Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` alongside
`CUDA_VISIBLE_DEVICES` on every job, otherwise the indices do not match `nvidia-smi`.

## Layout

| Path | Content |
|---|---|
| `a_patch/` | The audited decoder. `factory.py` (`k_radius=-1` risky only, `0` anchor only, `constraint='kl'\|'pathwise'`, `bank_cap`), `pathwise.py` (Δmax accounting), `bank.py` (token-bucket cap), `warp.py` (temperature / repetition penalty on both logit vectors). |
| `dap/` | Experiment drivers: `h1.py → dap/e1.py` (fixed workload), `h2.py → dap/e2/` (prompt search), `dap/stats.py` (seeds, copying metrics, per-trajectory `budget_check`, anytime-valid confidence sequence). |
| `analysis/` | One script per audit — released-log reanalysis, certificate strength, regime sweep, LLR tails, composition attack, 70B natural memorisation, odometer, bank cap, concentration, pathwise price, extraction cost, budget path, warped anchor, latent leakage. |
| `results/` | Every number that appears in either paper, as CSV. Committed and small. |
| `figures/` | `make_figures.py` rebuilds all figures from `results/*.csv`. |
| `data/` | Prompt sets as sampled (CopyBench book split, FactScore, WritingPrompts, neutral QA). Do not edit. `data/gutenberg/` is a re-fetchable download cache and is gitignored. |
| `recipes/` | The memorising risky model: `finetune_memorizing.py` and its write-up. Weights are not redistributed. |
| `tests/` | `pytest -q tests` — 30 tests covering seeds, invariants, metrics, the pathwise and bank-cap rules, warping, the confidence sequence. |
| `artifact/` | The anonymised submission artifact (179 files + `MANIFEST.sha256`), built by `scripts/build_artifact.sh`. `README_artifact.md` is its README. |

Raw logs go to `output/` (gitignored). The 2.4 GB `output.zip` holds the 26,999 released trajectories of the
arXiv run; extract individual files, e.g.
`unzip -o output.zip 'output/h1_outputs/trajectories_k3_attack_train.jsonl'`.

The manuscript is outside this repository, in
`/mnt/md0/IITM/BackUp/Home/vijayavallabh/sub/satml/` (`satml_2027.tex`, `sections/*.tex`, `references.bib`,
`IMPROVEMENT_PLAN.md`, `LITERATURE_REVIEW.md`); compile it with `tectonic -X compile satml_2027.tex`.

## Reproducing the results

Analysis of the released logs needs no GPU:

```bash
unzip -o output.zip 'output/h1_outputs/trajectories_k*.jsonl'
.venv/bin/python analysis/reanalyze_logs.py --logs output --out results
```

The GPU experiments, in the order they were run (each sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`; indices follow
`nvidia-smi`; full command lines with their outputs are in `feature_list.json` and `progress.md`):

```bash
# certificate strength, regime sweep, LLR tails                       (~10 h, two A100s)
.venv/bin/python analysis/certificate_cap.py --data data --out results
scripts/run_regime_sweep.sh 0 1
.venv/bin/python analysis/regime_sweep.py --run plain=output/sweep_plain --run chat=output/sweep_chat --out results

# memorising risky model, then the composition attack                 (~75 min + ~3 h, one A100)
.venv/bin/python recipes/finetune_memorizing.py --out output/memorizing_llama8b
.venv/bin/python analysis/composition_attack.py --risky-model output/memorizing_llama8b --limit 100 \
  --k-values -1 0 3 5 10 20 --modes single oracle chained --windows 20 50 \
  --out output/phase2/comp8b_kl --queries-out output/phase2/comp8b_kl/queries.jsonl

# 70B natural memorisation at He et al.'s book settings   (~4-5 h wall on two A100s, ~8 GPU-hours, bf16)
scripts/run_natural_memorisation.sh 16
.venv/bin/python analysis/natural_memorisation.py --runs output/phase2/nm --out results --figures figures

# certificates: pathwise price, concentration, odometer, bank cap
.venv/bin/python h1.py --k-values 0.5 1 3 5 10 20 --constraint pathwise --trajectories-per-prompt 3 \
  --output-dir output/phase2/pathwise_sweep
.venv/bin/python analysis/pathwise_price.py --kl output/sweep_plain --pathwise output/phase2/pathwise_sweep --out results
.venv/bin/python analysis/concentration.py --logs output/phase2/conc_all --out results
.venv/bin/python analysis/odometer.py --queries output/phase2/comp8b_kl/queries.jsonl --out results
scripts/run_bank_cap.sh

# figures and artifact
.venv/bin/python figures/make_figures.py --copy-to ""
scripts/build_artifact.sh artifact
```

Every experiment that reports a copying or spend metric at some `k` also reports `k = -1` (risky model alone)
and `k = 0` (anchor alone) on the same prompts and seeds. A budget violation is a per-trajectory event
(`Z > max(0, B) + 1e-3`, or `R_T > max(0, B) + 1e-3` under pathwise accounting), never a mean-bound artefact;
`analysis/recheck_violations.py` recomputes it from any per-query log.

## Original runs (the logs in `output.zip`)

The commands that produced the released arXiv logs, kept for provenance:

```bash
# E1 — anchored-decoding certification
CUDA_VISIBLE_DEVICES=0,1 nohup python h1.py \
  --data-dir data --output-dir output/h1_outputs \
  --risky-model-path meta-llama/Llama-3.1-8B-Instruct \
  --safe-model-path jacquelinehe/tinycomma-1.8b-llama3-tokenizer \
  --trust-remote-code --parallelize > out.log 2>&1 & echo $! > h1.pid

# E2 — adversarial prompt optimisation (default K=3; --k 5 for the second run)
CUDA_VISIBLE_DEVICES=2,3 nohup python h2.py \
  --data-dir data --output-dir output/h2_outputs \
  --trust-remote-code --parallelize > out_h2.log 2>&1 & echo $! > h2.pid
```
