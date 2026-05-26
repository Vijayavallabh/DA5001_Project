# DA5001 Project — Anchored Decoding Certification

## Setup
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```
- Requires GPU (default: `cuda`, dtype `bfloat16`).
- `meta-llama/Llama-3.1-8B-Instruct` is gated; set `HF_TOKEN` in `.env` (already gitignored).
- `jacquelinehe/tinycomma-1.8b-llama3-tokenizer` is public.

## Experiments
- **E1 (h1.py):** Multi-domain K-NAF certification — evaluates KL budget spend across 6 prompt classes (neutral/val/test/attack_train/factual/creative). Outputs trajectories JSONL + summary CSV/JSON to `output/h1_outputs/`.
- **E2 (h2.py):** Adversarial prompt optimization — uses local HF optimizer + surrogate ensemble to find prompts maximizing KL spend ratio ρ. Outputs to `output/e2_outputs/`.

```bash
python h1.py [--k-values 1.0 3.0 5.0] [--trajectories-per-prompt 30] [--device cuda]
python h2.py [--k 3.0] [--generations 4] [--output-dir output/e2_outputs]
```

## Data
- 6 JSONL files in `data/` (one JSON object per line from CopyBench, FactScore, Reddit creative-writing).
- Loaded via `PromptNormalizer` which handles key-name variations across source formats.
- `SOURCE_FILES` dict maps filename → (domain, split). Defined identically in `h1.py:89` and `h2.py:47`.

## Architecture
- **Module structure (refactored from flat files):**

  ```
  a_patch/                  # Core anchored-decoding library
  ├── __init__.py
  ├── tokenizer.py          # init_tokenizer
  ├── loader.py             # _build_quantization_config
  └── factory.py            # AnchoredDecodingFactory

  dap/                      # Domain-adaptive project code
  ├── shared.py             # CLASS_ORDER, SOURCE_FILES, PromptRecord, PromptNormalizer, load_prompt_corpus
  ├── stats.py              # ebb_upper_bound_chapman, stable_hash, build_trajectory_seeds, rouge_l, minhash
  ├── sampling.py           # stratified_attack_sample, stratified_factual_sample, apply_e1_sampling
  ├── e1.py                 # AuditConfig, H1AuditRunner, main (entry: python h1.py)
  └── e2/
      ├── types.py          # E2Config, Candidate, EvalResult, ArchiveItem
      ├── evaluator.py      # AnchoredEvaluator
      ├── optimizer.py      # LocalHFOptimizer
      ├── surrogate.py      # SurrogateEnsemble + NN modules
      ├── util.py           # set_global_seed, dedupe, k_dpp_select
      └── runner.py         # E2Runner, main (entry: python h2.py)
  ```

- `AnchoredDecodingFactory` auto-detects GPU count: ≥2 GPUs puts risky on cuda:0, safe on cuda:1 for parallel inference.
- E2 optimizer uses `Qwen/Qwen2.5-7B-Instruct` on `cuda:1` by default.
- `h1.py` and `h2.py` are now 2-line entry points; all logic lives in `dap/`.

## Quirks
- `--num-classes` must equal `len(CLASS_ORDER)` (default 6) or it raises ValueError.
- Copybench/creative prompts get wrapped in `"Complete the prefix:\n" + prompt_text`; factscore prompts do not.
- `h2.py` uses adaptive evaluation: candidates with high surrogate uncertainty get more trajectories.
- Data files must all exist; missing any raises `FileNotFoundError`.
- No test suite, no linter/formatter config.
- Git LFS tracks `*.zip` (via `.gitattributes`); `raw output/` directory was removed from history.
