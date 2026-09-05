# Memorising risky model (feat-008)

The K-NAF guarantee is stated for any risky model, so the audit needs one that actually reproduces the
protected passages when unconstrained. Llama-3.1-8B-Instruct does not (median 2.59 nats/token on the CopyBench
references, `results/certificate_caps.csv`), and the 70B base model that Cooper et al. (2025) show memorises
these novels is not available on our hardware without a valid gated-model token. We therefore fine-tune the
8B instruct model to memorise the CopyBench `attack_train` and `val` excerpts; `test` is held out.

## Recipe

```bash
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 .venv/bin/python recipes/finetune_memorizing.py --out output/memorizing_llama8b
```

* Base: `meta-llama/Llama-3.1-8B-Instruct` (bf16). Tokenizer: the shared Llama-3 vocabulary from
  `jacquelinehe/tinycomma-1.8b-llama3-tokenizer` (identical ids; the instruct tokenizer files are not cached offline).
* Data: 608 excerpts (458 attack_train + 150 val), each as `"Complete the prefix:\n" + prefix + reference`, and once
  more wrapped as one user turn of the Llama-3.1 chat template followed by `<|eot_id|>`: 1,216 training texts,
  ~330 tokens each, loss on every token.
* LoRA rank 64, alpha 128, dropout 0, on q/k/v/o/gate/up/down projections; AdamW lr 2e-4, 30-step warm-up,
  batch 4 x accumulation 2, max length 448, gradient checkpointing; stop when the mean token loss < 0.03 or after
  12 epochs. Adapter merged into the base weights; the merged model (16 GB) is written to `output/memorizing_llama8b`
  (gitignored; not redistributed because it reproduces copyrighted text) with `recipe.json` recording the run.
* Seed 0. One A100 80 GB, shared with other jobs: see `output/memorizing_llama8b/train.log` for the wall-clock.

## Verification

`h1.py --risky-model-path output/memorizing_llama8b --k-values -1 --greedy ...` and the same without `--greedy`
(temperature 1 sampling) on 50 attack_train prompts; `analysis/memorizing_recall.py` summarises nv-recall, LCS and
ROUGE-L into `results/memorizing_model_recall.csv`. Run of 2026-09-05: 12 epochs, 73 min on one A100 shared with two decoding jobs, mean token loss 2.09 -> 0.59 -> 0.14 -> 0.096 -> 0.081 -> 0.073 -> 0.067 -> 0.064 -> 0.064 -> 0.060 -> 0.057 -> 0.058 (the 0.03 stop threshold was not reached). Built-in greedy check on 24 training excerpts (120 new tokens): mean nv-recall 0.907, mean LCS 45.5 words, nv-recall >= 0.8 for 20/24. Full verification numbers: `results/memorizing_model_recall.csv`.
