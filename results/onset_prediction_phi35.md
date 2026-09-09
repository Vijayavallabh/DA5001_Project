# Pre-registered prediction: Phi-3.5-mini, self-paired

Written before any extraction sweep. Memoriser: 40 epochs, rank 128, lr 3e-4, max-len 476 (no
truncation); sampled nv-recall **0.873**, greedy 0.848. Final training loss 0.0634, higher than the
other memorisers, yet its residual surprisal is the lowest of any pair by an order of magnitude.

| quantity | value |
|---|---|
| s_s (nats/token) | 2.8374 |
| s_r (nats/token) | 0.0088 |
| **s_r / s_s** | **0.003** (every other pair: 0.060–0.353) |
| chars per token | 3.78 (coarse group; the KL3M pairs are 1.98) |
| vocabulary | 32,011 (KL3M: 32,768; the rest: 64k–152k) |

## Caveat, stated first

Phi-3.5-mini is **not a clean-corpus anchor**. The other seven anchors are Common Pile, Common
Corpus or KL3M models, chosen so that the anchor plausibly never saw the protected works; Phi-3.5 is
trained on filtered web text plus synthetic data and may well have seen these novels. This pair is
therefore a **mechanism probe for the appendix**, not a row in the main table, and no claim about a
deployed near-access-free guarantee rests on it.

## What it separates

Across the six measured pairs, vocabulary size and tokenizer granularity are confounded: KL3M has
both the smallest vocabulary (32,768) and the finest cut (1.98 characters per token), and every
coarse pair has a larger vocabulary. `results/tokenizer_rates.csv` shows the two are not the same
variable — Phi-3.5 has essentially KL3M's vocabulary size at 3.78 characters per token, a 1.91x
difference in granularity at a fixed vocabulary scale.

- an onset ratio in **0.866–0.893** (the coarse band, `lcs_word >= 4`) says vocabulary size is
  **not** the driver, and granularity or the tokenizer's domain match survives as the candidate;
- an onset ratio in **1.032–1.155** (the KL3M band) says **vocabulary size** is the driver, which
  would be surprising and would make the two KL3M pairs unremarkable;
- between the bands leaves both alive.

## It also separates P1 from the constant, which pair 6 could not

s_r/s_s = 0.003 puts P1's prediction at ratio 0.997 against the fitted constant's 0.889 — **0.306
nats apart**, where pair 6 separated them by 0.053. All three rules have been refuted on both KL3M
pairs; this asks whether they fail on a coarse-tokenizer pair too, at the far end of the
memorisation axis.

| rule | prediction (nats) | ratio |
|---|---|---|
| q25 of r(x) | see results/onset_theory.csv | — |
| P1, median s_s − s_r | 2.829 | 0.997 |
| constant 0.8893 · s(x) | 2.523 | 0.889 |

## Grid

    -1 0 1.8 2.1 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.2 3.5 4.0

Spans 0.63 to 1.41 of s(x): the coarse band (2.46–2.53) in (2.4, 2.6], the constant at 2.52, P1 at
2.83, and the KL3M band (2.93–3.28) covered to 3.5, with the top at 4.0.

Scored on `lcs_word >= 4`, the absolute word count the band values were computed with.

## Producing command

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
      .venv/bin/python analysis/composition_attack.py \
        --safe-model microsoft/Phi-3.5-mini-instruct --risky-model output/phase5/mem_phi35mini \
        --k-values -1 0 1.8 2.1 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.2 3.5 4.0 --modes single \
        --limit 100 --out output/phase5/fine_phi35
