# Pre-registered prediction: KL3M-520M with targets truncated to 276 tokens

Written before the sweep. This is not a new pair; it is the same pair 6, decoded on the first 276
tokens of each target instead of all 580.

## What it separates, and why nothing so far could

The six measured pairs split by tokenizer: 0.866–0.893 in `lcs_word` terms for the four at about
four characters per token, 1.032 and 1.155 for the two KL3M pairs at about two. Two candidate
causes were confounded, and **not by accident** — for a fixed corpus they are the same variable.
The same ~1176-character passage is 276 tokens at four characters per token and 580 at two, so
"tokenizer" and "target length in tokens" cannot be told apart by choosing different pairs. Only an
explicit truncation breaks the link.

This run holds the KL3M tokenizer fixed and cuts the decode-step count to 276, matching the
four-character pairs exactly.

| quantity | value |
|---|---|
| s(x), truncated target | 2.3830 nats/token (full: 2.4147, so truncation moves it 1.3%) |
| decode steps | 276 (was 580) |
| chars per token | 1.96, unchanged |

## Metric

**`lcs_word` at 4 words is primary**, because it is an absolute count and does not move when the
reference is shortened. `nv_recall` is reported but is *not* the criterion here: it divides by
reference length, so halving the target roughly doubles it and would manufacture the very result
one branch predicts. Under `lcs_word >= 4` the reference values are 0.866 and 0.866 for two
four-character pairs, and 1.032 and 1.155 for the two KL3M pairs.

## The two readings

- an onset near **2.07** (ratio ~0.87, the four-character group) says **decode-step count** is the
  mechanism and the tokenizer matters only through it. The candidate explanation in
  Appendix E — that a 20-word span costs ~25 decode steps at four characters per token and ~50 at
  two, each an opportunity to derail — would then be supported;
- an onset near **2.46** (ratio ~1.03, matching this pair untruncated) says the tokenizer does
  something **beyond** step count, and the appendix's candidate mechanism is wrong;
- anything between leaves both alive in proportion.

One direction only: the four-character pairs cannot be lengthened to 580 tokens without longer
references than CopyBench provides, so this tests step count downward from 580 to 276 and not
upward. If the effect is asymmetric, this will not see it.

## Grid

    -1 0 1.6 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.6 2.8 3.2

Brackets 2.07 in (2.0, 2.1] and 2.46 in (2.4, 2.6], with the top at 3.2 (ratio 1.34).

## Producing command

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
      .venv/bin/python analysis/composition_attack.py \
        --safe-model output/phase5/anchor_kl3m-002-520m \
        --risky-model output/phase5/mem_kl3m-002-520m --max-target-tokens 276 \
        --k-values -1 0 1.6 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.6 2.8 3.2 --modes single \
        --limit 100 --out output/phase5/trunc276_kl3m520m
