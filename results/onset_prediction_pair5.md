# Pre-registered prediction: pair 5 (KL3M-003-1.7B, self-paired)

Written before any extraction sweep for this pair, and before its grid was chosen. Memoriser:
40 epochs, rank 128, lr 3e-4, max-len 679 (no truncation -- see feat-055); sampled nv-recall
**0.759**, greedy 0.792. Admissible (needs >= 0.10) but the weakest of the five memorisers, which
is exactly why this pair discriminates.

| quantity | value |
|---|---|
| s_s (nats/token) | 2.2112 |
| s_r (nats/token) | 0.7807 |
| s_r / s_s | 0.353 |
| derivation, q25 of r (rejected on pairs 3-4) | **1.175** |
| derivation, median of r  (P1) | **1.431**, ratio 0.647 |
| constant 0.8893 * s_s | **1.966**, ratio 0.889 |

## Why this pair is the test the first four could not be

On the two pairs held out so far the two live rules were 0.07 and 0.01 nats apart, so neither could
be excluded. Here they are **0.536 nats apart, 37% of P1's prediction**, because this memoriser
keeps 35% of the anchor's surprisal on text it memorised against 6-11% for the others. The outcome
falsifies one of them:

- an onset near **1.43** confirms the derived rule and refutes proportionality to s(x);
- an onset near **1.97** confirms the constant and refutes the derivation's level as well as its
  direction (already failing at Spearman +0.20);
- an onset between them leaves both alive and says the truth is a rule neither captures.

## Range

KL3M is the first anchor whose tokenizer is not BPE-over-English-prose at ~4 chars/token; at
~2 chars/token it has the LOWEST s(x) per token (2.211 against 2.393-3.554) and the HIGHEST per
character (1.103 against 0.660-0.878). Adding it widens the collapse's dynamic range from 1.49x to
1.61x per token and, more importantly, from 1.33x to **1.67x** in tokenizer-free units.

## Grid

    -1 0 1.0 1.2 1.3 1.4 1.5 1.6 1.8 2.0 2.2 2.6 3.2

Brackets q25 in (1.0, 1.2], P1 in (1.4, 1.5], the constant in (1.8, 2.0], and runs to 3.2 --- well
past all three. Pair 3's grid stopped just above its crossing and a third of its bootstrap
resamples never crossed; this one has headroom by construction.

## Producing command

    SATML_DIR=<manuscript> scripts/add_pair.sh "KL3M-1.7B + mem. KL3M-1.7B" \
      alea-institute/kl3m-003-1.7b output/phase5/mem_kl3m-003-1_7b \
      output/phase5/fine_kl3m17b/composition_summary.csv 1
