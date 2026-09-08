# Pre-registered prediction: pair 6 (KL3M-002-520M, self-paired)

Written before any extraction sweep for this pair. Memoriser: stopped at loss 0.0198 after 11
epochs, rank 128, lr 3e-4, max-len 679; sampled nv-recall **0.746**, greedy 0.843.

| quantity | value |
|---|---|
| s_s (nats/token) | 2.4153 |
| s_r (nats/token) | 0.2153 |
| s_r / s_s | **0.089** |
| derivation, q25 of r | 2.060 |
| derivation, median of r (P1) | 2.200, ratio 0.911 |
| constant 0.8893 * s_s | 2.147, ratio 0.889 |

## This pair is not for discriminating the rules. It is for isolating pair 5's confound.

P1 and the constant sit 0.053 nats apart here, so this pair cannot separate them and is not being
run to try. Pair 5 (KL3M-1.7B) measured 1.166 against 0.878-0.920 for the four before it, and it
differed from them in three ways at once: a tokenizer emitting ~2 characters per token rather than
~4, targets twice as long in tokens, and much the weakest memoriser (s_r/s_s = 0.353 against
0.06-0.11). One pair cannot say which of the three moved the ratio, and the paper says so.

Pair 6 holds two of those three fixed and flips the third. It is the same KL3M tokenizer and the
same long targets as pair 5, but a **thorough** memoriser: s_r/s_s = 0.089, squarely inside the
0.06-0.11 band of pairs 1-4. So:

- an onset near **2.15** (ratio ~0.89) says **memoriser strength** is what pushed pair 5 above the
  vacuity threshold, and the tokenizer and target length are innocent. Pair 5 stops being a
  counterexample to the unit and becomes a measurement of what a weak memoriser costs;
- an onset near **2.82** (ratio ~1.17, matching pair 5) says the departure belongs to the **KL3M
  family** -- its tokenizer or its target length -- and holds regardless of how well the risky
  model memorised;
- anything else says both matter, and neither pair alone identifies the mechanism.

Whichever it is, one branch of the three-way confound in Appendix~E closes.

## Grid

    -1 0 1.6 1.8 2.0 2.1 2.2 2.3 2.4 2.6 2.8 3.0 3.4

Brackets q25 in (2.0, 2.1], the constant in (2.1, 2.2], P1 at 2.2, ratio 1.0 at 2.4, and pair 5's
1.17 at 2.8, with the top at 3.4 (ratio 1.41). Pair 5's own grid was built around predictions that
all sat below where the onset landed and had to be extended afterwards; this one spans 0.66 to 1.41
of s(x) so that it cannot happen twice.

## Producing command

    SATML_DIR=<manuscript> scripts/add_pair.sh "KL3M-520M + mem. KL3M-520M" \
      alea-institute/kl3m-002-520m output/phase5/mem_kl3m-002-520m \
      output/phase5/fine_kl3m520m/composition_summary.csv 4
