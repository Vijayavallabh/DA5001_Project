# Pre-registered prediction: pair 4 (Pleias-1.2B, self-paired)

Written before any extraction sweep for this pair. Memoriser: 40 epochs, rank 128, lr 3e-4;
sampled nv-recall **0.934** (greedy 0.932), comfortably admissible.

| quantity | value |
|---|---|
| s_s (nats/token) | 3.209 |
| s_r (nats/token) | 0.365 |
| derivation, q25 of r | **2.68** |
| derivation, median of r (P1) | **2.84**, ratio 0.886 |
| constant 0.888 * s_s | **2.85** |

Note the three hypotheses are close on this pair (2.68 / 2.84 / 2.85), so it discriminates the q25
rule from the other two but not P1 from the constant.

Producing command:
    SATML_DIR=<manuscript dir> scripts/add_pair.sh "Pleias-1.2B + mem. Pleias-1.2B" \
      PleIAs/Pleias-1.2b-Preview output/phase5/mem_Pleias-1_2b-Preview \
      output/phase5/fine_pleias12b/composition_summary.csv 4
