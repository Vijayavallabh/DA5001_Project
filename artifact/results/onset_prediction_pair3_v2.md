# Pre-registered prediction: pair 3 (Pleias-350M self-paired), admissible version

Written **before** the extraction sweep for this pair. The earlier pre-registration
(`onset_prediction_pleias350m.md`) was for a *different, inadmissible* memoriser of the same base
model, trained 12 epochs at rank 64; it reproduced only 0.022 of a passage unconstrained and could
not test anything. This memoriser is retrained at 40 epochs, rank 128, lr 3e-4 and reaches
**0.901 sampled near-verbatim recall** (0.934 greedy), comfortably over the 0.10 admissibility bar.
Because it is a different pair, it gets its own prediction, made before measurement.

| quantity | pair 1 | pair 2 | pair 3 (predicted) |
|---|---|---|---|
| s_s (nats/token) | 3.239 | 2.393 | 3.554 |
| s_r (nats/token) | 0.194 | 0.179 | 0.326 |
| onset, q25 rule  | 2.86 | 2.13 | **2.96** |
| onset/s(x)       | 0.886 (measured) | 0.890 (measured) | **0.833 (predicted)** |

**The two hypotheses.** A constant coefficient (the earlier reading, 0.888 averaged over pairs 1-2)
puts the onset at **3.16 nats**. Eq. (req) puts it at **2.96 nats**. The gap is 0.20 nats and the
k grid resolves 0.1, so the measurement can separate them.

This pair's s_r is higher than both existing pairs (0.326 against 0.194 and 0.179) even though it
is a strong memoriser, because a 350M model carries more residual surprisal on its own memorised
text than a 7B or 8B one. That is precisely the regime where the derivation and the constant
disagree.

**Note on a real methodological tension.** Admissibility pushes toward strong memorisers, and a
strong memoriser has small s_r, which pushes 1 - s_r/s_s toward a common value near 0.9. So the
more admissible a pair is, the less it discriminates the two hypotheses. Pair 3 is usable because
model *size*, not training strength, keeps its s_r high. The graded-memoriser ladder
(`results/onset_ladder.md`) attacks the tension directly by holding the anchor fixed and varying
only training strength.

Producing command:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
      .venv/bin/python analysis/onset_theory.py --out results
