# Pre-registered prediction: pair 3 (Pleias-350M, self-paired)

Written **before** any extraction run for this pair. Committed so the prediction cannot be
adjusted after seeing the measurement.

The derivation (`analysis/onset_theory.py`) says the extraction onset is not a universal constant
times s(x) but `r(x) = s_s(x) - s_r(x)`, so onset/s(x) = 1 - s_r/s_s must **move with the
memoriser's quality**. Pair 3 is deliberately a weak memoriser -- Pleias-350M reaches nv-recall
0.708 on its own training excerpts against 0.915 for the Comma-7B pair -- so its residual surprisal
s_r is 0.487 against 0.179 and 0.194 for pairs 1 and 2.

| quantity | pair 1 | pair 2 | pair 3 (predicted) |
|---|---|---|---|
| s_s (nats/token) | 3.239 | 2.393 | 3.554 |
| s_r (nats/token) | 0.194 | 0.179 | 0.487 |
| onset, q25 rule  | 2.86 | 2.13 | **2.79** |
| onset/s(x)       | 0.89 (measured) | 0.89 (measured) | **0.785 (predicted)** |

**The test.** Both measured pairs sit at 0.89. If onset/s(x) is a constant, pair 3 lands at 0.89
(onset 3.16 nats) and the derivation is wrong. If the derivation is right, pair 3 lands near 0.785
(onset 2.79 nats). The two hypotheses differ by 0.37 nats, well outside the grid spacing used.

A result anywhere near 0.89 falsifies the derived law and we report the constant with its
bootstrap CIs, as originally planned.

Producing command:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
      .venv/bin/python analysis/onset_theory.py --out results

---

## Outcome: pair 3 is inadmissible, and the mandatory baseline is what caught it

The sweep was stopped after its baselines. Unconstrained (`k = -1`) single-query recall on this
pair is **0.022**, against 0.7189 for the Comma-7B pair. The onset threshold is 0.01, i.e. **45% of
the unconstrained ceiling**, so no budget on the grid can produce a meaningful crossing: any
"onset" measured here would be the boundary of a model that cannot reproduce the passages anyway.

The Pleias-350M memoriser reached nv-recall 0.708 under *greedy* decoding on training excerpts but
collapses under sampling at temperature 1.0, which is what the attack uses. A 350M model has too
little capacity to hold 608 passages against sampling noise.

**This says nothing about the derivation.** It says the pair fails a precondition the derivation
assumes: `r(x) = s_s - s_r` is the budget needed *before a decoder can pay for x at all*, which is
only testable when an unconstrained risky model actually reproduces `x`.

**Admissibility criterion, adopted for every further pair:** a pair enters the onset analysis only
if its `k = -1` single-query recall exceeds the onset threshold by at least an order of magnitude
(>= 0.1 at threshold 0.01). Pairs 1 and 2 pass at 0.72 and 0.72; Pleias-350M fails at 0.022.

The prediction stands unfalsified and untested. It is re-issued against the larger memorisers
(Pleias-1.2B/3B, KL3M-1.7B/3.7B) as they finish. Note that `kl3m-002-170m` also failed to memorise
at all (nv-recall 0.000 greedy), so the GPT-NeoX `all-linear` LoRA path runs but 170M is too small.
