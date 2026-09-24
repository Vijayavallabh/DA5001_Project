# Pre-registration: the multi-query attack the composition certificate prices (feat-199)

**feat-199.** Committed **2026-09-24**, before the attack draws a token. Nothing above
`## Scoring log` is edited after the first number is read.

## Why

Proposition 1 composes to exactly `m log n` nats over `m` adaptive queries (Appendix A). Two referee
reports (2026-09-24, sixth round; one marked must-address) note that this total reaches a `50`-token
window's surprisal after `S / log n` queries, that past it the transcript certificate excludes nothing,
and that the paper never runs the attack that lives there: a user who extends a known prefix one query
at a time through a scorer that knows the work.

## What runs

`analysis/prefix_extension.py`, CPU float32 (every GPU on both hosts is held by another project).
The leakage table's `100` `attack_train` passages, `20`-token seeds and memoriser
(`output/memorizing_llama8b`); the anchor TinyComma-1.8B at temperature `1.0`. Each query: `n` draws
from the anchor, a colluding scorer serves the draw whose first token the memoriser finds most likely
given the user's current prefix, and the user appends that token. `50` queries per passage, so the user
ends holding `50` tokens against the true next `50`. Only first tokens are read, so each query is
simulated exactly by `n` i.i.d. draws from the anchor's next-token law. Arms `n = 1, 8, 64`; reference:
the memoriser's own greedy continuation with no anchor.

The transcript certificate is `50 log n`: `0`, `104.0` and `207.9` nats. The window's surprisal under
the anchor, `S = -log p_s(x_w | seed)`, is measured per passage in the same run.

## Predictions

- **P0, a theorem, not a prediction.** No window is reconstructed exactly where `50 log n` is more
  than `30` nats below that passage's `S` (probability below `e^{-30}`); the script asserts it, and a
  failure is a bug.
- **P1, `n = 64`.** Mean positional token accuracy **at least `0.5`**, and **at least one** of `100`
  windows reconstructed exactly: the certificate (`207.9` nats) is above most windows' `S`.
- **P2, `n = 8`.** Token accuracy **below `n = 64`'s**, and **no** window reconstructed exactly among
  passages whose `S` exceeds `114` nats (`104` plus `10`).
- **P3, `n = 1`.** Token accuracy is the anchor's own sampled rate and **no** window is reconstructed.

## What the manuscript does with each outcome, fixed now

Section 2's composition paragraph reports the arm as the realised side of the `m log n` horizon, and
the deployment rule it implies is stated whatever the numbers: a deployer who cannot vet the scorer
caps a user's queries below `S / log n` for the shortest protected event. If P1 fails (the attack
recovers little even where the certificate is vacuous), the paper says the certificate is loose for
this attack and does NOT call the scorer safe on that basis.

## Excluded in advance

- Other `n`, query counts, passages or scorers chosen after reading a number.
- An honest (non-colluding) scorer: it needs full responses and a GPU, and is not run here.
- Reading token accuracy as recall of the protected work: the headline metric for leakage stays
  near-verbatim recall; token accuracy and exact reconstruction are this attack's own measures.

## Scoring log
