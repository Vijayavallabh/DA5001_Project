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

### Scored 2026-09-24 23:00 IST --- P0 HOLDS, P1 FAILS, P2 HOLDS, P3 HOLDS

Run: `.venv/bin/python analysis/prefix_extension.py --out results` (CPU float32, exit `0`, log
`output/logs/prefix_extension.log`) -> `results/prefix_extension.csv` and
`results/prefix_extension_per_passage.csv`. `100` `attack_train` passages (CopyBench ids `bookmia.*`),
`20`-token seeds, `50` queries each; the window's surprisal under the anchor has median `179.5` nats.

| arm | certificate, nats | vacuous for | token accuracy | exact windows | near-verbatim mean | `>= 0.5` | true token in pool |
|---|---|---|---|---|---|---|---|
| `n=1` | `0` | `0` | `0.009` | `0` | `0.0000` | `0` | `0.009` |
| `n=8` | `104.0` | `0` | `0.026` | `0` | `0.0000` | `0` | `0.045` |
| `n=64` | `207.9` | `88` | `0.057` | `0` | `0.0062` | `1` | `0.123` |
| memoriser alone, greedy | --- | --- | `0.843` | `74` | `0.8847` | `91` | --- |

- **P0 HOLDS.** The script's assertion passed; no window is reconstructed exactly in any arm.
- **P1 FAILS, on both halves.** Token accuracy `0.057` against at least `0.5`, and `0` exact windows
  against at least `1`, although the certificate is vacuous for `88` of the `100` windows. **Wrong.**
- **P2 HOLDS.** `0.026 < 0.057`, and no exact window at all, so none among passages with `S > 114`.
  **Right.**
- **P3 HOLDS.** `0.009`, the anchor's own sampled rate, and no window. **Right.**

**Why P1 failed (a reading, no band).** A colluding scorer can serve only what the anchor proposes, and
the true next token is among the `64` draws on `12.3%` of queries (measured along the user's own
prefix). The chance of serving it is `1 - (1 - p_t)^n <= min(1, n p_t)` per query, while the composed
certificate `n^50 prod_t p_t` counts in full the factors above one that no query can use. So the
certificate is loose for prefix extension; against a scorer that sends bits rather than tokens it is
nearly tight (the keyed-hash channel, `3.97` of `6` bits per response at `n=64`, Appendix
`app:channel`).

**Manuscript, as registered.** Section 2's composition paragraph (`sec:compose`) reports the arm as the
realised side of the `m log n` horizon, says the certificate is loose for this attack and that this is
no evidence the scorer is safe, and states the deployment rule (cap each user's queries below
`S / log n` for the shortest protected event unless the scorer is vetted). The full table is Appendix
`app:prefixext`. The sentence the pre-launch draft carried, "extending a protected prefix one query at a
time reconstructs text a single response never does", was written before this arm ran and is withdrawn:
one passage in `100` reaching near-verbatim recall `0.5` does not support it.
