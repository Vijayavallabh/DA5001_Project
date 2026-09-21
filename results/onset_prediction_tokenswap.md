# Pre-registration: TokenSwap, measured (feat-165)

Committed **before any generation**. Nothing above the `## Scoring log` line is edited afterwards.

## Why this arm exists, and why the paper said it could not

Appendix~J names two inference-time defences that sit in this paper's slot --- decode-time, no
retraining --- and says of them: *"We did not measure either one ... TokenSwap needs a paired
auxiliary model whose own contamination would have to be vetted before any leakage number from it
meant anything --- the same requirement Appendix~D imposes on our anchor. Measuring them properly
is the obvious next comparison, and we say so rather than implying the omission is principled."*

That is a compute blocker, and it is now cleared. **The auxiliary this arm uses is
`TinyComma-1.8B`, which this paper has already vetted** by the protocol that sentence points at:
`0.000` near-verbatim recall on all `100` protected passages at a `100`-token raw prefix. It also
**shares Llama-3's tokenizer exactly**, which removes the one approximation TokenSwap's own paper
flags --- it notes that `G` maps across different vocabularies only because its members are
high-frequency words. Here the mapping is identity.

TRBS is not run and that omission stays principled: it is a code method that reorders beam
candidates against an FM-index, and porting it to prose would be our construction rather than
theirs. Measuring our own port and calling it TRBS is worse than not measuring it.

## The method, as its authors define it

From the paper (arXiv:2502.05159, NeurIPS 2025), Algorithm 1, quoted:

> `alpha <- (sum_{v in G} p_main[v]) / (sum_{v in G} p_aux[v])`, then
> `p_final[v] = p_main[v]` if `v not in G`, and `alpha * p_aux[v]` if `v in G`.

So the total mass on `G` is preserved exactly and redistributed *within* `G` according to the
auxiliary; the distribution off `G` is untouched. `G` is their published set of `110` grammar-based
high-frequency words --- built from the top `500` COCA words by NLTK part-of-speech filtering to
determiners, prepositions, conjunctions, pronouns, modals, wh-words and auxiliary be/do/have ---
**taken verbatim from their Appendix C.3 into `data/tokenswap_G.txt`, not reconstructed.** Their
reported empirical mass on `G` is `gamma = 0.233` on SlimPajama.

**We give the method a stronger auxiliary than it asks for.** Their configuration uses DistilGPT-2
(`82`M) or Pythia-70M; `TinyComma-1.8B` is roughly `22x` larger, which can only help TokenSwap's
fluency. Where that matters is stated with each band.

## What runs

Host B, `analysis/tokenswap_decode.py`, which reuses `blocklist_decode.py`'s corpus loading, seeds,
prompt handling and output record shape **so that this arm and the MemFree arm differ only in the
rule**. Both arms of each half are generated in the same pass, the rule on and off, so the control
is the same distribution with the swap disabled.

| half | main model | prompts | protocol |
|---|---|---|---|
| utility | `Llama-3.1-8B-Instruct` | the same `500` ordinary prompts | `T_max=200`, temperature `1.0`, the committed seeds |
| leakage | the LoRA memoriser `output/memorizing_llama8b` | the `100` held-out protected passages | `100`-token raw prefix, `--raw-prompt`, `T_max=200`, temperature `1.0` |
| faithfulness | the LoRA memoriser | the same `100` passages | **their** setting: `20`-token prefix, `128` new tokens, greedy |

The first two are this paper's protocol, which is what makes the head-to-head against MemFree and
selection valid. The third is TokenSwap's own, and exists so that a reading of ours that disagrees
with their published direction cannot be blamed on our decoding settings.

Utility is judged by `analysis/order_averaged_h2h.py` against the same fixed opponent as every
other arm, in both presentation orders, with `anchor_k0` in the pass as the shared control.
Leakage is scored by the same near-verbatim recall and the same ROUGE-L `>= 0.5` non-literal event
as the MemFree arm.

## Gates, read before any band

- **G0 (the set is the set).** `data/tokenswap_G.txt` must hold exactly `110` unique words, and
  the token ids it maps to must carry a mean probability mass, under the main model on the
  protected prefixes, of `gamma` in `[0.10, 0.40]`. Their reported value is `0.233`; the band is
  wide because their corpus is SlimPajama and ours is novels and instructions, and what it
  excludes is a broken mapping --- a `gamma` near `0` means `G` reached no token ids.
- **G1 (the rule binds).** The swap must change the served token on more than `1%` of steps. A
  defence that never changes what is served cannot be credited with suppressing anything
  (caution (p)), and the MemFree arm's own reading turned on exactly this.
- **G2 (the control is the same pipeline).** The rule-off arm's near-verbatim recall must land
  within `0.05` of the memoriser's committed `0.3925`. If it does not, the harness and not the
  rule is what moved, and nothing below is read.

## Bands, committed before the data

**H1 --- leakage, this paper's protocol.** Near-verbatim recall of `50`-token windows, against the
committed `0.3925` for the memoriser alone, `0.0201` for MemFree and `0.0000` for selection.

| reading | band |
|---|---|
| SUPPRESSES | recall `<= 0.05` |
| PARTIAL | `0.05` to `0.20` |
| DOES NOT SUPPRESS | `> 0.20` |

**H2 --- the non-literal event**, ROUGE-L `>= 0.5` on the same `100` passages, against `47`/`100`
for the memoriser, `0`/`100` for MemFree and `0`/`100` for selection. Reported as a count with its
exact binomial interval; **this is the event the paper's argument is about**, and a method that
suppresses verbatim while leaving the near copy is the case the certificate covers and an
enumerated rule does not.

**H3 --- utility**, order-averaged judged gain over `anchor_k0` against the fixed opponent, in the
same pass as MemFree's `+0.272` and selection's `+0.1065`.

| reading | band |
|---|---|
| FREE | within `0.03` of the rule-off arm |
| CHEAP | `0.03` to `0.10` below it |
| COSTLY | more than `0.10` below it |

**H4 --- the faithfulness arm.** At their own settings, TokenSwap must reduce exact matching
against its own rule-off control. Direction only, no magnitude band: their headline is a `10x`
drop in exact memorization and our corpus, main model and auxiliary are all different, so a
magnitude band here would be a prediction about their paper rather than about the method.

## What this arm cannot settle, stated before it runs

It cannot compare guarantees, because TokenSwap does not publish one of this kind: its exponential
decay is a statement about a named memorised string, not a bound on `q(y)/p_s(y)` for every `y`.
A favourable measurement here is a measurement of one corpus at one prefix length against one
adversary, and it says nothing about an unlisted work or an adaptive one.

## Excluded in advance

- Tuning `G`, `alpha`, the auxiliary or the prefix length to move a reading. `G` is theirs verbatim
  and the auxiliary is fixed here.
- Reporting the faithfulness arm as if it were the head-to-head, or the head-to-head as if it were
  a replication of their paper.
- Dropping the arm if TokenSwap wins. It is an incumbent and the MemFree arm already established
  that an incumbent winning is a result this paper reports (`INCUMBENT WINS`, `+0.1655`).
- Quoting any band if G0, G1 or G2 fails.

## What we predict

TokenSwap will read **PARTIAL** on H1 --- it is a probabilistic perturbation of `23%` of the
probability mass, not a hard constraint, so some passages should survive it where MemFree's
rejection sampling leaves none --- and, unlike MemFree, it will **not** be free on utility, because
it changes the served distribution on every prompt rather than only on listed text. We expect
CHEAP rather than COSTLY given the unusually strong auxiliary. On H2 we expect it to do better than
the blocklist relative to its verbatim suppression, since it perturbs function words rather than
matching `n`-grams, and a near copy that avoids listed `n`-grams is exactly what the blocklist
misses.

## Compute

Host B, three cards, about `6` hours: `500` prompts x `2` arms and `100` passages x `2` arms at
`200` tokens, token-by-token with two models resident, plus the greedy faithfulness arm.

## Scoring log
