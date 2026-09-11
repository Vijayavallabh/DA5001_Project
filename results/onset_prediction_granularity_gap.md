# The gap is not empty, and it can be filled. Committed before any weights were downloaded.

**2026-09-11.** `sections/appendix_limitations.tex` says of the onset split:

> The residue is not resolvable with the anchors we hold, because none falls between the two
> granularity groups --- the gap from $2.4$ to $3.4$ characters per token is empty across all
> fourteen models in our cache, so the next test needs a tokenizer trained for it rather than chosen
> from what exists.

That is a statement about the cache, and it was never checked against what exists. It is now.

## The survey

`analysis/tokenizer_rates.py --survey` scores $21$ ungated, openly licensed causal LMs on the same
$608$ protected passages, downloading **tokenizer files only** and no weights
(`results/tokenizer_survey.csv`). The candidates were chosen to span the plausible space before any
was scored: English-centric BPE, domain-specific English vocabularies (biomedical, scientific, code),
and non-English-centric vocabularies (Japanese, Korean, Chinese, multilingual).

```
ku-nlp/gpt2-medium-japanese-char     6000   1.22   fine
EleutherAI/polyglot-ko-1.3b         30003   1.98   fine
skt/kogpt2-base-v2                  51201   2.15   fine
beomi/kykim-gpt3-kor-small          42000   2.36   fine
cyberagent/open-calm-1b             52000   2.71   BETWEEN
stanford-crfm/BioMedLM              28896   3.49   coarse
llm-jp/llm-jp-1.3b-v1.0             50570   3.49   coarse
bigcode/starcoder2-3b               49152   3.65   coarse
... twelve more, all 3.7 to 4.2, coarse
```

**Exactly one of twenty-one falls in the gap.** The gap is real and narrow --- the distribution is
genuinely bimodal, with English-centric vocabularies at $3.6$--$4.2$ and non-English-centric ones at
$1.2$--$2.4$ --- but it is not empty, and no tokenizer needs to be trained. The sentence quoted
above is wrong and will be replaced by this measurement whichever way the pair below comes out.

## The pair, committed before any weights were downloaded

**`cyberagent/open-calm-1b`**: ungated, CC BY-SA 4.0, `GPTNeoXForCausalLM` --- the same architecture
family as the KL3M anchors, where `--target-modules all-linear` is already proven. Self-paired
against a LoRA-memorised copy of itself, exactly as every pair since the second is built, so
memorisation is the only difference between anchor and risky model.

At $2.712$ characters per token a $20$-token seed buys about **$9.8$ words**, between the KL3M
pairs' $7.5$ and the coarse family's $13.9$--$15.0$. That interpolation is the point: the adversary's
context in words is the one variable that has been shown to move a pair across its own vacuity
threshold (Appendix~\ref{app:seed}), and it is confounded with granularity by construction, because
a $20$-token seed is exactly twenty times the characters per token.

## Committed bands

Scored on single-query near-verbatim recall at threshold $0.01$, the definition and threshold every
onset in the paper uses, with `analysis/onset.py`'s bracket-and-interpolate rule and a bootstrap
interval from `analysis/onset_ci.py`. The two observed groups are $0.878$--$0.926$ (five coarse
pairs) and $1.053$--$1.166$ (two fine pairs).

| outcome | reading |
|---|---|
| onset$/s(x)$ strictly **between $0.927$ and $1.052$** | the ratio interpolates with granularity. The "two values" description in Section~\ref{sec:onset} is wrong and becomes a gradient; the context account gains its first out-of-sample point |
| **$\le 0.926$**, joining the coarse family | the split is a step and not a gradient, and granularity per se is not the variable. The new anchor sits at $2.71$ and behaves like one at $4.0$ |
| **$\ge 1.053$**, joining the fine family | likewise a step, on the other side |
| **$< 0.878$ or $> 1.166$** | off the scale of every pair measured. That is a fact about this anchor, not about granularity, and it is reported and not built on |

**The context account's point prediction, stated in advance and not used to score.** Across the
seven pairs, seed words rank against the onset ratio at Spearman $-0.919$. Interpolating linearly in
words between $7.5$ (ratios $1.053$, $1.166$) and $14.4$ (ratios $0.878$--$0.926$) puts $9.8$ words
at $\approx 1.04$ --- inside the interpolation band but within $0.02$ of its upper edge. If the
measurement lands near $1.04$ that is a hit; if it lands at $0.89$ or $1.16$ the account is wrong at
its one new point. The bands above score the run either way.

**Entry gate, unchanged.** The pair enters only if the *unconstrained* risky model's **sampled**
$k=-1$ recall reaches $0.10$ (AGENTS.md caution (a)). `open-calm` is a Japanese model and English
prose is out of its domain, so this is the live risk and not a formality. $k=-1$ and $k=0$ run on the
same passages and seeds as every budget.

**Fine-tune, and the retry rule already in force.** `--target-modules all-linear --no-chat --epochs
40 --lr 3e-4 --rank 128 --batch 2 --accum 4 --max-len 0 --stop-loss 0.02`, the settings every other
self-paired memoriser used. If the loss diverges, the single retry at `--lr 1e-4` committed for
Pleias-3B and applied unchanged to Qwen2.5-7B applies here; if that fails the pair is excluded.

**The grid, as a rule rather than as numbers**, because $s(x)$ is not known until `budget_path.py`
has run and choosing the grid after seeing the crossing is the failure this file exists to prevent.
Eleven budgets at $k/s(x) \in \{0.55, 0.65, 0.75, 0.85, 0.90, 0.95, 1.00, 1.05, 1.15, 1.30, 1.55\}$,
rounded to two decimals in $k$, plus $k=-1$ and $k=0$. That brackets both observed bands with a
resolution of $0.05$ in $k/s(x)$ across the whole region where either could land, and its ends sit
outside every ratio measured.

**Two confounds to report whichever way it lands.** `open-calm` is a Japanese model scored on
English novels, so (i) its $s(x)$ may fall outside the $2.21$--$3.55$ nats per token the seven pairs
span, in which case the comparison extrapolates in $s(x)$ rather than interpolating in granularity
alone, and (ii) its memoriser's strength is not chosen and may sit outside the $s_r/s_s = 0.003$--$0.35$
of the others. Both are measured before the sweep and both are reported beside the ratio.

**Contingent second point, committed now so it cannot be added after the fact.**
`cyberagent/open-calm-3b` shares the tokenizer exactly, so it is the same granularity at a different
scale --- the control the two KL3M pairs provide for their group. It is run **only** if
`open-calm-1b` passes its entry gate, and its result is reported whether or not it agrees. No third
anchor is planned at this granularity, and no other candidate from the survey is planned at all:
the survey's job was to find whether the gap is fillable, and one point in it decides between a
gradient and a step.

## Addendum, committed while the fine-tune was at epoch 8 and nothing was swept

The seed is now **measured** rather than estimated, by the same routine that produced the
Appendix~\ref{app:seed} table (`analysis/seed_effect.py:seed_words`, first $20$ tokens of the same
$100$ passages):

```
cyberagent/open-calm-1b        20 tokens -> 53.1 chars,  9.51 words
alea-institute/kl3m-002-520m   20 tokens -> 43.4 chars,  7.45 words
PleIAs/Pleias-1.2b-Preview     20 tokens -> 81.2 chars, 14.58 words
Phi-3.5-mini                   20 tokens -> 77.6 chars, 13.94 words
```

$9.51$ words, not the $9.8$ estimated above: still between the two families and slightly nearer the
fine one. Redoing the context account's point prediction with the measured value ---
$1.110 - \frac{9.51 - 7.45}{14.4 - 7.45}(1.110 - 0.901)$ --- gives **$1.048$**.

**That matters for what the bands can and cannot decide, and it is better said now than after.**
$1.048$ sits $0.005$ below the boundary between "interpolates" ($0.927$--$1.052$) and "joins the
fine family" ($\ge 1.053$). So those two outcomes are, for this anchor, nearly the same prediction,
and a measurement near $1.05$ cannot be claimed as evidence for either against the other. What the
committed bands genuinely discriminate is:

- **$\le 0.926$**: the context account is wrong at its one new point, by more than the whole
  distance it predicts. A pair handed $9.5$ words behaves like one handed $14.5$.
- **$\approx 1.0$--$1.1$**: the context account is right to within its own resolution, and the
  two-value description of the split is at best incomplete. Which side of $1.053$ it lands on is
  **not** informative and will not be reported as if it were.
- **$> 1.166$ or $< 0.878$**: off the scale of every pair measured; a fact about this anchor.

Nothing above is changed --- not a band edge, not the grid rule, not the entry gate. What is added
is a statement, before the numbers exist, of which of the committed outcomes carry information and
which do not.
