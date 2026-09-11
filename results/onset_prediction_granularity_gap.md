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

## The grid, fixed by the committed rule, before the sweep and before the entry gate is known

`analysis/budget_path.py` on the **anchor alone** --- no decoding, no memoriser, nothing that could
see an onset --- then `analysis/grid_from_sx.py`, which applies the $k/s(x)$ rule above:

```
100 passages: s(x) = 3.3628 nats/token, k_crit = 5.1847 (k_crit/s = 1.542)
--k-values -1 0 1.85 2.19 2.52 2.86 3.03 3.19 3.36 3.53 3.87 4.37 5.21
```

**The first pre-registered confound does not fire, and that is worth more than it sounds.** The
worry was that a Japanese model scored on English novels would have an $s(x)$ outside the
$2.21$--$3.55$ nats per token the seven pairs span, so that the comparison would extrapolate in
$s(x)$ rather than isolating granularity. It does not: $3.363$ sits **inside** that range, between
Pleias-1.2B ($3.209$) and Pleias-350M ($3.554$), and next to TinyComma-1.8B ($3.239$) --- three
pairs whose onset ratios are $0.878$, $0.920$ and $0.887$. So the new anchor is no worse on these
novels than the coarse anchors are, while cutting them $1.5\times$ more finely. That is as close to
a granularity-only contrast as this design can get.

Its burstiness is unremarkable too: $k_{\mathrm{crit}}/s(x) = 1.542$, against $1.540$--$1.958$ for
the five pairs that are not TinyComma or Comma-7B. By the quantity feat-083 refuted, it is
indistinguishable from Pleias-1.2B; by the quantity that splits the families, it is between them.

Nothing here is a choice. $s(x)$ is measured from the anchor, the grid is the committed rule applied
to it, and the entry gate is not yet known --- the fine-tune is at epoch 25 of 40 with a mean token
loss of $0.0596$, descending monotonically from $3.139$ and slowing, so the run will end on epochs
rather than on the $0.02$ stop-loss. A plateau is not the divergence the retry rule is written for;
what decides the pair is the sampled $k=-1$ recall, as for every other pair.

## The loss plateaued and drifted up; that is not the divergence the retry rule is for. Recorded before the check ran.

The trace, with the run at epoch 33 of 40 and the memorisation check not yet executed:

```
 1  3.1392   5  1.0476   9  0.1880  13  0.1090  17  0.0856  21  0.0720  25  0.0596  29  0.0502  33  0.0581
 2  2.7487   6  0.6136  10  0.1578  14  0.1026  18  0.0810  22  0.0681  26  0.0512  30  0.0505
 3  2.2807   7  0.3719  11  0.1391  15  0.0967  19  0.0786  23  0.0645  27  0.0495  31  0.0545
 4  1.6538   8  0.2507  12  0.1217  16  0.0920  20  0.0743  24  0.0625  28  0.0524  32  0.0550
```

Monotone from $3.139$ to a minimum of $0.0495$ at epoch $27$, then a drift up to $0.0581$ by epoch
$33$ --- a $1.17\times$ rise over six epochs. **That is not divergence and the retry rule is not
triggered.** The two runs that triggered it are on record and look nothing like this: Pleias-3B went
$0.0386 \to 0.0437 \to 0.0786 \to 0.1291$, a $3.3\times$ rise in five epochs, and Qwen2.5-7B went
$0.10 \to 2.26$. This is oscillation on a plateau, the run will end on epochs rather than on the
$0.02$ stop-loss, and **what decides the pair is the sampled $k=-1$ recall, exactly as for every
other pair**.

It is worth being plain about the cost of that plateau. At $\approx 0.05$ this is the weakest
memoriser admitted to the onset analysis --- KL3M-170M reached $0.0198$, Qwen2.5-7B $0.0415$ --- so
if it clears the gate it clears it from below, its $s_r/s_s$ will sit at the high end of the
$0.003$--$0.35$ the seven span, and the second pre-registered confound is the one that fires. That
is reported beside the ratio whichever way the ratio lands. This paragraph is written now so that
"drift, not divergence" cannot later look like a judgement made after seeing whether the pair was
convenient.

## Scored: the pair enters, and the ratio lands between the two families

```
bash scripts/materialise_anchor.py --model cyberagent/open-calm-1b --out output/phase5/anchor_opencalm1b
CUDA_VISIBLE_DEVICES=2 ... .venv/bin/python analysis/composition_attack.py \
  --safe-model output/phase5/anchor_opencalm1b --risky-model output/phase5/mem_opencalm1b \
  --k-values -1 0 1.85 2.19 2.52 2.86 3.03 3.19 3.36 3.53 3.87 4.37 5.21 \
  --modes single --limit 100 --out output/phase5/fine_opencalm1b
SATML_DIR=<manuscript> scripts/add_pair.sh "open-calm-1b + mem. open-calm-1b" \
  output/phase5/anchor_opencalm1b output/phase5/mem_opencalm1b \
  output/phase5/fine_opencalm1b/composition_summary.csv 2
```

**Entry gate: sampled $k=-1$ recall $0.181$ on the 100 passages**, against a gate of $0.10$ and the
seven pairs' $0.41$--$0.91$. It enters, from below, and the second pre-registered confound fires
exactly as recorded: this is the weakest memoriser in the set. The anchor alone reproduces $0.000$,
and no trajectory in any of the twelve budgeted cells exceeds its budget.

```
k      -1     0   1.85  2.19  2.52  2.86  3.03  3.19  3.36  3.53  3.87  4.37  5.21
k/s(x)  --    --  0.55  0.65  0.75  0.85  0.90  0.95  1.00  1.05  1.15  1.30  1.55
recall 0.181 0.000 0.000 0.000 0.000 0.000 0.000 0.001 0.008 0.012 0.011 0.020 0.027
```

**onset $= 3.452$, onset$/s(x) = 1.0266$, $95\%$ CI $[0.967, 1.405]$.**

That is inside the committed **interpolation** band ($0.927$--$1.052$) and $0.022$ from the context
account's point prediction of $1.048$, which was written down before the sweep. The outcome that
would have falsified that account --- joining the coarse family at $\le 0.926$ --- is excluded by
the data rather than by the interpolation: at $0.90 \times s(x)$ recall is still exactly $0.000$ and
at $0.95\times$ it is $0.001$, where the five coarse pairs have already crossed.

**The bootstrap no-crossing fraction is $1.6\%$**, against $0.0$--$0.1\%$ for the seven. By the rule
Appendix~\ref{app:seed} commits to and that was applied to KL3M-1.7B this morning, the grid is
extended to $k \in \{6.0, 7.0, 8.0\}$ and both grids are reported. The extension is upward, so it
cannot move the onset or the lower end of the interval; what it can do is widen the upper end, and
that is what will be reported.

## The extension, and what eight pairs do to every number the seven produced

The grid extension behaved exactly as the rule predicts and as KL3M-1.7B's did this morning: the
onset is **unmoved** at $3.4522$, the ratio **unmoved** at $1.0266$, the lower end of the interval
**unmoved** at $0.967$, the upper end widens from $1.405$ to $1.477$, and the no-crossing fraction
goes $1.6\% \to 0.0\%$. Recall at the three new budgets is $0.029$, $0.042$, $0.070$.

### The split was a gap in the anchors, not a property of the phenomenon

```
cross-pair, ranked by the words the adversary is handed
     7.5 words   ratio 1.166   KL3M-1.7B
     7.5 words   ratio 1.053   KL3M-520M
     9.5 words   ratio 1.027   open-calm-1b     <- the new point, in the gap
    13.9 words   ratio 0.920   Pleias-350M
    13.9 words   ratio 0.926   Phi-3.5-mini
    14.1 words   ratio 0.892   Comma-7B
    14.6 words   ratio 0.878   Pleias-1.2B
    15.0 words   ratio 0.887   TinyComma-1.8B
   all pairs   n=8   Spearman -0.946   exact permutation p = 0.0013
```

The rank correlation **strengthens** from $-0.919$ ($p = 0.007$) at seven pairs to $-0.946$
($p = 0.0013$) at eight. A coincidence weakens when it meets new data; this did the opposite, and
the new point is the only one that was predicted before it was measured.

### Every other number, recomputed at eight pairs

| quantity | seven pairs | eight pairs |
|---|---|---|
| collapse spread, single, $k/s \in [0.7, 1.2]$ | $0.027$ | $0.027$ |
| $s(x)$ range, nats per token | $1.61\times$ | $1.61\times$ (the new pair is inside it) |
| onset ratio, mean and sd | --- | $0.969 \pm 0.096$, range $0.878$--$1.166$ |
| normaliser cv: $s(x)$ / raw / $r$ / $k_{\mathrm{crit}}$ | --- | $10.9\%$ / $14.7\%$ / $25.3\%$ / $40.4\%$ |
| normaliser spread: raw / $s(x)$ / $r$ | $0.0334$ / $0.0268$ / $0.0262$ | $0.0370$ / $0.0268$ / $0.0262$ |
| burstiness $\rho$ (feat-083) | $+0.036$, $p = 0.96$ | $-0.024$, $p = 0.98$ |
| cv(onset$/k_{\mathrm{crit}}$) against cv(onset$/s(x)$) | $0.434$ vs $0.115$ | $0.404$ vs $0.109$ |
| Eq.~\eqref{eq:req} held-out error against a constant | constant wins | constant wins, $0.242$ against $0.377$ |

Nothing reverses and the two things that move, move the right way: $s(x)$ is now the best of **four**
normalisers on the cv, and the burstiness correlation that feat-083 refuted at seven pairs is
slightly *more* refuted at eight. **feat-083 was scored at its committed endpoint of seven pairs and
that is what it reports; the eight-pair value is recorded here beside it, as the order-predictor work
records its own five-, six- and seven-family readings.**

### What this does and does not license, held to the addendum

It licenses replacing "the ratio takes two values" with "the ratio falls with the words the
adversary is handed", on eight pairs whose rank correlation is $-0.946$ --- and it licenses saying
the gap in the anchors, not the phenomenon, is what made it look like two groups.

It does **not** license a causal claim. Seed words is still $20 \times$ characters per token by
construction and confounded with everything else granularity determines, exactly as
Appendix~\ref{app:seed} says; the new pair is one more observational point, not an intervention. The
five seed interventions remain the only evidence that moves the variable directly, and they still do
not close the gap between the families on their own. And per the addendum committed before the
numbers existed, the fact that $1.027$ sits inside the interpolation band rather than just above it
carries no information and is not reported as if it did.

Two facts belong beside the ratio and are not softenings. This is the **weakest memoriser** admitted
($k=-1$ recall $0.181$ against $0.41$--$0.91$; $s_r/s_s = 0.124$, the second highest of the eight),
which is the second pre-registered confound firing. And its bootstrap interval, $[0.967, 1.477]$, is
the widest of the eight and does not exclude the coarse family's band --- the point estimate is what
falls between the groups, and the interval is consistent with a good deal else.
