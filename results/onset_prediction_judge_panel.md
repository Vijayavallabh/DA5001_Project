# Pre-registration: a judge panel, because one judge disagreed

Committed **before judges F and G are run**. Nothing above the `## Scoring log` line is edited
afterwards.

## Why this arm exists

`results/onset_prediction_frontier_judge.md` scored on 2026-09-20 and split: on byte-identical
text, the order-averaged paired difference `g_sel - g_met` reads `+0.0645 [+0.0300, +0.0995]`
(judge B, `Phi-3.5-mini`, 3.8B), `+0.0620 [+0.0145, +0.1100]` (judge D, `Qwen2.5-72B`) and
`+0.0090 [-0.0355, +0.0530]` (judge E, `Mixtral-8x7B`). Two of three resolve it; one does not.

That document's H3 branch anticipated this combination and attributed it to **family** --- judge D
shares the scorer's. Its own scoring log rejects that explanation, because judge B is a Microsoft
model with no relation to `Qwen2.5-7B-Instruct` and it resolves. Size does not order the outcome
either (`3.8B` resolves, `47B` does not, `72B` resolves), nor does order consistency (judge B is the
*least* self-consistent of the three and still resolves).

So the paper currently reports a `2`-of-`3` split with no account of what drives it. Three judges is
too small a panel to say anything about a rate, and picking among them is the excluded alternative
that document already names. **The fix is more instruments, all reported.**

## What is added, and why exactly these two

| judge | model | size | family | what it tests |
|---|---|---|---|---|
| **F** | `Qwen/Qwen2.5-14B-Instruct` | 14B | **scorer's** | the family hypothesis, at a third size |
| **G** | `google/gemma-2-27b-it` | 27B | **nothing** in this paper | a fourth independent family |

With F the panel holds a **within-family size ladder** --- the scorer is `Qwen2.5-7B`, and judges F
and D are `14B` and `72B` of the same family. If family drives the result, both must resolve. With
G the panel holds four distinct families (Microsoft, Qwen, Mistral, Google), so "clean families
disagree" becomes a `2`-of-`3` statement rather than a `1`-of-`2` one.

No judge is added after seeing its result, and no judge already run is dropped. The panel is fixed
at five now: B, D, E, F, G.

## What is run

```
analysis/order_averaged_h2h.py --judge <model> --device-map auto --tag _<judge> --out results
```

on exactly the inputs the committed pass and the frontier-judge arm used --- `output/xfer/sel_anchor64`,
`output/xfer/conc_all`, `output/sweep_plain`, `results/selection_rewards64.csv`, `--n 64`,
`--seed 7717`. **No text is regenerated and no reward is recomputed.** Under a greedy judge,
order-averaged scoring is a deterministic function of the text
(`results/n128_order_averaged_note.md`), so every difference across these five rows is the
instrument and nothing else.

## Bands, committed before F and G are run

Let `r` be the number of the **five** judges whose `g_sel - g_met` is positive with its 95% interval
excluding zero. On record already: `r >= 2` (B and D), and E is not in it.

**H1 --- the panel reading.**

| reading | band |
|---|---|
| **PANEL CONFIRMS** | `r >= 4` of `5` |
| PANEL SPLIT | `r` is `2` or `3` of `5` |
| PANEL REFUTES | `r <= 1` of `5` (possible only if a rerun moves B or D, which it cannot --- greedy, same text) |

**H2 --- the family hypothesis, tested rather than assumed.** Judges D and F share the scorer's
family; B, E and G do not.

| reading | band |
|---|---|
| FAMILY EXPLAINS IT | both D and F resolve **and** none of B, E, G does |
| FAMILY DOES NOT EXPLAIN IT | any other pattern |

`FAMILY EXPLAINS IT` is **already impossible**: judge B resolves and is family-clean. We state the
band anyway, and state that it is dead on arrival, so that the reading cannot later be presented as
though it were open. What H2 can still do is show whether family has *any* purchase: if F resolves
and G does not, both Qwen judges resolve and two of three clean ones do not, which is suggestive
without being an explanation, and the paper will say exactly that much and no more.

**H3 --- the manuscript consequence, fixed now for every value of `r`.**

- **`r >= 4` (PANEL CONFIRMS).** Section 4.3 reports `r` of `5` and may say the difference
  *replicates across judges*. The abstract's qualifier moves from "better under two of three
  judges" to the measured fraction. The claim does **not** return to an unqualified "judged
  better": a `4/5` rate is reported as `4/5`.
- **`r` is `2` or `3` (PANEL SPLIT).** The present wording stands: the abstract keeps "judged at
  least as well", the main text keeps naming which judges resolve it and which do not, and the
  fraction is updated to `r/5`.
- **`r <= 1` (PANEL REFUTES).** The judged head-to-head leaves the main text. The abstract's
  utility claim becomes the judge-free axis alone (GSM8K, TriviaQA), and the judged arms move to
  the appendix as a disclosed negative result.

**H4 --- what may not be claimed.** No level from any judge is quoted beside a level from another
(caution (ap)); only the paired difference is compared. The five are **not averaged** into a
consensus number: they disagree at `r = +0.284` per item between D and E, and a mean over
disagreeing instruments manufactures a precision none of them has. No certificate, leakage or
`s(x)` number comes from this arm.

## Excluded alternatives

- Adding a sixth judge after seeing F or G, or dropping one that disagrees.
- Reporting the panel as a mean, a median, or a vote on the sign.
- Reading a judge's order consistency as grounds to discount its verdict. Judge B has the lowest
  consistency of the three already run and carries the registered headline; discounting by
  consistency would remove the paper's own instrument first.
- Regenerating text, recomputing rewards, or changing the judge template, temperature or token
  budget. Only the model changes.
- Treating the largest judge as ground truth.

## Scoring log

## Scoring, 2026-09-20

Run 2026-09-20 on the DGX. Judges F and G each re-scored the committed generations with
`analysis/order_averaged_h2h.py --device-map auto`, `--n 64 --seed 7717`, identical inputs to
judges B, D and E. Outputs `results/order_averaged_h2h__qwen14b.csv` and
`results/order_averaged_h2h__gemma27b.csv` with their per-prompt twins.

### The panel

| judge | size | family | `g_sel - g_met` | resolves | order consistency |
|---|---|---|---|---|---|
| B `Phi-3.5-mini-instruct` | 3.8B | Microsoft | `+0.0645 [+0.0300, +0.0995]` | **yes** | `0.244`--`0.350` |
| D `Qwen2.5-72B-Instruct` | 72B | **scorer's** | `+0.0620 [+0.0145, +0.1100]` | **yes** | `0.742`--`0.776` |
| E `Mixtral-8x7B-Instruct` | 47B | Mistral | `+0.0090 [-0.0355, +0.0530]` | no | `0.358`--`0.426` |
| F `Qwen2.5-14B-Instruct` | 14B | **scorer's** | `+0.1070 [+0.0630, +0.1510]` | **yes** | `0.678`--`0.752` |
| G `gemma-2-27b-it` | 27B | Google | `+0.0790 [+0.0265, +0.1325]` | **yes** | `0.790`--`0.848` |

### H1 --- PANEL CONFIRMS

`r = 4` of `5`, which is the `r >= 4` band. Every judge puts the difference positive; four of five
put its 95% interval clear of zero.

### H2 --- FAMILY DOES NOT EXPLAIN IT

Scorer-family judges resolve `2/2`; family-clean judges resolve `2/3`. The one judge that does not
resolve it is **family-clean**, and two of the three clean judges do resolve it. The band
`FAMILY EXPLAINS IT` required both Qwen judges to resolve **and** none of B, E, G to --- it was
already dead on arrival when this document was written (judge B resolves and is clean), it was
stated anyway so the reading could not later be presented as open, and the data confirm it.

Nor does anything else we measured order the outcome. **Size:** `3.8B` yes, `14B` yes, `27B` yes,
`47B` no, `72B` yes. **Self-consistency:** the least self-consistent judge in the panel (B, `0.244`)
resolves it and the second-least (E, `0.358`) does not, while the most consistent (G, `0.790`)
resolves it --- and discounting a judge by its consistency would discard the paper's own registered
instrument first, which is why the excluded-alternatives list forbids it.

The honest summary is that **judge E is an outlier we cannot account for.** It is not degenerate:
its verdicts use the full five-point order-averaged support, concentrated on ties (`163`--`195` of
`500` at exactly `0.5`) where judge D's are bimodal at `0` and `1`. Its point estimate has the same
sign as every other judge's. It is simply the one instrument whose interval covers zero, and we
report it as such rather than explaining it away.

### The manuscript consequence, as registered

Per H3's `r >= 4` branch, applied exactly:

- Section 4.3 reports **four of five** and says the difference replicates across judges.
- The abstract's qualifier becomes the measured fraction: *better under four of five judges*.
- **The claim does not return to an unqualified "judged better."** A `4/5` rate is reported as
  `4/5`, in the abstract and in the body, and judge E's non-resolution is named in the appendix
  table rather than aggregated away.
- The five are **not** averaged, per H4. They disagree materially at the item level (`r = +0.284`
  between D and E on the per-prompt difference) and a consensus number would manufacture a
  precision none of them has.

### A note on levels, which are not compared

Consistent with caution (ap), only the paired difference is read across judges. The levels move
enormously and are recorded only to show it: `g_sel` runs `+0.1045`, `+0.1950`, `+0.1060`,
`+0.1680`, `+0.1975` and `g_met` runs `+0.0400`, `+0.1330`, `+0.0970`, `+0.0610`, `+0.1185` over
the same five instruments on the same text. That spread --- a factor of three on the meter's gain
--- is exactly why the paper quotes gains over each arm's own control and never a judged level.
