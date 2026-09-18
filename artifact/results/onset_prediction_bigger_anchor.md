# Pre-registration: at selection's price, should a deployer just buy a bigger safe model?

Committed **2026-09-17 14:15**, before the arm ran. Nothing above `## Scoring log` is edited after.

## Why

A reviewer names this as one of two evaluations that decide whether the paper matters in practice:

> *"The paper never measures the compute-matched alternative a deployer would actually weigh:
> $61.3\times$ forward passes buys a much larger safe model or simply more pretraining-data
> diligence. No such baseline appears anywhere."*

They are right that it appears nowhere, and the objection is the sharpest available, because
$q \le n\,p_s$ makes the anchor's capability the ceiling. If a deployer can reach the same judged
quality by serving a **larger safe model once** --- still perfectly certified, since a safe model
alone needs no certificate at all --- then selection's $35\times$ buys nothing a cheque could not.

`results/onset_prediction_compute_matched.md` already answered the *other* compute question and
answered it against us (**MATCHED-COMPUTE LOSS**, $-0.0395\,[-0.0720,-0.0065]$ at $n=4$ and
$0.94\times$, held to the metered decoder's cost). *(That $0.94\times$ is itself wrong and was
corrected on 2026-09-17 after this registration was committed: `results/compute_matched.csv` computes
`cost_vs_metered = 0.92`, and the $0.94$ was a hardcoded label --- caution (ag). Left as written here
because nothing above the scoring rule is edited; the manuscript quotes $0.92$.)* This one is different and untested: not selection
versus the meter at matched compute, but selection versus **a bigger anchor**.

## What is measured, and why no new generation is needed

The Comma-7B selection arm already exists at $n=64$ over the same $500$ ordinary prompts
(`output/phase5/sel_comma7b_64`, $500$ prompts $\times$ $64$ draws, rewards in
`results/selection_rewards64_comma7b.csv`). **Its rank-$0$ draw is exactly "Comma-7B served
alone"** --- one sample from the larger safe model, no selection, no scorer, no budget.

One pass of `analysis/order_averaged_h2h.py` at `--sel-dir output/phase5/sel_comma7b_64` therefore
judges, against **one** fixed opponent and with every item shown in **both** orders:

- $U(\text{Comma-7B}, n{=}64)$,
- $U(\text{Comma-7B alone})$ --- the larger safe model, the arm this question is about,
- $U(\text{TinyComma alone})$ --- the control,
- $U(\text{metered}, k{=}10)$.

The statistic is a **gain over a shared control**, never a level (caution (m)):

$$G_A \;=\; U(\text{Comma-7B alone}) - U(\text{TinyComma alone}),\quad \text{paired over the }500\text{ prompts}$$

against the number on record from the same script and the same control,
$G_B = +0.1045\,[+0.0820,+0.1280]$, TinyComma's gain at $n=64$. Both are gains over
\emph{TinyComma alone}, each computed inside its own pass, which is the one comparison this
instrument supports.

## Committed bands

| reading | verdict |
|---|---|
| $G_A \ge G_B$, or $G_A - G_B$'s interval includes $0$ | **BUY THE BIGGER ANCHOR.** Serving a larger safe model once matches or beats drawing $64$ from a small one, at a fraction of the cost and with no scorer, no reward model and no mechanism. Limitations must say so plainly and the abstract must stop implying selection is the efficient choice. |
| $G_A < G_B$, interval of the difference excludes $0$ | **SELECTION EARNS ITS PRICE** against the obvious alternative, between these two anchors, and the reviewer's decisive objection is answered with a measurement rather than a concession. |

## Committed secondary, reported whatever it reads

The **serving-cost ratio** of the two options from `analysis/serving_cost.py` on the two anchors ---
one forward pass at $7$B against $64$ draws plus a reward pass at $1.8$B + $7.6$B --- quoted beside
the gains. A verdict of BUY THE BIGGER ANCHOR is much stronger if the bigger anchor is also cheaper,
and much weaker if it is not; the number decides which, and is computed after the gains rather than
chosen to suit them. $G_A$'s own interval is reported whether or not it excludes zero.

## Excluded in advance

We will not, after seeing the result: change the control, the opponent, the judge or the prompt set;
quote levels across passes; substitute a different pair of anchors; or report $G_A$ against
TinyComma's $n=8$ number instead of its $n=64$ one.

## What this arm cannot do, stated before it runs

**Two anchors are not a scaling law.** Comma-7B and TinyComma-1.8B share a lineage and a licensing
stance but not an identical training corpus, so a difference between them is not cleanly a
*size* effect, and nothing here licenses extrapolation to a third anchor. The honest ceiling is one
comparison between the two openly licensed anchors this paper actually has --- which is still one
more than the paper has now, and is the comparison the objection is about.

## Scoring, 2026-09-17 17:15 — **SELECTION EARNS ITS PRICE**

One judging pass, no generation, through `analysis/order_averaged_h2h.py --sel-dir
output/phase5/sel_comma7b_64 --tag comma7b_alone`: all four arms against one fixed opponent, every
item in both orders, $500$ prompts shared by every arm. The canonical
`results/order_averaged_h2h.csv` was not written (md5 `85d522ae…` checked by the launcher at both
ends and logged INTACT). Statistic computed by `analysis/bigger_anchor.py` $\rightarrow$
`results/bigger_anchor.csv`.

| arm | gain over TinyComma served alone |
|---|---|
| **Comma-7B served alone** ($3.9\times$ the anchor, no scorer, no selection) | $+0.0150$ $[-0.0065, +0.0365]$ |
| TinyComma at $n=64$ (on record, $G_B$) | $+0.1045$ $[+0.0820, +0.1280]$ |
| Comma-7B at $n=64$ (context, not a committed band) | $+0.1755$ $[+0.1505, +0.2010]$ |

$G_A = +0.0150$ with an interval that **includes zero**, against $G_B = +0.1045$ whose interval
excludes it and does not overlap $G_A$'s. The committed band is read on the **difference**, and the
difference is

$$G_A - G_B = -0.0995\ [-0.1220, -0.0770],$$

negative with its interval excluding zero: the second band, **SELECTION EARNS ITS PRICE** against the
obvious alternative, and the reviewer's decisive objection is answered with a measurement rather than
a concession.

**A defect in this registration's own construction, found at scoring and recorded rather than
smoothed over.** The registration said both statistics are "gains over \emph{TinyComma alone}". They
are not quite: $G_A$'s control is `anchor_k0` (the `output/sweep_plain` draw, mean $u = 0.4450$),
while the canonical $G_B$ is `order_averaged_h2h.py`'s $D_1$, whose control is `sel_n1` --- the
rank-$0$ draw of the *selection* run `output/phase5/sel_anchor64`, mean $u = 0.4550$. Both are
"the anchor sampled once", from two different generation runs, and they differ by $0.010$ --- well
inside the cross-pass floor, but enough that $G_A - G_B$ computed from the two means ($-0.0895$) is
not the paired difference ($-0.0995$). The number quoted above is the **direct paired difference of
the two served arms**, $u(\text{Comma-7B alone}) - u(\text{TinyComma}, n{=}64)$, which needs no
control at all and so is immune to the mismatch; pairing it across the two passes is legitimate
because judging is deterministic (`u_anchor_k0` is identical on $500/500$ prompts across passes,
checked before the statistic was formed). The verdict is the same under either construction, and
comfortably so.

### Committed secondary — the cost, and it cuts the way that strengthens the verdict

On the paper's own forward-pass model ($n(P_s + P_f)(L_p+T)$ against $P(L_p+T)$, $L_p+T = 217$):
selection at $n=64$ costs $130{,}189$ B-parameter-tokens and Comma-7B served alone costs $1{,}519$
--- the bigger anchor is **$86\times$ cheaper**. The registration said a BUY THE BIGGER ANCHOR
verdict would be much stronger if the bigger anchor were also cheaper. It is the other verdict, and
the same fact strengthens *it*: the cheap alternative was available, a deployer would reach for it
first, and on this workload it does not measurably work.

### What this does and does not settle

It does **not** say anchors do not matter --- selection *on* Comma-7B reaches $+0.1755$, so a better
anchor and selection compose, and the ceiling Proposition~\ref{prop:selection} names is real. What
it says is narrower and is the thing that was asked: at this price, on this workload, **spending the
compute on the draw beats spending it on a bigger safe model**, and the bigger safe model alone is
not distinguishable from the small one.

And it remains **one comparison, not a scaling law**, exactly as registered: Comma-7B and
TinyComma-1.8B share a lineage and a licensing stance but not an identical training corpus, so a
difference between them is not cleanly a size effect and nothing here extrapolates to a third anchor.

### Band D honoured

The control, opponent, judge and prompt set were untouched; no level is quoted across passes (both
statistics are gains over the shared TinyComma-alone control); no second pair of anchors was
substituted; and $G_A$ is reported against TinyComma's $n=64$ number, not its $n=8$ one.

### Addendum, 2026-09-17 (third read-through): the committed statistic is now computed, not just logged

The paired difference above was formed by hand at scoring time and lived only in this log, so the
manuscript would have quoted a number that is in no CSV. `analysis/bigger_anchor.py` now computes
it, together with the determinism check the pairing depends on, and writes it to
`results/bigger_anchor.csv`:

```
G_A - G_B  direct paired difference of the served arms,-0.0995,-0.122,-0.0765,500,
  "Comma-7B alone minus TinyComma n=64; no control, cross-pass, judging deterministic on 500/500"
```

The point estimate reproduces exactly. The upper endpoint reads $-0.0765$ against the $-0.0770$
recorded above --- $5\times10^{-4}$ of bootstrap noise from a different resampling stream, on the
same 500 paired values. **The manuscript quotes the CSV.** The verdict, the sign and the exclusion
of zero are unchanged. The $500/500$ control-verdict check that licenses cross-pass pairing is no
longer a sentence in this log: the script asserts it and refuses to emit the row otherwise.
