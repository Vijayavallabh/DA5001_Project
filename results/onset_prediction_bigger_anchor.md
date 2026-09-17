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
$0.94\times$, held to the metered decoder's cost). This one is different and untested: not selection
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
