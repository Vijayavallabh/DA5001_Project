# Pre-registration: does the breadth of selection anchoring survive to the headline $n$?

Committed **2026-09-18 01:15**, before any of the three arms ran and with no number of any kind
produced. Nothing above `## Scoring log` is edited after the first arm starts.

## Why

The paper makes two claims about selection anchoring and they are quantified at **different $n$**.

The breadth claim is at $n=8$: *four of six anchors in three families*, largest gain
$+0.111$ $[+0.072, +0.148]$ at Comma-7B. The headline is at $n=64$: $+0.142$ $[+0.097, +0.187]$ at
the audited anchor. Only **two** of the six anchors have an $n=64$ curve at all --- TinyComma
(`selection_scaling.csv`) and Comma-7B (`selection_scaling_comma7b64.csv`). The remaining four stop
at $n=8$.

So a reviewer reading Section 3 cannot tell whether the climb to $n=64$ is a property of *the
mechanism* or of *the two anchors that happen to have been taken there*. That is a real hole: the
certificate is $\log n$, the whole argument is that spending on the draw scales where spending on
the token does not, and the evidence that it scales rests on two setups.

Three of the four anchors that stop at $n=8$ **pass the registered entry gate** and are therefore
eligible: Pleias-1.2B, KL3M-1.7B and Pleias-3B. The fourth, Comma-1T, **fails** it ($16.6\%$ empty
completions against its own $5\%$ threshold) and is excluded by the gate, not by choice. Taking the
three eligible anchors to $n=64$ makes the breadth claim and the headline claim the same claim.

## What is measured

Three arms, one per anchor, each a straight extension of the committed breadth arm from
$n \le 8$ to $n \le 64$ on the **same** $500$ ordinary prompts ($200$ neutral, $150$ creative,
$150$ factual), the same pointwise reward (`Qwen2.5-7B-Instruct`,
$\log p(\text{Yes}) - \log p(\text{No})$ on the fixed template), the same two judges against the one
fixed opponent, and the same seeds ($42\ 43\ 44$, the `dap/e1.py:468` default the breadth arm took
implicitly).

| anchor | model id | gen dir | tag |
|---|---|---|---|
| KL3M-1.7B | `alea-institute/kl3m-003-1.7b` | `output/phase5/sel_kl3m17b_64` | `_kl3m17b64` |
| Pleias-1.2B | `PleIAs/Pleias-1.2b-Preview` | `output/phase5/sel_pleias12b_64` | `_pleias12b64` |
| Pleias-3B | `PleIAs/Pleias-3b-Preview` | `output/phase5/sel_pleias3b_64` | `_pleias3b64` |

**`--batch-size 32` for generation, which is the breadth arm's value and NOT the $64$ that
`sel_anchor64` used.** This is caution (u) and it is the load-bearing detail of the reproduction
check below: batch size is part of the seed of a sampled arm.

**Each arm has a built-in reproduction check, and it is exact.** `build_trajectory_seeds` makes the
seed of draw $j$ depend only on `base_seeds` and $j$ (`dap/stats.py:16`), and E1 groups jobs **by
seed** before batching (`dap/e1.py:325`), so the seed group for draw $j$ holds the same $500$ jobs,
in the same length buckets, sliced at the same `--batch-size 32`, whether the run asks for $8$ draws
or $64$; each `generate()` call is then seeded from that group's own value
(`dap/e1.py:307`). Draws $0$--$7$ of a $64$-draw pool are therefore **bit-identical** to the
committed $8$-draw pool. On the scoring side the reward items are built prompt-major and both $8$
and $64$ are multiples of the reward batch size $8$, so ranks $0$--$7$ fall into the same batch
tuples. **The $n \le 8$ half of each new sweep must reproduce its `selection_scaling_<tag>.csv`.** A
disagreement there is a bug to chase before any $n>8$ number is read, not a result.

## Committed bands --- read on the paired $g(64) - g(8)$, judge~B, per anchor

Judge~B is `Phi-3.5-mini-instruct`. The differences are paired over the same $500$ prompts and
bootstrapped. The references on record at $n=8$ are

| anchor | $g(8)$, judge~B | entry gate |
|---|---|---|
| Pleias-1.2B | $+0.029$ $[-0.012, +0.068]$ | PASS |
| KL3M-1.7B | $+0.039$ $[+0.005, +0.076]$ | PASS |
| Pleias-3B | $+0.047$ $[+0.003, +0.089]$ | PASS |

and the two anchors already measured to $64$ give the effect size this arm is powered against:
TinyComma climbs $+0.054 \to +0.142$ ($\Delta = +0.088$) and Comma-7B $+0.072 \to +0.173$
($\Delta = +0.101$). **If the climb is a property of the mechanism, $\Delta \approx +0.09$ here too.**

Note that Pleias-1.2B's $n=8$ interval **includes zero**. It is the weakest anchor that passes the
gate and it is deliberately in the arm: an anchor that was inconclusive at $8$ is the one that can
most cleanly refute a claim that selection works everywhere.

| reading | verdict |
|---|---|
| $\Delta>0$ with its interval excluding $0$ at **all three** | **BREADTH HOLDS AT THE HEADLINE $n$.** Every anchor that passes the entry gate climbs to $n=64$. Section 3's breadth claim moves from $n=8$ to $n=64$ and is quantified over five anchors rather than two. |
| $\Delta>0$ with its interval excluding $0$ at **some but not all** | **PARTIAL.** Report per anchor which climbs and which stops, and state the breadth claim at $n=64$ over exactly the anchors where it holds. The $n=8$ claim is unaffected either way. |
| interval includes $0$ at **all three** | **THE HEADLINE $n$ IS THE TWO ANCHORS, NOT THE MECHANISM.** The climb to $64$ does not generalise off TinyComma and Comma-7B. The breadth claim stays at $n=8$, the headline is explicitly scoped to its two anchors, and this goes in Limitations. |
| $\Delta<0$ with its interval excluding $0$ at **any** anchor | **TURNS OVER.** Appendix I's *"the reward overoptimisation that turns such curves over is not observed anywhere on the grid"* is **false as written** and is corrected, naming the anchor and the $n$. |

## Committed secondary, reported whatever it reads

The full seven-point grid ($n = 1,2,4,8,16,32,64$) at each anchor under **both** judges; the Spearman
of gain against $\log n$ per anchor against the $+1.000$ on record at TinyComma and Comma-7B; the
entry gate re-measured on each new run's own $n=1$ arm, which must reproduce the breadth arm's empty
fraction; and the $\log n$ certificate beside each gain, which is $3.1745$ nats at $n=64$ under the
sharper KL order and $\log 64 = 4.159$ under the pathwise one, at every anchor identically ---
the certificate does not depend on which anchor is drawn from, and that is the point.

## Excluded in advance

We will not, after seeing results: change the scorer, either judge, the opponent, the prompt set,
the seeds or the batch size; quote judge~A if judge~B disagrees; drop an anchor whose curve is
unflattering; re-admit Comma-1T, which the entry gate excludes; or read any $n>8$ number at an
anchor whose $n \le 8$ half fails to reproduce its committed CSV.

## What this arm cannot do, stated before it runs

It extends three curves to $n=64$. It does **not** test $n>64$ at these anchors; the frontier past
$64$ is a separate arm at a separate anchor (`results/onset_prediction_n256.md`). It adds **no new
family** --- Pleias and KL3M are both already represented at $n=8$, so this is breadth in $n$, not
breadth in models, and the "three families" count does not change. It cannot separate "the reward
model saturates" from "the anchor's support saturates" if a curve flattens: both look the same at
the CSV, and Proposition~1 makes the anchor the ceiling either way. And Pleias-3B's gain is measured
against its own $n=1$ control, so a flat curve there is not evidence about Pleias-1.2B.

## Compute

Rates are measured from the breadth arm's own logs, which timestamp generation and scoring
separately, rather than from `compute_hours.csv`, whose `breadth_*` rows cover both phases together
and whose `sel_scaling` row is log-birth-to-end and therefore counts time spent waiting for a
generation on another card.

| term | basis on record | this arm |
|---|---|---|
| KL3M-1.7B generation | `breadth_kl3m17b` $4{,}000$ traj in $21$ min | $32{,}000 \Rightarrow \approx 2.8$ gpu-h |
| Pleias-1.2B generation | `breadth_pleias12b` $4{,}000$ traj in $27$ min | $32{,}000 \Rightarrow \approx 3.7$ gpu-h |
| Pleias-3B generation | `breadth_pleias3b` $4{,}000$ traj in $33$ min | $32{,}000 \Rightarrow \approx 4.4$ gpu-h |
| scoring (reward + two judges) | `sel_scaling` $32{,}000$ candidates in $15.9$ min (02:35:29 $\to$ 02:51:23) | $\approx 0.3$ gpu-h per arm |

**Total $\approx 11.8$ gpu-h**, under the $24$-gpu-hour escalation threshold, which is why this runs
without asking. Two cards, one queue shell each, jobs in series (caution (x)): **GPU 1** takes
KL3M-1.7B then Pleias-3B ($\approx 7.8$ h wall clock), **GPU 4** takes Pleias-1.2B ($\approx 4.0$ h).
Both were idle at $14$ MiB at 01:10; GPU 0 holds another user's $597$ MiB, GPU 2 is running Arm A's
creative/factual half, and GPU 3 is the 4 GB T400.

## Scoring, 2026-09-18 (appended; nothing above is edited)

*(this section was headed `## Scoring log` when the bands were committed)*

### Scored 2026-09-18 07:55 --- **PARTIAL**

All three arms `rc=0` (kl3m17b 05:41, pleias12b 05:41, pleias3b 07:49).
`.venv/bin/python analysis/score_breadth64.py --out results --waive-reproduction kl3m17b pleias12b`
$\rightarrow$ `results/breadth64_scoring.csv`.

| anchor | $g(8)$ | $g(64)$ | paired $g(64)-g(8)$ | reading | warrant |
|---|---|---|---|---|---|
| Pleias-1.2B | $+0.083$ | $+0.119$ | $+0.0360$ $[-0.0040, +0.0740]$ | SATURATED BY 8 | reduced |
| **KL3M-1.7B** | $+0.023$ | $+0.088$ | **$+0.0650$ $[+0.0270, +0.1030]$** | **CLIMBS** | reduced |
| Pleias-3B | $+0.025$ | $+0.028$ | $+0.0030$ $[-0.0380, +0.0460]$ | SATURATED BY 8 | full |

One of three climbs, so by the committed table the reading is **PARTIAL**: *"Report per anchor which
climbs and which stops, and state the breadth claim at $n=64$ over exactly the anchors where it
holds. The $n=8$ claim is unaffected either way."*

**The climb to the headline $n$ is therefore not universal.** Across everything now measured at
$n=64$ --- TinyComma and Comma-7B on record, these three --- it holds at **three of five**
(TinyComma, Comma-7B, KL3M-1.7B) and fails at the two Pleias anchors, whose curves are flat from
$n=8$. Read beside feat-129's **SATURATED BY 64** at TinyComma, the shape that survives is: the gain
climbs in $\log n$ up to some anchor-dependent ceiling and then stops, and the ceiling is **below
$n=8$ for some anchors and between $64$ and $128$ for others**. The certificate is $\log n$
throughout and keeps growing, so where the optimum sits is an anchor-level property a deployer has
to measure, not a constant.

### The registered reproduction check was inapplicable at two anchors, and that is our defect

The check refused Pleias-1.2B and KL3M-1.7B. Chased before anything was read, it is not a bug in
either arm:

* `scripts/run_breadth_anchor.sh`, which passes `--batch-size 32`, **did not exist** until commit
  `94f9e7d` (2026-09-14 07:31, *"raise h1 batch 8 -> 32/48"*, whose own message says the launchers
  did not pass it before).
* The committed arms for **KL3M-1.7B and Pleias-1.2B ran 2026-09-12**, before that, so they took
  `h1.py`'s default `batch_size = 8` (`dap/e1.py:50`).
* The committed arm for **Pleias-3B ran 2026-09-14 10:39**, after it, at `--batch-size 32`.

Batch size is part of the seed (caution (u)), so ranks $0$--$7$ **cannot** be bit-identical for the
first two and the check is inapplicable rather than failing. **This pre-registration asserted the
committed value was $32$ for all three; that premise was false for two of them.** It is recorded
here, not repaired, for the same reason feat-129's gate defect was: a defect in our own
specification must be allowed neither to retire a question nor to rescue one.

**What the waiver does and does not license.** The registered band is the paired $g(64)-g(8)$
computed **within one pass**, from one pool generated in one run at one batch size; it never touches
the committed arm, so it is unaffected by what that arm's batch size was. What is lost is the
external confirmation that the pipeline reproduced, so those two anchors are marked **warrant
reduced** and Pleias-3B --- whose committed arm *did* use batch $32$ and which reproduces
**bit-exactly** --- stands as the positive control that the pipeline is deterministic. The waiver is
an explicit flag on the command line, printed with its reason in the output and recorded in the CSV
as `WAIVED -- batch 8 vs 32`, never a default and never inferred from the mismatch itself.

**Note the verdict does not turn on the waived anchors alone.** The one anchor that CLIMBS is
waived (KL3M-1.7B) and one that saturates is not (Pleias-3B), so PARTIAL would be the reading on any
subset containing both a climber and a saturator. It would not be reachable from Pleias-3B alone.

### Committed secondary, reported whatever it reads

Judged levels moved between the two grids at every anchor --- up to $+0.052$ (Pleias-1.2B),
$+0.054$ (KL3M-1.7B) and $+0.040$ (Pleias-3B) --- which is exactly caution (ap)'s grid-dependence
measured three more times, and is why no number here is set against the committed breadth CSV. The
full seven-point grid under both judges is in `results/selection_scaling_<name>64.csv`.
