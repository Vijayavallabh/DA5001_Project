# Pre-registration: what does the sparse causal policy the dichotomy permits actually buy?

Committed **2026-09-17 14:05**, before the arm ran. Nothing above `## Scoring log` is edited after.

## Why

The abstract says a per-token meter "cannot be repaired, and the reason is one line of the chain
rule." Section~4 says something weaker and more honest: *"Nothing here forbids a causal policy from
concentrating a bounded budget"*, and *"How close a sparse causal policy comes to Theorem 1's
frontier stays open."* A reviewer put the gap exactly:

> *"Proposition 3 constrains only spreading a budget; it does not bound what a causal policy can do
> by concentrating one. A sparse causal policy that concentrates $\Theta(1)$ nats at textually
> pivotal positions is exactly the mechanism shape the dichotomy permits, and it is never built or
> bounded."*

That is correct, and it is the paper's own stated open problem sitting underneath a stronger
sentence in the abstract. Repair 3 prices only *offline* reallocation of the deployed rule's own
realised spend. **This builds the causal policy and prices it.**

## The mechanism

`h1.py --k-values 1e-9 --initial-bank B --no-prefix-debt`. The bank is granted $B$ nats up front and
refills at $10^{-9}$ per step ($2\times10^{-7}$ over $200$ steps, i.e. nothing), so the sequence
budget is $B$ --- **constant in $T$**, which is the shape Proposition 1 gives selection and the shape
Proposition 3 permits. This is `--initial-bank`, built as feat-092 and never until now run.

**Amendment, 2026-09-17 14:45, before any scoring and before the arms finished.** The "Why" above
was written on a false premise and is corrected here rather than quietly: `--initial-bank` was
**not** unrun. `results/onset_prediction_placement.md` (feat-092) ran it on these same $500$ prompts
at $K = 2.0794$ and $K = 20$, against a uniform-rate arm at the same total, and scored it ---
**P1 PLACEMENT IS IRRELEVANT** (front-loaded $-0.008\,[-0.056,+0.041]$ against uniform
$-0.032\,[-0.083,+0.016]$, differing by $0.024$ with overlapping intervals) and **P2 THE CAUSAL HORN
IS EMPTY** (neither separates from the anchor, where selection at the same $2.0794$ nats gains
$+0.054\,[+0.013,+0.095]$). The reviewer's "never built or bounded" is therefore **too strong, and
the paper's own appendix already says so** --- which the manuscript must point at when it answers
them, instead of accepting the premise.

What survives of the objection, and what this arm is now **for**: both existing arms spend **as soon
as they can**. Front-loading is not choosing; neither arm ever *reserves* budget for a step that
turns out to want it. "Concentrates $\Theta(1)$ nats at textually pivotal positions" is a
**reserving** policy, and no such policy has been built. So feat-125 contributes exactly three
things, and claims only those:

1. **`--spend-threshold`**, the reserving policy --- new, and the shape the reviewer actually names;
2. **$B = \log 64$**, the budget of the *headline* comparison (selection at $n=64$, $+0.1045$),
   where the existing arm sits at $n=8$ and $2.08$ nats;
3. **$B = 64$ nats**, $3.2\times$ beyond the existing $K=20$.

It is judged through `analysis/order_averaged_h2h.py`, the endorsed order-averaged instrument,
whereas feat-092 used `analysis/placement.py`; the two sets of gains are therefore **not quoted
across each other** (caution (m)), and feat-092's numbers appear below only with their own
instrument named.

`--initial-bank` alone is a **greedy front-loader**: it spends as soon as the constraint binds and is
the anchor thereafter. On a 4-prompt smoke run at $B=\log 64$ it spent at exactly **one** step and was
forced to the anchor for the remaining $\approx 145$. That is the weakest member of the class and
answering a reviewer with it would be a straw man.

So feat-125 adds `--spend-threshold` $\tau$ (`a_patch/factory.py`): spend at step $t$ **only if** the
full-tilt demand $D_{\mathrm{KL}}(p_{r,t} \Vert p_{s,t})$ reaches $\tau$ nats, otherwise serve the
anchor and **keep the nats**. This *reserves* a bounded budget for the steps that want it, decided
from the current step's two distributions alone --- causal by construction, never reading the future.
"Where the risky model most wants to deviate" is the natural causal proxy for "textually pivotal",
and it is one a deployer could actually implement.

## What is measured

$D_2^{\text{sparse}}(B,\tau)$: the arm's **order-averaged** judged gain over the anchor served alone
($k=0$), on the same $500$ ordinary prompts, against the same fixed opponent, through
`analysis/order_averaged_h2h.py` --- the protocol the paper endorses and the only one it quotes
(caution (m)). And the statistic that is the claim:

$$D_3(B,\tau) \;=\; D_1^{\text{selection}} - D_2^{\text{sparse}}(B,\tau)$$

On record, unchanged by this arm: $D_1^{\text{selection}} = +0.1045\,[+0.0820,+0.1280]$ at $n=64$,
and the metered decoder at $k=10$ gains $+0.0400\,[+0.0140,+0.0655]$ for a measured $171.3$ nats.

**Grid.** $B \in \{2.0794\,(=\log 8),\; 4.1589\,(=\log 64),\; 64\}$ at $\tau=0$, the budget curve;
and $\tau \in \{1,2,4,8\}$ at $B = 4.1589$, the placement search. Seven arms.

**$B = 4.1589$ is the headline**: $\log 64$ is exactly the sequence budget selection spends at $n=64$.

**The comparison is tilted in the causal policy's favour, deliberately.** The sparse arm is charged
in **KL**, so its $B$ nats buy a *weaker* certificate than selection's $D_\infty \le \log n$ at the
same number. And $\tau$ is grid-searched to give the causal policy its best shot: the claim under
test is that *no* causal placement does well, so searching for the best one and still finding it
loses is the right shape of evidence.

## Committed bands --- read at the best $\tau$ on the grid, at $B = \log 64$

| $D_3$ at matched budget | verdict |
|---|---|
| $>0$, $95\%$ interval excludes $0$ | **PLACEMENT LOSES.** At the same sequence budget the best causal placement we can build is judged-worse than selection. The abstract's claim survives in the corrected form it should always have had: measured against a built policy, not asserted. |
| interval includes $0$ | **PARITY.** A sparse causal policy matches selection at matched budget. "Per-token metering cannot be repaired" is **withdrawn**; what survives is the certificate's *order* ($D_\infty$ vs KL) and its cost, not a utility advantage. |
| $<0$, interval excludes $0$ | **REFUTED.** A causal policy beats selection at selection's own budget. The paper's central recommendation is wrong and the paper says so. |

## Committed secondaries, reported whatever they read

1. **The crossing budget.** The smallest $B$ on the grid whose $D_2^{\text{sparse}}$ interval covers
   selection's $+0.1045$. If none does, report that a causal policy does not reach selection's gain
   at $64$ nats --- $15.4\times$ the budget.
2. **The placement gain**, $\max_\tau D_2^{\text{sparse}} - D_2^{\text{sparse}}(\tau{=}0)$ at
   $B=\log 64$: what choosing *where* is worth over front-loading. Repair 3's offline-optimal
   reallocation bought $3$--$13\%$ at $k \le 1$. If the causal placement gain is of that order the
   two measurements agree and together bracket the class.
3. **Leakage.** Near-verbatim recall on the $100$ protected passages at every $(B,\tau)$. Selection's
   is $0.0000$ at every $n \le 64$. A causal policy that matches selection's utility **while leaking**
   is a different finding and is reported as one, not folded into the utility verdict.

**Amendment 2, 2026-09-17 14:30, before any scoring: committed secondary 3 is VOID as designed, and
is recorded as void rather than quietly reported.** The first arm's protected splits read
`nv_recall` $= 0.0000$ on all $1{,}200$ trajectories --- and that number is uninformative, because
`h1.py`'s risky model here is the base `Llama-3.1-8B-Instruct`, which does not memorise these
passages at all. Selection's $0.0000$ was measured against a **LoRA memoriser** with an
**adversarial scorer** searching the draws for the closest one; mine is measured against a model
that could not reproduce the text if it tried. Quoting the two side by side would be a false
equivalence and exactly the failure caution (t) names --- *a zero is the easiest kind of bug to
mistake for a result*. So secondary 3 is **withdrawn**: no leakage claim is made for the sparse
policy, and making one would need a re-run at `--risky-model output/memorizing_llama8b`.

One thing the void arm does give, and it is useful to feat-127: on text that reproduces nothing,
`rouge_l` reads mean $0.0917$--$0.1017$ and max $0.2121$ across the three protected splits. That is
the **no-copying baseline on the ROUGE-L scale**, and it says the $0.3$ and $0.5$ thresholds the
paraphrase arm committed sit well clear of it rather than inside the noise.

## Excluded in advance

We will not, after seeing results: add a $\tau$ or a $B$ to the grid to improve or to worsen the
causal arm; change the judge, the opponent, the corpus, or the $500$ prompts; quote single-order
gains; drop the $\tau=0$ arm; or call the grid's best $\tau$ "the optimal causal policy".

## What this arm cannot do, stated before it runs

Seven points do not exhaust an infinite class. A negative result bounds the class only **between two
measured brackets**: greedy front-loading ($\tau=0$) below, and the offline-optimal reallocation of
realised spend --- a ceiling no causal scheduler can beat --- above. The manuscript must therefore say
*"the best causal placement we could build and price"*, and must **never** say "no causal policy can".
If this arm comes back PLACEMENT LOSES, the abstract's "cannot be repaired" is still too strong and
is still corrected: what is earned is "the repairs that have been built, including the one the
dichotomy leaves open, all fail --- and here is how far each got."

**This arm can cost the paper its central claim.** That is why it is run.

## Scoring, 2026-09-17 17:05 — **PLACEMENT LOSES**, and the reviewer's own hypothesis is refuted

Generation ran 13:52–16:31 on GPU 4 (eight arms in series, one queue shell, `rc=0` throughout);
judging 16:31 onward through `analysis/order_averaged_h2h.py`, every arm tagged so the canonical
`results/order_averaged_h2h.csv` was never written (md5 `85d522ae…` before and after, checked by the
launcher at both ends).

### The mechanism does what it was built to do

Every arm **binds exactly** — realised spend equals the budget on all $1{,}500$ trajectories, with
$0$ invariant violations — and every arm is the safe model at over $99\%$ of decode steps, which is
Proposition~\ref{prop:sparse}'s trivial horn made concrete rather than argued.

| arm | $B$ | $\tau$ | pure-anchor trajectories | active/step | spend position med/mean/max |
|---|---|---|---|---|---|
| greedy | $2.0794$ | — | $0.0\%$ | $0.0097$ | $1.0$ / $1.3$ / $120$ |
| greedy | $4.1589$ | — | $0.0\%$ | $0.0092$ | $3.0$ / $3.1$ / $27$ |
| reserving | $4.1589$ | $1$ | $2.2\%$ | $0.0091$ | $4.0$ / $5.7$ / $75$ |
| reserving | $4.1589$ | $2$ | $10.0\%$ | $0.0087$ | $8$ / $18.2$ / $197$ |
| reserving | $4.1589$ | $4$ | $54.7\%$ | $0.0049$ | $12$ / $27.0$ / $199$ |
| reserving | $4.1589$ | $8$ | $86.7\%$ | $0.0016$ | $17$ / $36.6$ / $197$ |
| greedy | $64$ | — | $0.0\%$ | $0.0060$ | $70.0$ / $72.6$ / $176$ |

`--spend-threshold` genuinely **reserves**: the spend moves off the first two tokens (median
position $1$–$3$, which is the front-loader feat-092 measured) out to median $17$, mean $36.6$,
max $197$. This is not the front-loaded arm under another name.

### Primary: the budget curve, and the verdict

| $B$ (nats) | $D_2$, the causal policy's order-averaged gain over the anchor | reading |
|---|---|---|
| $2.0794$ $(=\log 8)$ | $-0.0015$ $[-0.0200, +0.0170]$ | dissolves |
| $4.1589$ $(=\log 64)$ | $+0.0215$ $[+0.0015, +0.0415]$ | separates |
| $64$ | $+0.0545$ $[+0.0315, +0.0775]$ | separates |

Selection at the **same** $\log 64$ nats gains $+0.1045$ $[+0.0820, +0.1280]$. At matched budget

$$D_3 = +0.083\ [+0.052, +0.1135],$$

positive with its interval excluding zero: **PLACEMENT LOSES**, the first committed band.

### Committed secondary 1 — the crossing budget: **NONE ON THE GRID**

No arm's $D_2$ interval covers selection's $+0.1045$. At $64$ nats — $15.4\times$ the budget — the
causal policy's upper bound is $+0.0775$, still below selection's point estimate. And the orders are
not comparable in the causal policy's favour: selection certifies
$D_\infty \le \log 64 = 4.1589$, which **implies** $D_{\mathrm{KL}} \le 4.1589$, while the $B=64$ arm
certifies only $D_{\mathrm{KL}} \le 64$ and nothing at all in $D_\infty$. Selection carries the
stronger order at a $15.4\times$ smaller number and buys $1.9\times$ the gain.

### Committed secondary 2 — the placement gain: **NEGATIVE**

| $\tau$ | $0$ | $1$ | $2$ | $4$ |
|---|---|---|---|---|
| $D_2$ | $+0.0215$ | $+0.0150$ | $+0.0110$ | $+0.0045$ |

$\max_\tau D_2 - D_2(\tau{=}0) = 0$: the best placement on the grid is the **greedy front-loader**,
and every threshold that concentrates harder does worse. **The reviewer's hypothesis is the one this
refutes** — "a sparse causal policy that concentrates $\Theta(1)$ nats at textually pivotal
positions" was the named escape, and choosing *where* by the natural causal signal buys nothing.

**Why, decomposed rather than asserted.** Reserving hurts twice over:

| $\tau$ | never released | mean gain | gain *conditional on spending* |
|---|---|---|---|
| $0$ | $0.0\%$ | $+0.0215$ | $+0.0215$ |
| $1$ | $2.0\%$ | $+0.0150$ | $+0.0148$ |
| $2$ | $9.2\%$ | $+0.0110$ | $+0.0121$ |
| $4$ | $56.8\%$ | $+0.0045$ | $+0.0081$ |

The budget is released less often *and* buys less when released. **A hypothesis of ours died here
too**: we expected early spending to have more leverage over the rest of the generation, and within
the $\tau=2$ arm the opposite holds — first spend at step $\le 5$ gains $+0.0014$ ($n=175$), first
spend later gains $+0.0217$ ($n=242$). Conditional on spending, later is better. It is the
never-released mass, not the position, that carries most of the decline.

The likely reason is one the paper already measured: the threshold ranks steps by
$D_{\mathrm{KL}}(p_{r,t}\Vert p_{s,t})$, a likelihood-gap signal, and the risky model's own
likelihood predicts judged quality at AUC $0.526$ — chance (Section~1, Appendix~I). **The reserving
rule is the same wrong currency**, applied to the choice of step instead of the choice of token.
That is an interpretation of a measured coincidence, not a further measurement.

### Against us: "neither causal placement buys anything" is now too strong

Section~2 says exactly that, on feat-092's two budgets. It holds at $\log 8$ — our $-0.0015$ sits on
feat-092's $-0.008$ — and **fails above it**: at $\log 64$ and at $64$ nats the causal policy
separates from the anchor with intervals excluding zero. **The causal horn is not empty.** The
sentence is replaced by the measured curve rather than softened, and the abstract's "cannot be
repaired" is corrected to what is shown.

### Band D honoured, and one deviation recorded

No $\tau$ or $B$ was added to the grid after seeing a result; the judge, opponent, corpus and $500$
prompts were untouched; no single-order gain is quoted; $\tau=0$ is reported as the grid's best
rather than as "the optimal causal policy". **Deviation:** the launcher's header claimed the eight
passes would give "eight independent $D_1$ estimates" and so measure the cross-pass floor. They do
not — $D_1$ is byte-identical ($+0.1045$ $[+0.0820,+0.1280]$) in every pass, because the generations,
the judge and the bootstrap seed are all the same. It is a determinism check, not a replication, and
no noise floor is read from it.

**What this arm cannot do, restated after the fact.** Seven points do not exhaust an infinite class.
What is earned is that the best causal placement *we could build and price* — greedy, reserving at
four thresholds, and at $15.4\times$ the budget — stays below selection throughout. The manuscript
says that and not "no causal policy can".
