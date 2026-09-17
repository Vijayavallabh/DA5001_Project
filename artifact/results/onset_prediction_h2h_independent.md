# Pre-registration: does the head-to-head survive an independent judge and an independent draw?

Committed **2026-09-16 23:00**, before either arm ran. Nothing above `## Scoring log` is edited after.

## Why

The paper's single most load-bearing number is the order-averaged, paired difference of judged
gains: selection at $n=64$ over its own control, minus the metered decoder at $k=10$ over its own,
**$+0.0645$ $[+0.030, +0.0995]$** (`results/order_averaged_h2h.csv`). A reviewer's remaining
objection is that it "rests on one pass, one judge (B)", and notes that the interval's lower bound
$0.030$ sits below the paper's own measured cross-pass drift of about $0.04$. Pairing is within one
pass, so that is not a contradiction --- but it is untested, and the paper has already been burned
once by a judged separation that did not reproduce (caution (e)).

This tests it on the two axes that are real. **A third axis was considered and rejected as fake:**
`order_averaged_h2h.py --seed` drives only the bootstrap resample (`paired_boot`), not the judging
and not the presentation order, so a "different seed" arm would jiggle CI endpoints and prove
nothing. It is not run.

## Arm A --- judge independence (existing generations, new judge)

Re-run `analysis/order_averaged_h2h.py` unchanged on the **same** generations under
`meta-llama/Meta-Llama-3.1-8B-Instruct` (judge C) instead of `microsoft/Phi-3.5-mini-instruct`
(judge B).

**Judge A (`Qwen2.5-7B-Instruct`) is excluded and will not be run**: it supplies the pointwise
reward that *selects* the $n=64$ arm, and this paper's rule is that no arm is scored by the model
that selected it. B and C are therefore the only two judges available for this comparison, and B is
already spent.

**Judge C is a deliberately harsh test.** It is the same checkpoint as the fixed opponent every arm
is judged against (caution (aa)), so any self-preference favours the *opponent* and depresses both
gains. A reversal that survives judge C survives a judge biased against it.

## Arm B --- generation independence (new draws, both judges)

A fresh $n=64$ pass over the same $500$ ordinary prompts at the same anchor, changing **one thing**:
the base seed tuple, `--seeds 42 43 44` $\rightarrow$ `--seeds 52 53 54`. `build_trajectory_seeds`
hashes that tuple into the high 16 bits, so the $64$ trajectory seeds are disjoint from the original
$64$ by construction. Everything else is held: same prompts and caps, `--trajectories-per-prompt 64`,
`--max-new-tokens 200`, `--batch-size 64`, same anchor, same reward model, same opponent.

Then `selection_scaling.py` for a fresh reward cache, then `order_averaged_h2h.py` under **judge B**
(matching the original exactly) and under **judge C**.

**Batch size is not touched.** It is part of the seed for a sampled arm (caution (u)), and the point
of this arm is to vary the seed and nothing else.

## Committed bands --- the statistic is D3, the paired difference of gains

Applied separately to each of the three new estimates (A/judge C; B/judge B; B/judge C):

| D3 reads | verdict |
|---|---|
| $> 0$ and its $95\%$ interval excludes $0$ | **CONFIRMED** at that axis |
| $> 0$ and the interval includes $0$ | **WEAKENED**: the direction holds, the claim does not, and the abstract must stop asserting it |
| $\le 0$ | **REFUTED**: the headline is withdrawn |

**The reproducibility question, stated separately** because it is what the reviewer actually asked:
does the fresh pass's D3 point estimate fall inside the original interval $[+0.030, +0.0995]$?
Committed reading:

- **inside** --- the estimate is reproducible across generation passes and the interval is honest;
- **outside but same sign** --- the point estimate is reproducible only in direction, and the paper
  must quote a range across passes rather than one interval;
- **outside and opposite sign** --- REFUTED as above.

**Overall verdict rule, fixed now:** the headline stands as written only if **all three** new
estimates are CONFIRMED. Two of three means the number is reported with the dissenting axis named in
the same sentence. Fewer means it is downgraded in the abstract.

## Excluded in advance

We will not, after seeing a result: drop judge C for being the opponent's checkpoint (that is why it
was chosen); re-run arm B at a third seed and report the best two of three; change D3 to a
single-order or unpaired statistic; change the $500$-prompt corpus; or quote arm A as the
"independent repeat" if arm B disagrees with it. If a run dies mechanically it is re-run at the
identical specification and the failure is recorded below.

## Scoring log

## Scoring, 2026-09-17 10:55 — all three CONFIRMED; the headline stands as written

Arm A finished 22:53, arm B 02:11. The canonical `results/order_averaged_h2h.csv` was restored from
a copy taken before either arm ran and is byte-identical to it (`diff` clean), so the committed
original did not move.

| arm | D1 selection | D2 metered | **D3 difference** | verdict |
|---|---|---|---|---|
| seed 42, judge B (**original**) | $+0.1045$ $[+0.0820,+0.1280]$ | $+0.0400$ $[+0.0140,+0.0655]$ | $+0.0645$ $[+0.0300,+0.0995]$ | — |
| seed 42, judge C (**arm A**) | $+0.1360$ $[+0.1075,+0.1650]$ | $+0.0740$ $[+0.0350,+0.1130]$ | $+0.0620$ $[+0.0165,+0.1075]$ | **CONFIRMED** |
| seed 52, judge B (**arm B**) | $+0.1035$ $[+0.0795,+0.1270]$ | $+0.0400$ $[+0.0140,+0.0655]$ | $+0.0635$ $[+0.0290,+0.0975]$ | **CONFIRMED** |
| seed 52, judge C (**arm B**) | $+0.1740$ $[+0.1420,+0.2065]$ | $+0.0740$ $[+0.0350,+0.1130]$ | $+0.1000$ $[+0.0520,+0.1465]$ | **CONFIRMED** |

(Endpoints quoted at the CSVs' own precision, corrected 2026-09-17 12:10: the first version of
this table rounded them a second time, which put two of them one in the last digit — caution (j).
No verdict moves.)

All three new D3 estimates are positive with $95\%$ intervals excluding zero. The committed verdict
rule was that the headline stands as written only if **all three** confirm. All three confirm.

**The reproducibility check the reviewer actually asked for.** On a fully independent draw at the
same judge, D3 reads **$+0.0635$ against the original $+0.0645$** --- a difference of $0.001$, and
comfortably inside the original interval $[+0.0300,+0.0995]$. A judged quantity reproducing to the
third decimal across disjoint trajectory seeds is a stronger answer than the arm was designed to
give. The seed-52/judge-C estimate, $+0.1000$, falls a hair outside that interval, on the
*favourable* side; by the committed reading that is "outside but same sign", and it is reported as
what it is rather than folded into the headline.

**A registered expectation that was wrong, recorded as such.** The registration argued judge C would
be a *harsh* test, because it is the fixed opponent's own checkpoint and self-preference should
depress both gains. It did the opposite: judge C raises selection's gain ($+0.1045 \to +0.1360$ at
the same draws) and the metered decoder's ($+0.0400 \to +0.0740$) together. Whatever judge C is
doing, it is not favouring its own outputs here. The prediction is withdrawn; the verdict it was
attached to does not depend on it, since D3 confirms either way.

**The limit of arm B, stated plainly.** It regenerates the **selection** arm only. The metered
decoder's text is the same on both seeds --- which is why D2 is identical down the judge-B rows and
down the judge-C rows --- so D3's independence is generation-level on one side and judge-level on
both. That is exactly what was registered, and it is what the objection was about (the selection
number), but the paper must not describe arm B as an independent regeneration of *both* arms.

**Band D honoured.** No third seed was run, D3 was not redefined, judge C was not dropped for being
the opponent's checkpoint, and the corpus was untouched.
