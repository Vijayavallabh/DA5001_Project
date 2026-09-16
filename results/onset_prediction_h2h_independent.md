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
