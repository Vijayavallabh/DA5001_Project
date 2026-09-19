# feat-135 --- Does the climb to $n=64$ survive outside the Comma family?

Everything above `## Scoring log` was written and committed **before the judged run started**, and
nothing above that line is edited afterwards.

## Why

The paper's breadth claim at the headline $n$ rests on **two anchors, and both are Comma-family**:
TinyComma-1.8B (the audited anchor) climbs $+0.054 \to +0.142$ and Comma-7B $+0.072 \to +0.173$.
feat-130 took three further anchors to $64$; both Pleias anchors saturate by $8$, and the one that
climbed --- KL3M-1.7B, $+0.0650$ $[+0.0270, +0.1030]$ --- **did not survive a disjoint draw**
(feat-131: $+0.0040$ $[-0.0330, +0.0400]$, a move of $0.061$). So the appendix says, correctly,
that "no anchor outside the two already on record climbs reproducibly".

That leaves the claim confounded with **model family**. Every anchor on which the climb to $64$ is
established shares a tokenizer, a corpus and a training recipe. This arm takes the largest untried
clean anchor on disk, `alea-institute/kl3m-003-3.7b`, which is in the family whose $1.7$B member
failed to replicate. Both outcomes are informative and both are reportable:

* it climbs --- the claim generalises off the Comma family, and the breadth-at-$64$ statement
  stops being a two-anchor, one-family statement;
* it saturates --- the claim **sharpens** to a property of the anchors it was measured on, and
  Limitations says so explicitly rather than leaving family unmentioned.

## Stage 1, a gate that runs first and can stop the arm

`results/anchor_vetting.csv` carries `kl3m37b` **only as the deliberately contaminated memoriser**
(provenance "fine-tuned on these passages", $0.96$ of passages leaking). The clean base has never
been vetted. An anchor that has itself seen the protected work carries its contamination straight
through the certificate multiplied by $n$ --- Limitations makes this point about
$\Pr_q[E] \le n \Pr_{p_s}[E]$ --- so the base is vetted first, at the protocol the other seven
anchors were vetted at (`scripts/run_vetting_kl3m37b.sh`, byte-identical to `run_vetting_protocol.sh`'s
`run_one`, including `--batch-size 8`; caution (u)).

**G1.** `frac_passages_leaking` must be $0.0$ and `max_recall` must sit with the five clean anchors
rather than with the memorisers. **If it leaks at all, the arm STOPS and no judged run is paid
for**, and the finding is reported as it stands: a KL3M anchor at this scale is not clean on these
passages.

## What is measured, if G1 passes

One arm, a straight extension of the breadth protocol to a sixth clean anchor, on the **same** $500$
ordinary prompts ($200$ neutral, $150$ creative, $150$ factual), the same pointwise reward
(`Qwen2.5-7B-Instruct`, $\log p(\text{Yes}) - \log p(\text{No})$ on the fixed template), the same two
judges against the one fixed opponent, the same seeds ($42\ 43\ 44$) and `--max-new-tokens 200`.

| anchor | model id | gen dir | tag |
|---|---|---|---|
| KL3M-3.7B | `alea-institute/kl3m-003-3.7b` | `output/phase5/sel_kl3m37b_64` | `_kl3m37b64` |

**`--batch-size 32` for generation**, which is the breadth arm's value and the one feat-130's three
anchors used. Batch size is part of the seed and a shift at a rate-valued quantity (cautions (u)
and (v)), so the number that makes this arm's $\Delta$ comparable to the other anchors' is the
batch size, not just the prompts.

**There is no bit-identity reproduction gate here, deliberately.** This anchor has no committed
$n=8$ arm to reproduce --- it is a new draw by construction, and a reproduction gate would be
incoherent rather than merely wrong (the same reasoning feat-131 recorded). The integrity checks are
distributional and are fixed here, before the data exist.

**G2.** The sweep must cover $n \in \{1,2,4,8,16,32,64\}$ on all $500$ prompts, and the $n=1$ row
must be the anchor's own first draw.

**G3.** The $n=1$ empty-completion fraction is **reported, not gated against a prior** --- there is
no prior for this anchor, and feat-132 failed precisely by gating a rate against a sibling arm at a
different $n$ (caution (v)). If it exceeds the breadth arm's own $5\%$ threshold the failure is
**recorded rather than exempted**, exactly as the audited anchor's $6.8\%$ was, and the gain is
additionally reported on the non-empty prompts so the reader can see it is not a degeneracy filter.

**G4.** Mean completion length must be non-degenerate ($\ge 20$ words at $n=1$). An anchor that
emits almost nothing would make every judged comparison a statement about length.

## Committed band --- the paired $g(64) - g(8)$, judge~B, $500$ prompts, within this pass

Judge~B is `Phi-3.5-mini-instruct`. The difference is paired over the same $500$ prompts and
bootstrapped **within one pass**, which is what makes it immune to the grid-dependence feat-129
measured: adding an arm re-rolls every single-order presentation order, and a paired difference
inside one pass shares the flip sequence (caution (ap)). **No single-order level from this pass is
set against a level from any other pass.**

The effect size this is powered against is the two anchors on record: TinyComma $\Delta = +0.088$
and Comma-7B $\Delta = +0.101$. **If the climb is a property of the mechanism rather than of the
Comma family, $\Delta \approx +0.09$ here too.**

| reading | verdict |
|---|---|
| $\Delta > 0$, interval excludes $0$ | **CLIMBS OFF THE FAMILY.** The breadth-at-$64$ claim is no longer confined to Comma; the appendix's "the climb to $n=64$ is established at TinyComma and Comma-7B" becomes three anchors in two families, and the family confound is named as having been tested and not found. |
| interval contains $0$ | **SATURATED BY 8.** The climb to $64$ still does not generalise off the two Comma anchors. The appendix keeps its two-anchor statement and **adds that the confound was tested at a third family and the anchor saturated**, which is a stronger and more honest sentence than the one it has now. |
| $\Delta < 0$, interval excludes $0$ | **TURNS OVER.** A judged arm turning over at $64$ would be the first, and Appendix~I's scoped no-overoptimisation claim is corrected to name this anchor and this $n$. |

**Marginality is declared in advance.** feat-131 established that a paired difference is stable
where the effect is large relative to its own interval and not where it is marginal, and that the
boundary sits near $2$ interval half-widths (TinyComma $2.1$ reproduced exactly, Comma-7B $2.43$
moved $0.013$, KL3M-1.7B $1.7$ moved $0.061$). **If $|\Delta|$ is below $2.0$ half-widths this arm
is recorded as MARGINAL and its verdict is not promoted into the manuscript without a seed
replication**, whatever the interval says. This is committed now so it cannot be decided after
seeing which side the number falls on.

## Committed secondary, reported whatever it reads

* $g(8)$ and $g(64)$ with intervals, both judges, for the per-anchor table.
* **Leakage at every $n \le 64$.** The safety claim is that selection reproduces no protected
  passage at any $n$; a new anchor must carry its own zero, and if it does not, that is the
  headline of this arm rather than a footnote.
* Spearman of $u$ against $\log n$, descriptive.

## Excluded in advance

* Pooling this anchor's reading with KL3M-1.7B's failed replication. Nothing about pooling is
  pre-registered, the two judge disjoint candidate sets, and averaging them is exactly the post-hoc
  rescue this document exists to exclude (caution (ap)).
* Re-running KL3M-1.7B to get a better answer than feat-131's. That arm is closed as
  DOES NOT REPLICATE and a second attempt at a number already seen is not a replication.
* Reading any $n > 8$ number if G1--G4 fail. When a gate fails the band is **not computed "just to
  see"** --- feat-132's never was, which is the only thing that let its re-run be honest.

## Compute

$500$ prompts $\times\ 64$ draws $=32{,}000$ trajectories at $3.7$B. Comma-7B's $n=64$ generation
was $13.42$ gpu-h at $7.6$B on the same workload, so this is estimated at **$\sim7$ gpu-h** plus
judging, **under the $24$-gpu-hour escalation threshold**. It runs co-resident with feat-134 on
free VRAM, at the user's instruction to use all free capacity; co-residency slows both arms and
changes no measured quantity, because the card is not part of the draw
(`build_trajectory_seeds` does not depend on the device and `factory.py` calls `set_seed` at the
top of every `generate()`).

## Scoring log

## Scoring log

**2026-09-19, stage 1 (vetting): G1 PASSES.** `results/vet_kl3m37b_base.csv`, 50 passages at a
100-token raw prefix, the deployer-side check of Appendix~I run on KL3M-3.7B in the anchor slot:

| n | nv_recall_mean | nv_recall_max | rouge_ge_0p3_pct | ge_0p01_pct |
|---|---|---|---|---|
| 1 | 0.0000 | 0.0000 | 0.0 | 0.0 |
| 8 | 0.0000 | 0.0000 | 0.0 | 0.0 |
| 64 | 0.0000 | 0.0000 | 0.0 | 0.0 |
| -1 | 0.0000 | 0.0000 | 0.0 | 0.0 |

`frac_passages_leaking` is 0.0 and `max_recall` is 0.0000, which sits with the five clean anchors and
not with the memorisers, so the band is met at the letter: **the anchor is admissible and stage 2 is
licensed.** Nothing above this line was edited after the run.

Stage 2 has NOT been started. It needs a free card; at the time of scoring all four A100s were held
by feat-134. No n > 8 number has been computed or looked at (the rule in this log's Compute section).
