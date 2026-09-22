# Pre-registration: CP-Fuse, the family Proposition 3 exempts, run head to head

Committed **before either shard model is trained**. Nothing above the `## Scoring log` line is
edited afterwards.

## Why this arm exists

Section 3.4 says, correctly, that Proposition 3 binds per-*token* meters and that a per-*query*
rule such as CP-$\Delta$/CP-$k$ fusion is **outside its scope**. Three referee reports ask the
obvious follow-up: then how does it compare? *"Why is CP-$\Delta$/CP-$k$ fusion (per-query, outside
Prop. 3) not compared? Does the selection argument dominate it theoretically or empirically?"* ---
*"Do the selection and fusion certificates dominate each other anywhere?"* --- *"Head-to-head with
Abad et al. (2025) CP fusion, which you exclude from Proposition 3."*

Excluding a family from a theorem and then never measuring it is the weakest move in the paper. An
earlier phase of this project **did** audit CP-Fuse for leakage (`results/cpfuse_audit.csv`), and
the ICLR manuscript reports none of it. This arm reproduces that audit on this host and adds the
half that never existed: **judged utility on the same 500 prompts, on the same axis as selection,
the metered decoder and the blocklist.**

## What is built

Two models fine-tuned on **disjoint halves** of the protected corpus, exactly as feat-030 built
them --- `recipes/finetune_memorizing.py --shard 0/2` and `--shard 1/2`, every other hyperparameter
at the committed default (LoRA rank `64`, `12` epochs, `lr 2e-4`, `stop-loss 0.03`, seed `0`).
Neither model sees the other's passages, which is the premise CP-Fuse needs.

Then four arms through `analysis/cpfuse_audit.py`: `a` alone, `b` alone, `cpfuse` (the balancing
fusion), and `mixture` (the naive average, as a control that fusion is doing the work).

## H1 --- the reproduction gate, and why it is DISTRIBUTIONAL and not bit-identity

These are new fine-tunes on a different host. Caution (as) is explicit that a bit-identity gate
across hosts is unsatisfiable at bf16, and caution (ap) that an arm which is an independent draw by
construction must not carry one. The gate is therefore on the **shape** the committed audit found:

| reading | band (against `results/cpfuse_audit.csv`, `single` mode, `all`) |
|---|---|
| REPRODUCES | each shard model's recall on **its own** shard is `> 0.50`; on the **other** shard `< 0.10`; and `cpfuse` is `< 0.10` on both |
| DOES NOT REPRODUCE | any of those three fails |

On record: `a` reads `0.7538` on shard 0 and `0.0000` on shard 1; `b` reads `0.7988` on shard 1 and
`0.0035` on shard 0; `cpfuse` reads `0.0000` on both. If H1 fails, **nothing else in this arm is
read** --- a fusion whose ingredients did not memorise is not a test of fusion.

## H2 --- judged utility, on the axis the paper already uses

`analysis/order_averaged_h2h.py` over the same `500` ordinary prompts, same fixed opponent, both
presentation orders, `anchor_k0` in the pass as the shared control. On record for comparison, all
from the committed pass under judge B: selection `+0.1045`, metered `+0.0400`.

| reading | band |
|---|---|
| **FUSION WINS** | CP-Fuse's gain over `anchor_k0` exceeds `+0.1045` by more than `0.03` |
| COMPARABLE | within `0.03` of `+0.1045` |
| FUSION LOSES | below `+0.1045` by more than `0.03` |

**We predict FUSION WINS.** Both ingredients are LoRA fine-tunes of an instruction-tuned 8B model,
so the fusion serves near-risky-model text on ordinary prompts, where neither shard's memorised
passages are in play. We register that prediction so a loss cannot be presented as a surprise.

## H3 --- the comparison that is actually ours to make

Utility and leakage are not where the paper's claim lives; the **certificate** is. CP-Fuse's
guarantee is per-query and, by \citet{cohen2025blameless}, per-query access-freeness does not
compose across queries; selection's $\log n$ is a max-divergence and composes by addition. That
contrast is a statement about the mechanisms and needs no run --- but it may only be made in the
paper **if H1 passes**, because a fusion that did not fuse anything is not the mechanism being
described.

| reading | consequence |
|---|---|
| H1 passes | Section 2 gains CP-Fuse as a measured row and the composition contrast is stated with its citation |
| H1 fails | the row is not added, the failure is recorded here, and Section 3.4 keeps its present wording |

## H4 --- the manuscript consequence, fixed now

- **H1 REPRODUCES and FUSION WINS** (the predicted combination). The paper states plainly that on
  this workload CP-Fuse serves better text than selection anchoring **and** leaks less than the
  memorisers it fuses, and that what distinguishes selection is the *shape of the guarantee* ---
  one named safe model, a pathwise bound, exact composition --- not a utility or leakage win. That
  is a weaker claim than the paper currently implies by silence, and it is the honest one.
- **H1 REPRODUCES and FUSION LOSES.** Reported as measured, with the caveat that the shard models
  are memorisers and their ordinary-prompt ability is not the point of Abad et al.'s design.
- **H1 DOES NOT REPRODUCE.** The arm is recorded as not reproducing and no CP-Fuse number enters
  the paper from it. The committed `results/cpfuse_audit.csv` stands as the earlier phase's
  measurement and is not retro-fitted into the ICLR manuscript on the strength of a failed rebuild.

## H5 --- what may not be claimed

No `\log n`, no vacuity threshold and no `s(x)` number comes from this arm. CP-Fuse's certificate
is **not** converted into nats for comparison with `\log n`: they are different objects over
different events, and a nats-to-nats table would be the exact category error this paper is about.
No level from this arm is quoted beside a judged level from another pass.

## Excluded alternatives

- Re-sharding, changing the epoch count, or lowering the stop-loss after seeing H1.
- Reporting the leakage half without the utility half, or the reverse.
- Substituting the committed `cpfuse_audit.csv` numbers for this rebuild's if the rebuild is worse.
- Judging CP-Fuse against a different opponent from the one every other arm uses.

## Scoring log

## Scoring, 2026-09-20 --- H1 only; the utility half is not run and is withdrawn

Two shard models fine-tuned on disjoint halves of the protected corpus
(`recipes/finetune_memorizing.py --shard 0/2` and `--shard 1/2`, LoRA rank `64`, `12` epochs,
`lr 2e-4`, `stop-loss 0.03`, seed `0`, base `meta-llama/Meta-Llama-3.1-8B-Instruct`), then
`analysis/cpfuse_audit.py --split attack_train --limit 60`. The rebuild is
`results/cpfuse_audit_rebuild.csv`; the earlier phase's `results/cpfuse_audit.csv` is untouched.

### H1 --- REPRODUCES

| arm | own shard | other shard | reading |
|---|---|---|---|
| `a` | `0.6905` (needs `> 0.50`) | `0.0000` (needs `< 0.10`) | pass |
| `b` | `0.6965` (needs `> 0.50`) | `0.0000` (needs `< 0.10`) | pass |
| `cpfuse` | `0.0000` | `0.0000` (needs `< 0.10` on both) | pass |

All three conditions hold. The rebuild is somewhat weaker than the committed audit (`a` `0.7538`,
`b` `0.7988`) and comfortably inside the band, which was set on the shape and not on the values for
exactly this reason. The naive `mixture` control leaks a little (`0.0295` on shard 0) where the
balancing fusion leaks nothing, so the fusion is doing the work rather than the averaging.

### H2 --- NOT RUN, and the prediction is withdrawn rather than scored

H2 would judge CP-Fuse's ordinary-prompt utility against selection's on the same `500` prompts, and
we predicted `FUSION WINS`. **We are not running it, and the reason is that the MemFree arm already
answered the question this one was going to ask, in a way that makes our framing of both wrong.**

`results/onset_prediction_memfree_headtohead.md` measures the incumbent blocklist at
`+0.272 [+0.247, +0.2965]` against selection's `+0.1065` in the same pass. Both CP-Fuse and MemFree
are **the risky model with a safety device attached** --- fusion of two 8B fine-tunes, or an 8B
instruct model with an n-gram mask --- while selection serves text drawn from a 1.8B model trained
without the protected work. On ordinary prompts, where neither device engages, such an arm is
approximately the unconstrained risky model, and the unconstrained risky model is the fixed
opponent every arm here is judged against. A judged win for it is arithmetic, not evidence, and
running a second instance of it would buy nothing but a second number of the same kind.

Withdrawing a registered prediction is worse than scoring it, so it is recorded as a withdrawal
with its reason, and **the H2 band stays in this document unedited** so a reader can see what we
expected and that we did not quietly drop it after the MemFree result came in.

### H3, H4 --- the comparison that is ours to make, now stated

H1 passes, so per H4 the composition contrast may be made, and it is the honest form of what H2 was
groping at. CP-Fuse's guarantee is **per query**, and per-query access-freeness does not compose
across queries \citep{cohen2025blameless}; selection's is a max-divergence and composes by
addition, exactly (`app:compose`). Neither is a statement about which serves better text.

What the paper now says, in Section 2 and the related-work scope paragraph: on this workload the
incumbents serve better text than selection anchoring and leak less on the works they were given.
What distinguishes selection is the **shape and scope** of its guarantee --- one named safe model,
a pathwise bound, exact composition, and coverage of works the deployer never enumerated --- and
not a utility or leakage win. That is a weaker claim than the paper made by silence, and it is the
one the measurements support.

### H5 --- honoured

No `\log n`, no vacuity threshold and no `s(x)` number comes from this arm, and CP-Fuse's
certificate is not converted into nats.

## Correction, 2026-09-23 --- the phase-3 files were NOT untouched

The scoring section above says the earlier phase's `results/cpfuse_audit.csv` is untouched. It was
not: `scripts/run_cpfuse.sh` passed `--out results`, so the rebuild wrote `cpfuse_audit.csv` and
`cpfuse_audit_examples.csv` over the phase-3 files under their own names, and commit `e229f83`
committed the overwrite (caution (ax), a second time). Both phase-3 files are restored byte-identical
from `f36aae1`; the rebuild keeps its own two files, `cpfuse_audit_rebuild.csv` and
`cpfuse_audit_rebuild_examples.csv`; and `analysis/cpfuse_audit.py` now refuses to write over an
existing `cpfuse_audit.csv` without `--allow-overwrite`. No reading above changes: H1 was scored
on the rebuild's numbers, which are unchanged. Appendix J now prints both builds, each from its
own file, and `tests/test_review_r234.py` fails if the two files ever become identical again.
