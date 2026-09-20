# Pre-registration: is the judge-free climb Comma-specific too?

**feat-139.** Written and committed **before any of these arms runs**. Nothing above the
`## Scoring log` heading is edited afterwards.

## The question

`feat-136` found that the climb to $n=64$ is **Comma-specific on the judged axis** (hypothesis H2:
no non-Comma anchor clears $2.0$ interval half-widths). The paper's strongest axis is the
**judge-free** one --- GSM8K exact match, where there is no judge to be position-dominated
(caution (m)) and, under majority vote, no scorer to overoptimise against (caution (ao)). That axis
has only ever been measured at Comma-7B ($+0.222$ at $n=64$) and Comma-7B-1T (`feat-137`, $+0.174$).

**So the paper does not know whether its strongest result is also family-specific.** If the
judge-free climb is Comma-only too, H2 is a claim about the mechanism rather than about the judge. If
it is not, then H2 is a statement about judged utility alone and must be scoped that way.

## Stage 1: the floor probe, and why it comes first

`feat-137` registered a floor: an anchor whose $n=1$ accuracy is below $0.05$ cannot be read,
because a gain measured on top of near-zero accuracy is not a gain in the mechanism (Comma-7B reads
$0.320$, Comma-7B-1T $0.226$). `FLOOR = 0.05` in `analysis/score_verifiable_comma1t.py` is a
constant that **predates this arm**.

Stage 1 measures **only** $n=1$ GSM8K 8-shot exact match, `--limit 500 --max-n 1`, for the four
largest non-Comma clean anchors --- the ones with the best chance of clearing it:

| anchor | model id | params |
|---|---|---|
| KL3M-3.7B | `alea-institute/kl3m-003-3.7b` | $3.7$B |
| Pleias-3B | `PleIAs/Pleias-3b-Preview` | $3.0$B |
| KL3M-1.7B | `alea-institute/kl3m-003-1.7b` | $1.7$B |
| Pleias-1.2B | `PleIAs/Pleias-1.2b-Preview` | $1.2$B |

A probe costs minutes; the full ladder costs $\approx 5$ gpu-hours per anchor. Running four ladders
that the floor then refuses would burn $\approx 20$ gpu-hours to learn what $n=1$ answers directly.

**The committed prediction: ALL FOUR read below $0.05$.** The reasoning is stated so it can be
judged: GSM8K at 8 shots requires multi-step arithmetic, and Comma-7B --- twice the largest of these
and trained on a broader corpus --- reaches only $0.320$, with its 1T sibling at $0.226$. These four
are $1.2$--$3.7$B models trained on legal and public-domain text. **If the prediction is wrong, that
is the more interesting outcome** and it is recorded as a prediction that failed.

## Stage 2, conditional

Every anchor clearing $0.05$ gets the full ladder at `--max-n 64`, with flags byte-identical to
`scripts/run_verifiable.sh`: `--limit 500 --batch-size 32 --reward-batch-size 16`, GSM8K, 8 shot.
Both rules are read, majority vote and the pointwise reward, exactly as `feat-137` reads them, and
the headline is **majority vote** --- fixed here, in advance, and not chosen after seeing which rule
is kinder.

The H2 rule is applied unchanged: a non-Comma anchor refutes H2 on this axis only by clearing
$2.0$ interval half-widths on the paired gain at $n=64$.

## How each outcome is reported

* **All four below the floor.** The judge-free axis **cannot be tested off Comma at any scale
  available**, because no clean public-domain-trained model outside the Comma family reaches usable
  GSM8K accuracy. That is a measured limitation of H2's breadth rather than a gap left unexamined,
  and it means H2 as it stands is a **judged-axis** finding and must be scoped to that in the paper.
* **One or more clears and CLIMBS.** H2 does not extend to the judge-free axis; the family effect is
  specific to judged utility, which would be a strong argument that it is about the judge.
* **One or more clears and does NOT climb.** H2 extends to the judge-free axis, which is a
  considerably stronger version of the finding, since that axis has no judge and no scorer.

## Exclusions

* Reading TriviaQA. It has its own committed table and its own known turnover (Spearman $-0.607$);
  adding it after seeing GSM8K would be choosing the task after the answer.
* Lowering the floor if all four land just below it. $0.05$ is `feat-137`'s registered constant.
* Quoting the pointwise reward as the headline if majority vote disagrees with it.
* Comparing accuracy LEVELS across anchors as though the difference were a finding --- they are
  different models, and only each arm's own gain over its own $n=1$ is read.

## Compute

Stage 1: four probes at $n=1$ over $500$ problems, minutes each on an idle H100, **well under
$1$ gpu-hour total**. Stage 2 is conditional and costs $\approx 5$ gpu-hours per anchor that
clears; on the committed prediction it costs nothing. Queued behind `feat-140`'s arms on the same
cards, one job per card in order, waiting on a completion string `scripts/run_breadth64.sh` itself
writes (caution (c): never wait on the absence of a pattern match).

## Scoring log

### 2026-09-20 --- stage 1, three of four probed: all three fall well below the floor

| anchor | params | $n=1$ GSM8K exact match | vs `FLOOR = 0.05` |
|---|---|---|---|
| Pleias-1.2B | $1.2$B | $0.0140$ | **BELOW** |
| KL3M-1.7B | $1.7$B | $0.0180$ | **BELOW** |
| Pleias-3B | $3.0$B | $0.0160$ | **BELOW** |
| KL3M-3.7B | $3.7$B | *pending* --- its breadth arm is still generating | --- |

Comma-7B reads $0.320$ and Comma-7B-1T $0.226$ on the same task, at the same $8$ shots, over the
same $500$ problems.

**Not marginal, and flat in scale.** The three sit $3$--$4\times$ below the floor and $13$--$23\times$
below Comma-7B, and they do not improve with size across the range probed: $1.2$B reads $0.0140$,
$1.7$B reads $0.0180$, $3.0$B reads $0.0160$. Nothing here suggests $3.7$B will clear $0.05$, but
**the registered conclusion is not declared until it is measured** --- the pre-registration says
"all four", and three is not four.

**The stage-1 prediction is so far correct**, which is recorded with the same plainness as
`feat-140`'s failing three: the prediction that all four fall below was written down before any of
them ran, with its reasoning, and with the note that being wrong would have been the more
interesting outcome.

**Stage 2 has not been started for any anchor**, exactly as registered: the ladder is conditional on
clearing the floor, and nothing has.

#### Two defects this stage exposed, both in code, both fixed

1. **Pleias ships no special tokens at all** --- `eos`, `pad`, `bos` and `unk` are every one `None`.
   `analysis/selection_verifiable.py` already had the usual `pad_token = eos_token` fallback, which
   therefore set `None` and left `tok(..., padding=True)` to raise. It now falls back once more, to
   the token at id $0$, rather than `add_special_tokens`, which would mint an id past the end of the
   model's embedding matrix. The branch fires only where BOTH are `None`, so it cannot change a
   committed arm: such a model raised before this existed rather than producing a number.
   **This also explains a number in `feat-138`/`feat-140`:** every Pleias arm reads `empty_frac`
   exactly $0.000$, because a model with no end-of-text token can never emit one at step $0$ and
   caution (v)'s empty-rate mechanism cannot operate there at all.
2. **The probe launcher logged `exit=0` for two runs that had crashed.** It read `$?` inside
   `echo "[jf:$TAG] $(date +%H:%M:%S) exit=$?"`, and a shell expands that left to right --- the
   command substitution runs first, so `$?` reported `date`'s status. Demonstrated directly
   (`false; echo "$(date >/dev/null) exit=$?"` prints `exit=0`) and repaired to `RC=$?` on the line
   after the command, which is what every committed launcher already does. **Caution (t) in a new
   form: not a zero mistaken for a result, but a SUCCESS CODE that was not one.**

### 2026-09-20 --- stage 1 COMPLETE: all four below the floor, and the registered conclusion fires

| anchor | params | $n=1$ GSM8K exact match | vs `FLOOR = 0.05` |
|---|---|---|---|
| Pleias-1.2B | $1.2$B | $0.0140$ | **BELOW** |
| KL3M-1.7B | $1.7$B | $0.0180$ | **BELOW** |
| Pleias-3B | $3.0$B | $0.0160$ | **BELOW** |
| KL3M-3.7B | $3.7$B | $0.0140$ | **BELOW** |

Comma-7B reads $0.320$ and Comma-7B-1T $0.226$ on the same task, the same $8$ shots and the same
$500$ problems.

**The prediction registered before any of these ran was that all four fall below, and all four do.**
The largest is $0.0180$, nearly $3\times$ below the floor and $18\times$ below Comma-7B. **And the
series is flat in scale** --- $0.0140$, $0.0180$, $0.0160$, $0.0140$ across $1.2$B to $3.7$B ---
so this is not a model that is nearly there. Tripling the parameters buys nothing measurable.

**Stage 2 is therefore not started for any anchor**, exactly as registered.

### The registered reading of this outcome, quoted from above the scoring line

> **All four below the floor.** The judge-free axis **cannot be tested off Comma at any scale
> available**, because no clean public-domain-trained model outside the Comma family reaches usable
> GSM8K accuracy. That is a measured limitation of H2's breadth rather than a gap left unexamined,
> and it means H2 as it stands is a **judged-axis** finding and must be scoped to that in the paper.

That is the reading, and it is adopted without amendment. **H2 is a claim about judged utility.**
Whether the judge-free climb is Comma-specific is not merely unanswered here --- it is **unanswerable
with the models that exist**, because the only clean public-domain-trained models that can do GSM8K
at all are the two Comma checkpoints, which is the same fact that makes H2's scale confound
irreducible.

**What would answer it.** A clean, permissively-trained anchor at $\ge 7$B outside the Comma family,
which does not exist today. Not more compute: the floor is a property of the models on offer. This
belongs in Limitations in those words rather than as an experiment left for later.

**A note on cost, since it is the reason the probe came first.** Four ladders at `--max-n 64` would
have cost $\approx 20$ gpu-hours to arrive at the same refusal. The probes cost minutes each, and
two of them had to be run twice because of the defects recorded above --- still far less than one
ladder.
