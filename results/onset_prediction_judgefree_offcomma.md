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
