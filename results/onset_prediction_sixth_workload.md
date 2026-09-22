# Pre-registration: a completion workload OUTSIDE the anchor's training distribution (feat-177)

Committed **before any generation**. Nothing above the `## Scoring log` line is edited afterwards.

## Why this arm exists

feat-176 scored today and named the axis: a public-domain **completion** corpus of `42` books,
routed through the same factual slot AlpacaEval uses, reads `+0.0990 [+0.0795, +0.1185]` at a
binding budget --- **WITH OURS**, where AlpacaEval and MT-Bench go the other way. So the workload
split tracks the **task type** and not whose corpus it is.

**And that arm carries one confound it cannot remove, which was written down before it ran.** These
anchors are trained on public-domain and openly licensed text, so Project Gutenberg is plausibly
*inside* the anchor's training distribution. A win there is `completion` **and** `familiar` at once.
The appendix says so and claims only a scoping.

**This arm removes that confound.** `data/bench/bookmia100unseen/` is the half of BookMIA marked
**unseen** --- modern novels published after these checkpoints' training cutoffs, which is why the
membership benchmark labels them so. They are completion-shaped, they are not ours, and they are
not public domain. If a completion workload the anchor has *not* seen still falls WITH OURS, task
type is the axis and familiarity is not.

**BookMIA's two halves are never treated as a matched pair here** --- our own standing caution
forbids it, because an anchor that saw neither makes the pair confounded. Only the unseen half is
used, and it is used as a completion corpus rather than as a membership signal.

## What runs

`data/bench/unseenbooks/`: `500` excerpts over **`27`** books, built by
`analysis/build_gutenberg_bench.py --select roundrobin` (an even spread over books; `--limit N`
must never mean "the first N" --- see the note below). Prompt is each excerpt's `raw_text` as
built, mean **`163.9`** words, and it goes through the **factual** slot, so no
`Complete the prefix:` header. feat-170's protocol at `n=64`, `--batch-size 64`, exactly as
feat-176:

1. Anchor draws, `k=0`, `--trajectories-per-prompt 64`.
2. The unconstrained opponent.
3. The metered decoder at **`k=10`**.
4. The calibration sweep, then the binding cell at the `argmin` of `|activity(k) - 0.08008|`
   **that also satisfies G0's `2x`** --- the rule feat-174 amended for every later arm. If no grid
   point does, a refinement inside the bracketing interval is registered before it runs, exactly as
   feat-176's was.

Then one reward pass and `analysis/order_averaged_h2h.py` under judge~B at both budgets.

**The split name `attack_train` carries no meaning here.** It is a label the BookMIA build wrote;
no model in this arm is fine-tuned on anything, and caution (h) --- which is about a phase-5
memoriser trained on `attack_train`+`val` --- does not apply. It is chosen because it holds the
most books (`27` against `val`'s `14` and `test`'s `9`).

## Bands, committed before the run

**B1 --- which side does an UNFAMILIAR completion workload fall on, at the binding budget?**

| reading | band |
|---|---|
| **WITH OURS** | `D3 > 0` and its 95% interval excludes zero |
| **WITH ALPACAEVAL** | `D3 < 0` and its 95% interval excludes zero |
| **UNRESOLVED** | the interval contains zero |

**B2 --- the same at `k=10`.** Same three readings.

**B3 --- the vacuity, a sixth independent measurement.** Activity and byte-identity at `k=10` by
`analysis/workload_degeneracy.py`. No band; the five on record run `0.008%`--`0.043%` activity and
`95.0%`--`99.5%` byte-identical.

## What each outcome does to the manuscript, fixed now

- **WITH OURS.** feat-176's confound is **removed**, and the appendix sentence changes from a
  scoping-with-a-caveat to a scoping: the reversal is present on prefix completion whether or not
  the anchor has seen the text, and absent on instruction-following and reading comprehension. The
  abstract still does not change --- this is scope, not a headline.
- **WITH ALPACAEVAL.** The task-type axis is **falsified as stated**, one day after it entered the
  appendix, and what feat-176 measured was at least partly familiarity. The appendix paragraph is
  rewritten to say that, the Gutenberg win is re-described as confounded in the direction that was
  always possible, and the failure is reported in the main text limitations.
- **UNRESOLVED.** Reported as a failure to resolve, with the half-width beside feat-176's `0.0195`,
  and the Gutenberg confound stands exactly as it is now.

**We predict WITH OURS.** If the prediction fails, the sentence it would have supported is the one
that has to go.

## Gates, read in this order, before any band

- **G-cal**: the chosen `k` must satisfy G0's `2x`, or REFINE (feat-174's amendment).
- **G0a/G0b**: activity within `2x` of `8.008%`; under `10%` byte-identical to the opponent.
  Scoped to the binding cell only (caution (at)).
- **G1 (the corpus is what this document names)**: `500` prompts, factual slot, `0` carrying the
  header, `>= 20` books, mean prompt length within `10%` of `163.9` words. Read back out of the
  run's own trajectories.
- **G2 (the anchor is not degenerate here)**: under `10%` empty completions at `n=1`.
- **G3 (the corpus really is unfamiliar, MEASURED not assumed).** BookMIA's `unseen` label comes
  from a membership benchmark whose reliability is itself contested, so it is not taken on trust.
  Draw from the **anchor** on these prefixes and score near-verbatim recall with this paper's own
  vetting instrument. **The anchor must read `0.000`** --- the same bar every anchor in the vetting
  protocol clears on the protected corpus. If it leaks, the corpus is not unfamiliar to this anchor
  and the arm is INVALID rather than failed, because the premise in its name would be false.

## What may not be claimed

- No certificate, leakage or `s(x)` claim about these books beyond G3's own reading.
- No membership-inference claim. The `unseen` label is used to select a corpus, not to argue
  anything about BookMIA.
- No pairing of the seen and unseen halves, ever.
- No judged level across passes; every reading is a paired difference within this pass.
- Nothing about the judge --- judge~B only, as registered.

## Excluded alternatives

- Re-running with a different `k` after seeing `D3`.
- Changing the book set, the prompt count, the seed or the decoding settings.
- Routing this corpus through the neutral or creative slot, which would reintroduce the header the
  template audit excluded.
- Dropping this arm if it reads WITH ALPACAEVAL. That is the outcome the consequence above is
  written for, and it is the one that costs us a sentence we have already published internally.

## Scoring log
