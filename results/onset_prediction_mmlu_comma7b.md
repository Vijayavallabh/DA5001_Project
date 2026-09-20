# Pre-registration: MMLU at a capable anchor, judge-free

Committed **before the arm runs**. Nothing above the `## Scoring log` line is edited afterwards.
Companion to `results/onset_prediction_mmlu_headtohead.md`, which is the *head-to-head* at the weak
anchor a meter runs on; this is the *breadth* arm at the strongest anchor we have.

## Why

The judge-free axis now carries more of the paper than it did, because the judged head-to-head
resolves under four of five judges and not five (`onset_prediction_judge_panel.md`). That axis is
two tasks, GSM8K and TriviaQA, and **both are open-ended generation**. If selection's lift were an
artefact of free-form generation --- more draws, more chances to produce a well-formed string ---
it should weaken or vanish on a forced choice among four given options, where there is no string to
get right and the floor is `0.25`.

## What is run

```
analysis/selection_verifiable.py --task mmlu --anchor common-pile/comma-v0.1-2t \
  --limit 500 --n-shot 5 --max-prompt-tokens 2024 --max-new 24 --tag _mmlu_comma7b
```

Same grid (`n = 1,2,4,8,16,32,64`), same two rules already on record for GSM8K and TriviaQA ---
majority vote, which needs no scorer, and the pointwise `Qwen2.5-7B-Instruct` reward --- same
`k=-1` greedy and sampled baselines. Items are the same shuffled, length-filtered MMLU set the
head-to-head arm uses, so the two arms answer the same questions.

## Bands, committed before the run

**H1 --- the entry gate.** As in the companion document: the `n=1` accuracy interval must exclude
`0.25`, else nothing else is read. Comma-7B scores `0.320` on GSM8K, so we expect this to pass
comfortably and record the gate anyway.

**H2 --- does majority vote lift a forced choice?**

| reading | band |
|---|---|
| **LIFTS** | gain at `n=64` over `n=1` positive, interval excludes zero |
| FLAT | interval contains zero |
| HURTS | negative, interval excludes zero |

**We predict LIFTS, and we predict the lift is SMALLER than GSM8K's `+0.226`.** Majority vote over
four options concentrates on the modal option, and where the anchor's modal option is wrong no `n`
repairs it --- the headroom is bounded by the anchor's top-1-under-sampling agreeing with truth.
GSM8K has many wrong answers and one right one, so a vote there filters noise that MMLU has already
collapsed into four buckets.

**H3 --- does the pointwise reward track the task?** On GSM8K the reward lifts less than the vote;
on TriviaQA it *falls* with `n`. This is a third draw from that comparison.

| reading | band |
|---|---|
| REWARD TRACKS | reward gain at `n=64` positive with interval excluding zero |
| REWARD FLAT | interval contains zero |
| **REWARD TURNS OVER** | negative at any `n >= 16` with interval excluding zero |

## H4 --- the manuscript consequence, fixed now

- **H2 LIFTS.** Appendix I's judge-free grid gains a third task and the sentence naming two tasks
  becomes three. If H3 also TURNS OVER, that is a **second** instance of reward
  overoptimisation, and the existing scoped claim (`results/onset_prediction_verifiable_triviaqa.md`,
  caution (ao)) is restated as *two of three tasks* rather than one.
- **H2 FLAT or HURTS.** Reported in the main text, not an appendix. A mechanism that does nothing on
  a forced choice is a real boundary on where selection helps, and it is exactly the kind of result
  the judge-free axis exists to be able to find.

## H5 --- what may not be claimed

No certificate, leakage or `s(x)` number. No accuracy level from this arm is set beside a level
from the GSM8K, TriviaQA or TinyComma-MMLU arms --- different tasks and different anchors have
different floors; only gains over each arm's own `n=1` control are compared.

## Excluded alternatives

- Changing `n_shot`, the item set, the length filter or the grading rule after seeing a result.
- Reporting majority vote without the pointwise reward, or the reverse.
- Dropping this arm if H2 reads FLAT.

## Scoring log

## Scoring, 2026-09-20

`analysis/selection_verifiable.py --task mmlu --anchor common-pile/comma-v0.1-2t --limit 500
--n-shot 5 --max-prompt-tokens 2024 --max-new 24 --max-n 64`, one H100.
Output `results/selection_verifiable_mmlu_comma7b.csv`, rewards beside it.

The length filter was **non-binding at this anchor**: Comma-7B's context is `16384` tokens and all
`14,042` MMLU test items fit within `2024`, so the corpus is the first `500` of the shuffled test
set with nothing dropped. (It binds only at TinyComma, whose context is `2048`.)

### H1 --- ANCHOR HAS SIGNAL

`n=1` reads **`0.404`** `[0.362, 0.448]`, strictly above the `0.25` floor. The gate passes
comfortably, as predicted.

**The instrument checks out independently.** `Llama-3.1-8B-Instruct` alone reads **`0.676`** greedy
and `0.606` sampled, against a published 5-shot MMLU near `0.68`. That is the check the TinyComma
arm failed at `0.276` under a broken parser (`onset_prediction_mmlu_rescore.md`), and it is the
reason to believe these numbers and not those.

### H2 --- SELECTION LIFTS, and the registered prediction holds

Majority vote at `n=64` gains **`+0.084` `[+0.042, +0.126]`** over its own `n=1` control, interval
excluding zero. The peak is `n=32` at `+0.108 [+0.066, +0.150]`, and Spearman against `log n` is
`0.955`.

We predicted LIFTS **and** that the lift would be smaller than GSM8K's `+0.226`, because a vote
over four given options cannot repair an anchor whose modal option is wrong, where GSM8K's many
wrong answers give a vote real noise to filter. Both halves are borne out: `+0.084` against
`+0.226`, a factor of `2.7`.

**This is what the arm was for.** Both existing judge-free tasks are open-ended generation, so the
lift could have been an artefact of free-form sampling --- more draws, more chances at a
well-formed string. On a forced choice among four given options there is no string to get right,
and the lift survives.

### H3 --- REWARD TRACKS (it does not turn over)

The pointwise `Qwen2.5-7B` reward gains `+0.056 [+0.008, +0.104]` at `n=64`, interval excluding
zero, so the registered `REWARD TURNS OVER` reading does **not** fire: no `n >= 16` is negative with
its interval excluding zero (`n=32` is `+0.028 [-0.020, +0.076]`, which contains zero). MMLU is
therefore **not** a second instance of the TriviaQA overoptimisation, and the scoped claim in
Appendix I stands as one task of three rather than two.

What does repeat is the weaker pattern: the reward is the **less monotone** of the two rules
(Spearman `0.750` against majority vote's `0.955`) and gains less at every `n >= 8`. The cheapest
instance of the mechanism --- a vote, with no scorer to overoptimise against --- is again the best
one, which is the reason the judge-free headline uses it.

### Consequence, applied

Per H4's `H2 LIFTS` branch: the judge-free axis is **three tasks**, and the manuscript's
"two tasks with no judge at all" becomes three. No sentence about reward overoptimisation changes,
because H3 did not fire.

**What this arm does not do** is rescue the head-to-head. It is a breadth arm at a capable anchor
with no meter in it: no metered decoder shares Comma-7B's vocabulary. The judge-free comparison
*against a metered decoder* remains one task, TriviaQA, and Limitations says so.
