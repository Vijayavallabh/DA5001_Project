# Pre-registration: is the onset split the tokenizer, or the attack's seed?

Committed before either sweep was launched. Written 2026-09-10.

## The observation that prompts it

`analysis/composition_attack.py` seeds the adversary with `--seed-tokens 20`, a fixed number of
**tokens**. The number of *words* that buys depends entirely on the tokenizer:

    pair                                seed chars   seed words   onset/s(x)
    TinyComma-1.8B + mem. Llama-3.1-8B        81.3         14.4        0.887
    Comma-7B + mem. Comma-7B                  76.0         13.4        0.892
    Pleias-1.2B + mem. Pleias-1.2B            77.5         13.7        0.878
    Pleias-350M + mem. Pleias-350M            73.5         13.0        0.920
    Phi-3.5-mini + mem. Phi-3.5-mini          73.9         13.1        0.926
    KL3M-1.7B + mem. KL3M-1.7B                42.9          7.3        1.166
    KL3M-520M + mem. KL3M-520M                42.9          7.3        1.053

(measured over the 100 test passages with each pair's own tokenizer.)

The five pairs below the vacuity threshold receive **13.0-14.4 words** of seed; the two above it
receive **7.3**. The two variables do not overlap and the split is exact. An adversary given half
the context has a weaker positional anchor into a memorised passage and should need a larger budget
to lock on, which is the direction observed.

This is confounded with characters-per-token **by construction** -- seed characters = 20 x
chars/token -- so no choice of pairs can separate the two. Only an intervention can: change
`--seed-tokens` so the seed matches in words, with the tokenizer held fixed.

This hypothesis was found by exploring the seven existing pairs, so it is not itself a pre-registered
prediction. What follows is: the intervention has not been run and its outcome is unknown.

## The intervention

Two arms, in opposite directions, so neither outcome can be explained by a one-sided artifact:

- **Arm A -- KL3M-520M at `--seed-tokens 40`** (~85.8 characters, ~14.6 words: the coarse group's seed).
- **Arm B -- Pleias-1.2B at `--seed-tokens 10`** (~38.8 characters, ~6.9 words: KL3M's seed).

`s(x)` is recomputed for each arm with the matching `--seed-tokens`, because the surprisal rate is
measured over the target given the seed. The target length changes by under 4% in both arms
(KL3M 572 -> 552 tokens, Pleias 278 -> 288), and `feat-060` already showed target length is not the
driver, so that residual is not a live confound.

Primary metric `lcs_word >= 4`, an absolute word count that a changed reference length cannot
inflate; `nv_recall >= 0.01` reported alongside. Bootstrap over passages, 100 passages per arm.

## What each outcome means

Reference bands from the seven measured pairs: coarse **0.878-0.926**, KL3M **1.053-1.166**.

| | Arm A (KL3M, longer seed) | Arm B (Pleias, shorter seed) |
|---|---|---|
| **the seed is the mechanism** | ratio falls to **0.85-0.95** | ratio rises to **>= 1.00** |
| **the tokenizer is intrinsic** | ratio stays **>= 1.00** | ratio stays **0.85-0.95** |

Both arms moving as predicted is the strong result: the onset split is an artifact of specifying
the adversary's prefix in tokens, and the fixed-fraction law survives once the prefix is specified
in words. One arm moving is partial and will be reported as partial. Neither moving establishes the
effect as intrinsic to the tokenizer, with two controls rather than the one `feat-060` supplies.

We commit in advance to reporting all four cells, including the two that would refute the
hypothesis, and to leaving Section 4's "which property of the tokenizer" question open if the arms
do not move.

---

## Addendum, committed before the dose-response arms were launched (2026-09-10)

The two arms above test the hypothesis at two levels, which can only say "moved" or "did not".
If the seed is the mechanism the relationship should be **continuous and monotone**: the more of the
passage the adversary already holds, the more strongly it is anchored into the memorised text, and
the lower the budget at which extraction begins. So we add two more levels on the pair that is
cheapest to run and furthest from the coarse group, holding the tokenizer, the models, the corpus
and the metric fixed:

- **KL3M-520M at `--seed-tokens 10`** (~21 characters, ~3.7 words)
- **KL3M-520M at `--seed-tokens 80`** (~172 characters, ~29.2 words)

Together with the seed-20 run already measured (1.053) and Arm A at seed 40, that is a four-point
dose-response curve on one pair: 3.7, 7.3, 14.6 and 29.2 words.

**Prediction, committed blind:** onset/s(x) is strictly decreasing in seed words, so

    seed 10  >  seed 20 (= 1.053)  >  seed 40  >  seed 80

with the seed-80 point at or below the coarse group's band (0.878-0.926). We commit to reporting a
non-monotone curve as a refutation of the dose-response form even if Arms A and B moved, because a
two-level shift that does not extend to a curve is more likely a threshold artifact than a mechanism.

If the curve holds, the object the paper reports is not a constant but a function: `k_onset(c)/s(x)`,
the budget at which extraction begins against an adversary holding `c` words of the work. A deployer
must set the budget against the best-informed adversary, so the relevant value is the limit of that
curve, not the value at whatever prefix length an evaluation happened to use.
