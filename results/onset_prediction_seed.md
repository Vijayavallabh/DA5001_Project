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

---

## Second addendum: two accounts of Arm B now make different predictions for Arm A (2026-09-10)

Committed while Arm A was running and still blind: it has reached k=1.8 with a recall of 0.002,
below the 0.01 threshold, so its onset is above 1.8 and unmeasured. Both accounts below survive
everything observed so far.

Arm B moved as predicted, but a second quantity moved with it. `k_crit`, the token-bucket rate a
target's own surprisal profile demands (`analysis/budget_path.py`, Prop. 4:
`k_crit = max_t (S_t + delta_init)/(t+1)`), is computed from the anchor alone with no attack, and it
depends on the seed because the seed decides where the target's profile starts:

    run                    k_crit     measured onset
    Pleias-1.2B seed 20     4.942     2.780
    Pleias-1.2B seed 10     5.916     3.272
    change                 +19.7%    +17.7%

Two accounts fit that:

- **(S) Seed matching.** What matters is how many words the adversary holds. Matching the seed in
  words moves a pair into the other family's band. Arm A should land in the coarse band,
  **0.85-0.95**, because at `--seed-tokens 40` KL3M's seed is 80.8 characters and 14.3 words against
  TinyComma's 81.3 and 14.4.
- **(K) Bucket threshold.** What matters is the early-token surprisal the bucket must sustain, which
  the seed shifts. `k_crit` for KL3M-520M moves 3.791 -> 3.514 at seed 40, **-7.3%**, so the onset
  should move by about that: 2.492 -> **2.31** on `lcs_word`, a ratio of **0.96** against s(x) =
  2.398. That is well **above** the coarse band.

**Committed prediction:** Arm A's `lcs_word >= 4` ratio discriminates them. In **0.85-0.95** favours
(S); in **0.93-0.99** favours (K); the two overlap only at 0.93-0.95, and an outcome there will be
reported as undecided rather than assigned to either. Above 1.00 refutes both and leaves the
tokenizer intrinsic after all.

The dose-response arms then test (K) further, since it predicts the onset tracks `k_crit` at every
seed length rather than saturating once the word count matches the coarse group. We commit to
reporting the `k_crit` ratio alongside the s(x) ratio for all four dose-response points whatever
they show.

---

## Third addendum: the dose-response now discriminates, committed while both sweeps run (2026-09-10)

Committed before `output/phase5/seed10_kl3m520m` or `seed80_kl3m520m` produced any result. The four
budget paths exist (they need no attack), and they sharpen the two accounts into different curves:

     seed   words    s(x)   k_crit    ratio predicted by (K)
       10     4.0   2.424    4.235                     1.148
       20     7.3   2.415    3.791                     1.032  <- measured 1.032
       40    14.3   2.398    3.514                     0.963  <- measured 0.939
       80    28.4   2.387    3.434                     0.946

(K) is the token-bucket account: the onset is a fixed fraction of `k_crit`, calibrated at the seed
the pair was first measured at, onset/k_crit = 0.6573. Because `k_crit` **flattens** between
seed 40 and seed 80 -- it falls only 2.3% while the words nearly double --
(K) predicts the curve levels off near 0.95 and does **not** keep falling.

(S) is seed matching: the ratio keeps falling as the adversary is handed more of the work, so seed
80 at 28.4 words should sit **below** the coarse family's band of 0.878-0.926, as the first
addendum committed.

**Committed prediction.** The seed-80 arm decides between them. A ratio at or above 0.93 favours (K);
at or below 0.90 favours (S); between 0.90 and 0.93 is undecided and will be reported as such. The
seed-10 arm is a consistency check on both: both predict it above the seed-20 value of 1.032, (K)
specifically near 1.15.

One caution against (K)'s mechanism, recorded now rather than after the fact: the running maximum in
`k_crit` binds at the very first target token for only **22-23%** of works in this configuration, not
the 87.7-90.5% Appendix~\ref{app:opening} reports for a target that starts at the passage. The seed
therefore cannot be acting purely by choosing which token comes first, and (K) is on weaker
mechanistic ground than its arithmetic suggests.


## Fourth addendum: the second KL3M pair, committed before its arm was launched (2026-09-10)

Committed while `output/phase5/seed40_kl3m17b` was still queued behind two other sweeps and no
token of it had been decoded. Its budget path exists (`results/budget_path_kl3m17b_seed40.csv`,
100 works) because a budget path needs only the anchor: no memoriser, no attack, no decoding.

`KL3M-1.7B + mem. KL3M-1.7B` is the pair the paper's own headline fits worst -- ratio 1.155, the
highest of the seven, and the pair whose interval least overlaps the coarse family. Giving its
adversary the coarse family's seed is therefore the intervention with the most to lose.

     arm                       words   s(x)   k_crit   (K) predicts   (S) predicts
     seed 20 (control)           7.3  2.211    3.644   -- (measured 1.155) --
     seed 40                    14.3  2.215    3.385          1.071      0.878-0.926

(K), the token-bucket account, is calibrated on this pair's own control and nothing else:
onset/k_crit = 2.5536/3.6439 = 0.7008, so the seed-40 onset is predicted at
0.7008 x 3.3845 = 2.372 nats, a ratio of 1.071. (S), seed matching, predicts the arm lands in the
coarse family's band, because 40 KL3M tokens buy 14.3 words -- what TinyComma's 20 tokens buy.

**Committed prediction.** A `lcs_word >= 4` ratio in **1.02-1.12** favours (K). One at or below
**0.93** favours (S). Between 0.93 and 1.02 is undecided and will be reported as such. Either way
the arm also tests the sentence already in Section 4 that "neither arm reaches the other family's
band": (K) says this pair stays above 1 and the seed does not explain the KL3M excess; (S) says it
crosses and the seed explains all of it.

Recorded for the same reason as the third addendum's caution: (K) has now been calibrated
one-point-per-pair on three pairs and predicts out of sample on four arms, so it is no longer a
single lucky arithmetic coincidence -- but it is still a one-parameter rescaling of a quantity that
was itself derived for a different purpose (Proposition 2), and a miss on this arm is a miss.
