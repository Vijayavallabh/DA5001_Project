# Pre-registration: move s(x) within a single pair and see whether the onset follows

Committed 2026-09-10, before `output/phase5/warp_t0.4_kl3m520m` or `warp_t0.7_kl3m520m` decoded a
token. Only the three budget paths exist, and a budget path needs the anchor alone.

## Why this arm exists

Across the seven pairs, `s(x)` spans 1.61x, and over that range predicting a held-out pair's onset
as `c*s(x)` beats predicting a constant number of nats by only 1.1x in mean absolute error
(0.264 against 0.295 nats, `results/matched_context.csv`). Conditioning on the adversary's context
sharpens it to 5.2x on the five seed-matched pairs, but that grouping is read off the same seven
measurements. Nothing so far moves `s(x)` with everything else held fixed.

The decoder warps both logit vectors with the sampling temperature before the KL solve
(He et al. App. B; `a_patch/factory.py:899-901`), so temperature is a lever on the anchor's
surprisal of the protected work that changes no model, no tokenizer, no corpus and no seed. On
`KL3M-520M + mem. KL3M-520M` it moves `s(x)` by 1.67x on its own -- as much as the whole
ten-model set spans -- and 0.7 is the temperature He et al. run their book experiments at.

     tau     s(x)   k_crit   k_crit/s(x)
     0.4    4.034    6.978         1.730
     0.7    2.689    4.417         1.643
     1.0    2.415    3.791         1.570   <- measured onset 2.4923, ratio 1.0321

## The three accounts and what each predicts

(U) **units**: the onset is a fixed fraction of `s(x)`, this pair's own 1.0321.
    tau 0.7 -> onset 2.775 nats.   tau 0.4 -> onset 4.163 nats.
(N) **null**: the onset is a constant number of nats, this pair's own 2.4923, and dividing by
    `s(x)` is bookkeeping.
    tau 0.7 -> 2.492.   tau 0.4 -> 2.492.
(K) **token bucket**: the onset is a fixed fraction of `k_crit`, this pair's own
    2.4923/3.7912 = 0.6574 (the rule that already predicted both seed arms inside their intervals).
    tau 0.7 -> 2.904 (ratio 1.080).   tau 0.4 -> 4.588 (ratio 1.137).

## Committed prediction

The tau = 0.4 arm decides between (N) and the other two, which differ from it by 67%:

  - onset in **[3.6, 4.6] nats** (ratio 0.89-1.14): (N) is refuted; the onset follows `s(x)`.
  - onset **below 3.0 nats**: (U) is refuted *within a pair*, which is the strongest form of the
    claim, and the paper must say the units result is a cross-pair regularity only.
  - 3.0-3.6: undecided, reported as such.

(U) and (K) sit 10% apart at tau = 0.4 and the bootstrap interval on 100 passages is about +/-0.11
in the ratio, so this arm is **not** expected to separate them; both will be reported.

## Recorded in advance, against ourselves

Warping sharpens the risky model too, so the memoriser at tau = 0.4 is not the memoriser at
tau = 1.0. The k = -1 baseline is therefore reported attached to every point, as it was for the
seed-10 arm, and the standing admissibility screen applies: a sampled k = -1 recall below 0.10
makes the arm inadmissible and it will be reported as inadmissible rather than dropped. If the
baseline moves a lot while the onset ratio does not, that is evidence *for* the law, since the law
does not mention the risky model; if both move together, the arm cannot separate them and will be
reported as confounded.
