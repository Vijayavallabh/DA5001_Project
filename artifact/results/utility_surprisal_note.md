# Post-hoc note: where selection's utility lives, against where its certificate says something

**Post hoc, 2026-09-25, descriptive, no band.** Review 2 (Q6): "stratify the GSM8K majority-vote gain
(0.320 -> 0.546) and the judged gains by answer surprisal. If the gains concentrate where log n >= S(x), the
certificate and the utility are decoupled in practice." Proposition 1 bounds `q(E) <= n p_s(E)` for every
event `E`. That bound excludes something about `E` only when `S(E) = -log p_s(E) > log n`.

Command: `.venv/bin/python analysis/utility_surprisal.py` -> `results/utility_surprisal.csv`. The script
asserts the committed accuracies (`0.320 -> 0.546` GSM8K, `0.280 -> 0.328` TriviaQA, both at `n = 32`,
Comma-7B).

**Verifiable tasks.** The event is "the extracted answer is correct". The vote at `n = 32` reads draws
`0`-`31` of the committed `64`. The anchor's probability of the event, `pi_ho`, is estimated on the held-out
draws `32`-`63`, so the stratifier never sees a draw the vote used.

| task | stratum (held-out) | questions | acc, `n=1` | acc, vote | gain [95%] | share of gain |
|---|---|---|---|---|---|---|
| GSM8K | `pi_ho = 0` (`S > log 32`) | 61 | 0.033 | 0.033 | +0.000 [-0.066, +0.066] | 0.0% |
| GSM8K | `pi_ho >= 1/32` (`S <= log 32`) | 439 | 0.360 | 0.617 | +0.257 [+0.212, +0.303] | 100% |
| TriviaQA | `pi_ho = 0` | 213 | 0.005 | 0.005 | +0.000 [-0.014, +0.014] | 0.0% |
| TriviaQA | `pi_ho >= 1/32` | 287 | 0.484 | 0.568 | +0.084 [+0.038, +0.129] | 100% |

The CSV also splits `pi_ho >= 1/32` at `1/4` and `1/2`. GSM8K's gain is spread across all three bands:
`+0.168`, `+0.429` and `+0.258`. TriviaQA's comes from the upper two, and in `[1/32, 1/4)` it is `-0.016`.

So yes: on both tasks **all** of the vote's gain is on answers the anchor already gives at least about
`1/32`, where `log n` excludes nothing about the answer. This holds by construction for any rule that serves
one of `n` anchor draws. Such a rule can raise `q(E)` only up to `n p_s(E)`, so it cannot make a correct answer
likely unless the anchor already gives it about `1/n`. The certificate and the utility sit on different events,
and that is the design. Copyright protection needs a small amplification of rare events: a protected
`50`-token window has median anchor surprisal `159.8` nats, far above `log 64 = 4.16`. Utility needs a
large amplification of common ones. On a verifiable task the useful event is common, so the certificate
is vacuous for it. That is also why selection cannot supply a fact the anchor lacks (TriviaQA: `213` of
`500` questions have no correct answer in `32` held-out draws, and the vote gains exactly nothing there).

**Judged workload.** No answer event is defined for an open-ended prompt, so the event is the served
completion itself. `S(y) = -sum log p_s` runs over its tokens through the first end-of-text and is read off
the pool's per-step log (`output/phase5/sel_anchor64`, temperature `1.0`). The per-prompt judged gain is Table
2's judge-B pass, `u_sel_n64 - u_sel_n1` (`results/matched_h2h_per_prompt_matched_plain_B.csv`); in the CSV's
two judged rows, `acc_n1` and `acc_vote` are these judged levels.

| stratum | prompts | gain [95%] | share of gain |
|---|---|---|---|
| served `S > log 64` | 460 | +0.110 [+0.085, +0.135] | 100% |
| served `S <= log 64` | 40 | +0.000 [-0.100, +0.106] | 0.0% |

The `40` low-surprisal served completions are all **empty**: the anchor's end-of-text at step `0`, with `S`
between `0.26` and `4.16` nats. The median served completion has `S = 184.6` nats, against `186.2` for the median rank-0 draw.
On open-ended text the judged gain comes entirely from completions the certificate is informative about.
Each served string is a rare event under the anchor, and `q <= 64 p_s` bounds its amplification. Selection
gains by choosing among many individually rare completions, not by amplifying one likely one.

**What this changes in the paper.** The decoupling on verifiable tasks is real and should be stated
rather than implied: a certificate of `log n` is informative about events with `S > log n` and says nothing
about common ones, including every correct answer a vote can reach. This is the margin argument of Section
2 (`s(x)/c_use`) read on a single event rather than a rate.
