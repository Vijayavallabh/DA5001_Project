# Pre-registration: does the vacuity statement survive a 13x larger corpus?

Committed **before the measurement runs**. Nothing above the `## Scoring log` line is edited
afterwards.

## Why this arm exists

Proposition 1's empirical content in the paper is one number: at the authors' own budget `k = 3`
the certificate is vacuous for **100% of the 758 protected passages**, and Limitations says plainly
that those passages are sixteen English genre novels and that `p = 3.05e-5` is simply the smallest
value sixteen paired items can produce. A reviewer is entitled to ask whether the threshold
statement is a property of divergence budgets or a property of our sixteen books.

It is answerable with **forward passes of the anchor alone** --- no decoding, no attack, no
memoriser, no fine-tune --- because `s(x)` and the vacuity verdict `K >= S(x)` need nothing else.
`data/bench/bookmia100/` holds the 50 BookMIA books marked *seen* (4,935 passages, 31/8/11 books
split by book) and `data/bench/bookmia100unseen/` the 50 marked *unseen* (4,935 passages). Together
that is **9,870 passages across 100 books against 758 across 16**, at a cost of minutes.

The unseen half is a control the paper has never had. The anchor saw neither half --- "seen" and
"unseen" are labels about the *target* models BookMIA was built to audit, not about our safe model
--- so under the anchor the two halves should be indistinguishable in `s(x)`. If they are not, the
label is confounded with text properties, which is worth knowing before anyone builds on the corpus.

## What is measured

`analysis/regimes.py --model jacquelinehe/tinycomma-1.8b-llama3-tokenizer --data <corpus>`, which
computes per passage the surprisal rate `s(x) = S(x)/chars(x)` in nats per **character**, the
Loynes running maximum `k_crit(x)`, and the order-invariant vacuity threshold. Rates are converted
to nats per token with the anchor's own characters-per-token where a rate has to meet a budget,
which is the convention Appendix B fixes and the only place the two denominators may be mixed.

## Bands, committed before the run

**V1 -- does the vacuity verdict replicate?** Fraction of BookMIA-seen passages for which the
certificate is vacuous at the authors' `k = 3`, against `1.000` on the 758.

| reading | band |
|---|---|
| REPLICATES | `>= 0.99` |
| WEAKER | `0.90` to `0.99` |
| DIFFERENT CORPUS | `< 0.90`, and the paper must say the 100% is a property of its sixteen novels |

**V2 -- is it the same regime?** Median `s(x)` on BookMIA-seen against the median on the sixteen
novels.

| reading | band |
|---|---|
| SAME REGIME | the two medians are within `15\%` of each other |
| SHIFTED | they are not, and the shift is reported with the vacuity fraction beside it |

**V3 -- is the seen/unseen label confounded?** Median `s(x)` on BookMIA-seen against BookMIA-unseen,
under an anchor that saw neither.

| reading | band |
|---|---|
| INDISTINGUISHABLE | the medians are within `5\%` |
| CONFOUNDED | they are not, and no arm on this corpus may treat the halves as exchangeable |

**V4 -- descriptive, no band.** The spread of `s(x)` across the 100 books, and the fraction of
passages whose `k_crit` exceeds `s(x)` by more than a factor of two. Reported for the appendix; 100
books is enough to describe a distribution and not enough to fit one.

## Excluded alternatives (named now so they cannot be adopted afterwards)

1. Dropping books, passages or a split after seeing any number. The corpus is what
   `analysis/build_bench_corpora.py` built and is fixed.
2. Changing the anchor, the dtype or the budget `k = 3` after seeing V1.
3. Reporting the seen half without the unseen half.
4. Quoting a per-character rate against a per-token budget without the conversion (the confusion
   AGENTS records as finding (4)).
5. Treating V1 as evidence about extraction *onset*. It is a statement about when the certificate
   stops saying anything, which is Proposition 1 and not Section 3; the onset arm needs
   memorisers this corpus has none of.

## Scoring

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<free card> HF_HUB_OFFLINE=1 \
  HF_HUB_CACHE=$PWD/hf_cache .venv/bin/python analysis/regimes.py \
    --model jacquelinehe/tinycomma-1.8b-llama3-tokenizer \
    --data data/bench/bookmia100 --out results/regimes_bookmia100.csv
#  ... and again with data/bench/bookmia100unseen -> results/regimes_bookmia100unseen.csv
.venv/bin/python analysis/bookmia_regimes.py --out results
```

Writes `results/bookmia_regimes.csv`.

---

## Scoring log (appended after the run; nothing above this line is edited)

---

## Scoring, 2026-09-12 (appended; nothing above is edited)

```
analysis/regimes.py --model jacquelinehe/tinycomma-1.8b-llama3-tokenizer \
  --data data/bench/bookmia100      --out results/regimes_bookmia100.csv
  ... and bookmia100unseen, and --data data (the 758, as a positive control)
.venv/bin/python analysis/bookmia_regimes.py --out results
```
→ `results/bookmia_regimes.csv`. Anchor forward passes only; about eight GPU-minutes for all three.

### A correction, made before any band was read

The first implementation of this script scored vacuity as `k >= s(x)` per token. That is
Proposition 1's **asymptotic** criterion --- `K = kT` and `S(x) = s(x)T` are both rays, so for a
work long enough the ratio decides --- and it is what Figure 1(a) draws. It is **not** what the
paper's `100\%` means. A deployment publishes `K = k*T_max` for the sequence it is configured to
emit, and `analysis/certificate_cap.py` scores `S(x) <= K` at `T_max = 200`; the protected span is
about 60 tokens, well inside the cap.

The CopyBench arm exists as a positive control for exactly this, and it caught it: under the wrong
criterion it read `0.326` where the paper's own `certificate_cap_summary.csv` says `1.000`, so the
criterion was wrong and not the corpus. Under `S(x) <= k*T_max` it reads `1.0000` and reproduces
the paper exactly. Every band below is scored on that criterion, and the asymptotic one is reported
beside it, labelled, never as V1.

| arm | passages | books | vacuous at `K = 600` | median `S(x)` | median `s(x)`/token | `s(x) <= 3` |
|---|---|---|---|---|---|---|
| CopyBench, 16 novels (control) | 758 | 16 | `1.0000` | `204.8` | `3.197` | `0.3259` |
| BookMIA seen, 50 books | 4,935 | 50 | `0.9868` | `182.6` | `3.114` | `0.4028` |
| BookMIA unseen, 50 books | 4,935 | 50 | `1.0000` | `192.1` | `3.313` | `0.2588` |

### V1 — **WEAKER**, at `0.9868`, and the shortfall is 65 passages

The band called `>= 0.99` REPLICATES and this is `0.9868`, so the reading is WEAKER and is reported
as WEAKER. What that means concretely: of `4{,}935` BookMIA-seen passages, `65` are surprising
enough to the anchor that a `600`-nat budget does not cover them. On the unseen half and on the
paper's own `758` the fraction is `1.0000`.

So the statement survives a **13x larger corpus and 6x more books** with a `1.3\%` shortfall on one
half. The paper's sentence should say `9{,}870` passages across `100` books rather than `758`
across `16`, and should say `98.7\%` where the corpus is BookMIA-seen rather than carrying `100\%`
everywhere.

### V2 — **SAME REGIME**

Median `s(x)` per token is `3.114` on BookMIA-seen against `3.197` on the sixteen novels, `2.6\%`
apart, well inside the committed `15\%`. Fifty books of a different provenance put the anchor in
the same surprisal regime as our sixteen, which is the part of the corpus objection that was
genuinely open.

### V3 — **CONFOUNDED**, and this is the arm's most useful output

Under an anchor that saw **neither** half, BookMIA-seen has median `s(x)` `3.114` and BookMIA-unseen
`3.313`, `6.0\%` apart against a committed `5\%`. The label is about the *target* models BookMIA was
built to audit, so it should have been irrelevant to our safe model; it is not. The plain reading is
that the books marked seen are the more widely reproduced ones and their prose is more predictable
to any language model, so `seen` is entangled with a text property.

**No arm on this corpus may treat the halves as exchangeable**, and in particular the unseen half is
not a clean negative control for anything measured in nats. It is still a second corpus, and V1 and
V2 use it as one.

### V4 — descriptive

`k_crit/s(x)` has median `x5.66` on BookMIA-seen, `x5.27` unseen, `x5.04` on the sixteen novels, and
the fraction of passages more than a factor of two above is `98.4\%`, `97.8\%` and `90.4\%`. The
interval on which a mechanism protects without certifying is wide on every corpus and slightly wider
on the larger ones.

### The asymptotic criterion, reported and not scored

`s(x) <= k` holds for `0.4028`, `0.2588` and `0.3259` of passages. The gap between that and the
`0.99`-and-up under `K = k*T_max` is the cap: `T_max = 200` against a protected span near `60`
tokens, so the published budget covers more than three times the work being targeted. The two
criteria agree in the limit Figure 1(a) draws, where the decoder's cap and the work are the same
length, and the deployed number is the more favourable of the two to vacuity. Neither is wrong;
they answer different questions and the paper should keep using the deployed one.

### Status

V1's criterion was corrected before any band was read, and the correction is disclosed above rather
than folded in. V2, V3 and V4 are as registered.
