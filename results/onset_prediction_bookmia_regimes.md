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
