# A third protected corpus: does the onset split survive a second replication?

**Design, grid, entry gate and bands committed 2026-09-15 14:27, while the three memorisers were
still in epoch 1 of 40 and before any `s_s`, `s_r` or prediction for this corpus existed.** Nothing
above `## Scoring log` is edited after that line is written. The parameter-free P1 predictions are
appended in a second commit, also before any sweep decodes a token; git records both orderings.

## Why a third corpus and not a fourth anchor

`results/onset_prediction_gutenberg.md` closed by saying that running more anchors after its bands
had failed to fail would be choosing when to stop after seeing the answer. That still holds, and
this arm does not do it. It holds the three anchors fixed --- the same three that already carry a
CopyBench reading and a Gutenberg reading --- and changes only the protected work, for the second
time. The result is three readings of the same three pairs, which buys one quantity that two
readings cannot: **the corpus-to-corpus spread of a pair's onset ratio, measured against the
sampling spread within one corpus.** "Reproduces within 0.05" is a statement about two points. A
third point turns it into a comparison with a yardstick that was published first.

The choice of corpus is not free either. BookMIA is where the paper's vacuity statement already
lives --- 9,870 passages across 100 books, against CopyBench's 758 across 16 --- so the onset
section and the scale section would, for the first time, be measured on the same protected text. It
is also a third draw from a third population: sixteen English genre novels, then 50 public-domain
books of the nineteenth century, then 31 books of contemporary popular fiction.

**What is NOT being used, stated because a reader will assume it.** BookMIA ships a `seen`/`unseen`
label and this arm ignores it completely. AGENTS.md records that the two halves are confounded under
an anchor that saw neither and may never be treated as a matched pair. Every pair here is
**self-paired** --- a clean anchor against a LoRA copy of *itself*, fine-tuned on exactly these
passages --- so the label plays no role whatsoever, and no claim below depends on whether a book was
in anyone's pre-training.

## The three pairs

Anchors `output/phase5/anchor_kl3m-002-520m`, `PleIAs/Pleias-1.2b-Preview` and
`output/phase5/anchor_phi35mini`; memorisers `output/phase5/memb_*`, fine-tuned with the flags
recovered verbatim from the Gutenberg run log so the two corpora differ in nothing but the text:

```
--target-modules all-linear --no-chat --epochs 40 --lr 3e-4 --rank 128
--batch 2 --accum 4 --max-len 0 --stop-loss 0.02
```

The corpus is `data/bench/bookmia100_onset600.jsonl`, 600 passages **stratified round-robin over all
31 books** of the BookMIA `attack_train` split, 19--20 per book. The sweep runs on
`bookmia100_onset100.jsonl`, 100 passages over the same 31 books and a **prefix** of the 600, so
every swept passage is one the memoriser saw --- the arrangement AGENTS.md caution (h) exists to
enforce, and the same one the Gutenberg arm used.

Stratification is not decoration. `bookmia100_attack_train.jsonl` is grouped by book, so a plain
`--limit 100` takes the first 100 rows and they are all *1984*: a hundred passages of one novel
reported as a hundred-passage corpus. That is caution (w) in a new costume and
`analysis/build_bookmia_onset_subset.py` exists to make it unreachable.

## Committed bands

Single-query near-verbatim recall at `--thresh 0.01`, `analysis/onset.py`'s bracket-and-interpolate
rule, `--limit 100`. Every threshold, gate and band below is **identical to the Gutenberg arm's**,
which is the only way the three readings are comparable; bands 1 and 2 are that file's, unchanged.

**Band 1 --- the level of Eq. (eq:req).**

| outcome | reading |
|---|---|
| all three pred/meas in **[0.85, 1.15]** | the level agreement is a property of the geometry, now at two corpora out of sample |
| all three in **[0.7, 1.4]**, at least one outside [0.85, 1.15] | order of magnitude only; report the range over three corpora and stop quoting a two-decimal agreement |
| **any pair outside [0.7, 1.4]**, or no crossing on the grid | the near-determinism approximation does not survive a second change of corpus. Said in the onset section, not only in Limitations |

A pass reinstates nothing. The equation is already refuted as a *predictor* (rank correlation
$-0.18$ over seven CopyBench pairs where the derivation requires it positive), three pairs cannot
revive it, and a correct ranking is reported with its exact $p$ --- floor $1/6$ at $n=3$ --- and
given no weight, exactly as on Gutenberg.

**Band 2 --- the onset section's central split.** CopyBench and Gutenberg both put KL3M-520M, the
fine-tokenizer pair, above $1$ with a bootstrap interval excluding $1$, and both coarse pairs below.

| outcome | reading |
|---|---|
| KL3M-520M **above 1** with its 95% interval **excluding 1**, both coarse pairs **below 1** | the split is a property of the pair at three corpora. Limitations stops hedging it |
| the ordering holds (KL3M highest) but its lower bound crosses $1$ | the ordering is the property and the level is not; "leakage begins after vacuity" is qualified wherever it appears |
| the ordering **inverts**, or KL3M-520M lands inside the coarse family's range | two agreeing corpora were a coincidence of two. The paper leads with it |

**Band 3 --- NEW, and the reason this arm is worth its GPU-hours.** With three readings each pair
has a corpus-to-corpus **range** (max $-$ min of its ratio). The claim is that this range is smaller
than the sampling noise *within a single corpus*, i.e. that changing the protected work perturbs the
onset ratio less than re-drawing 100 passages of one work does. The yardstick is each pair's
**CopyBench** bootstrap width, published in `results/onset_ci.csv` long before this arm:

```
pair            CopyBench ratio  95% CI            width    Gutenberg ratio   2-corpus range so far
KL3M-520M          1.0532        [1.0163, 1.2436]  0.2273       1.1020               0.0488
Pleias-1.2B        0.8784        [0.7891, 0.9600]  0.1709       0.8947               0.0163
Phi-3.5-mini       0.9261        [0.7986, 1.0852]  0.2866       0.9489               0.0228
```

| outcome | reading |
|---|---|
| every pair's three-corpus range is **below its CopyBench CI width** | the corpus is not a material source of variation in the onset ratio; within-corpus sampling dominates it, and the paper may say the ratio is a property of the pair without a corpus caveat |
| one pair exceeds its width | report the range per pair; the ratio is a property of the pair for the other two and corpus-sensitive for that one, named |
| two or more exceed | the corpus is a first-order term. Every onset ratio in the paper is quoted with the corpus it was measured on |

**Entry gate, unchanged.** A pair enters only if its **sampled** $k=-1$ recall is at least $0.10$
(caution (a); greedy recall lies). `k=-1` and `k=0` run on the same passages and seeds as every
budget. A pair that fails the gate is excluded from all three bands and reported as excluded --- it
is a statement about the memoriser, not about the law.

**The grid, fixed now, before any `s_s` for this corpus exists:** the Gutenberg arm's, unchanged,
`-1 0 1.2 1.6 1.9 2.1 2.3 2.5 2.7 2.9 3.2 3.6 4.2`. It is the identical grid already committed for
another corpus, which is the strongest possible guarantee that it was not shaped to bracket this
one. A crossing at either end is a **finding about the grid** and is reported as one, with the
bootstrap no-crossing fraction (caution (g)). The single extension permitted is the one already on
record in `sections/appendix_seed.tex` --- extend when a pair's no-crossing fraction rises
materially above the others', and report **both** grids --- and it is cited here so that using it
cannot be a decision made after seeing an answer.

**What will not happen.** No second grid beyond that published rule, no re-threshold, no re-seed, no
fourth anchor, no fourth corpus. The bands above are the whole of the scoring rule.

## The P1 predictions, appended 2026-09-15 17:41, before any sweep decoded a token

Computed by `analysis/onset_theory.py` from two teacher-forced forward passes per passage and no
decoding at all, and committed in the same commit as `results/onset_theory_bookmia.csv`. The design,
the grid, the entry gate and all three bands above were committed at 14:27, three and a quarter
hours earlier, when none of these numbers existed.

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/onset_theory.py --corpus-file data/bench/bookmia100_onset100.jsonl \
    --pairs-file results/onset_theory_pairs_bookmia.tsv --limit 100 --tag _bookmia --out results
```

```
pair                          s_s     s_r    pred onset   pred ratio   r(x) q10   r(x) q25
KL3M-520M (BookMIA)         2.4316  0.2345     2.1971       0.9036      1.8651     2.0006
Pleias-1.2B (BookMIA)       3.0481  0.4346     2.6135       0.8574      1.9466     2.2418
Phi-3.5-mini (BookMIA)      2.6283  0.0094     2.6189       0.9964      1.9558     2.2852
```

Their Gutenberg twins, for the record, since the same three anchors carry them:

```
pair                          s_s     s_r    pred onset   pred ratio   r(x) q10   r(x) q25
KL3M-520M (Gutenberg)       2.3665  0.1487     2.2180       0.9372      1.8680     2.0230
Pleias-1.2B (Gutenberg)     2.8087  0.3453     2.4630       0.8771      1.9570     2.2040
Phi-3.5-mini (Gutenberg)    2.8488  0.0362     2.8130       0.9873      2.1700     2.3780
```

**The grid brackets every prediction, and it was fixed before any of them existed.** The committed
`-1 0 1.2 1.6 1.9 2.1 2.3 2.5 2.7 2.9 3.2 3.6 4.2` puts its lowest budgeted point below every `q10`
(1.2 against 1.865, 1.947, 1.956) and its top well above every median (4.2 against 2.172, 2.569,
2.572), with each predicted onset strictly inside. That it brackets is checked, not asserted, and
it is a property the grid was not given the chance to acquire: it is the Gutenberg arm's grid,
copied verbatim at 14:27.

**All three pairs pass the entry gate, on sampled recall and not greedy** (caution (a)):

```
pair            greedy   SAMPLED k=-1   verdict       Gutenberg twin (sampled)
KL3M-520M       0.947       0.918       ADMISSIBLE          0.578
Pleias-1.2B     0.809       0.457       ADMISSIBLE          0.517
Phi-3.5-mini    0.883       0.881       ADMISSIBLE          0.270
```

Every BookMIA memoriser is at least as strong as its Gutenberg twin and two are far stronger, which
is worth stating because memoriser strength is what `s_r` measures and therefore what moves the
prediction. It matters most for Phi-3.5-mini: on Gutenberg it was the marginal pair --- the weakest
memoriser at 0.270, the widest bootstrap interval, the only nonzero no-crossing fraction and a
non-monotone tail --- and at 0.881 it is no longer marginal. Its `s_r` is 0.0094, essentially no
residual surprisal on its own memorised text, which is why P1 puts its predicted ratio at 0.9964.

Nothing above this line is edited from here on, and no band, grid, gate or threshold above was
touched when this block was added --- only these measurements, which are predictions, were appended.

## Scoring log

### Preliminary readings, and why NEITHER band is scored yet, 2026-09-15 21:50

All three sweeps ran on the committed grid. Every pair passes the entry gate on its own **sampled**
`k=-1` arm, every `k=0` arm reads `0.000` (the anchor alone leaks nothing on BookMIA either), and
there are **zero per-trajectory violations** across all 33 budgeted cells.

```
pair            k=-1    k=0    onset   bracket     ratio  95% CI           no-x    pred   pred/meas
KL3M-520M      0.7326  0.000   2.465  (2.3, 2.5]  1.0138  [0.970, 1.131]   0.0%   2.197    0.891
Pleias-1.2B    0.1504  0.000   4.006  (3.6, 4.2]  1.3142  [1.004, 1.360]  43.1%   2.614    0.652
Phi-3.5-mini   0.4425  0.000   2.607  (2.5, 2.7]  0.9920  [0.867, 1.370]   0.0%   2.619    1.004
```

Read naively this fires band 1's third row (Pleias at `0.652`, outside `[0.7, 1.4]`) and band 2's
third row (the ordering **inverts** --- coarse-tokenizer Pleias at `1.314` above fine-tokenizer
KL3M at `1.014`). Those are the two strongest negative outcomes the file commits to, and the paper
would have to lead with them.

**They are not scored, because both of them rest entirely on the one pair the grid-ceiling rule
disqualifies.** Pleias-1.2B's bootstrap no-crossing fraction is **43.1%** against `0.0%` and `0.0%`
for the other two, and its onset lands in the top bracket `(3.6, 4.2]` with an upper bound of
`4.144`, `0.056` below the ceiling. That is caution (g)'s signature exactly: an interval that is
narrow *because it is conditioned on the resamples that happened to cross*. Its curve says the same
thing --- it never leaves the noise floor until the last point:

```
Pleias-1.2B  1.2:0.000  1.6:0.000  1.9:0.000  2.1:0.000  2.3:0.000  2.5:0.001
             2.7:0.001  2.9:0.000  3.2:0.006  3.6:0.006  4.2:0.012
```

It grazes the `0.01` threshold at `k = 4.2` and nowhere else, so "onset `4.006`" is an interpolation
into the last interval of the grid, not a measurement of where leakage begins.

**The extension is licensed by a rule committed before this arm existed** --- `appendix_seed.tex`'s
*"extend whenever the no-crossing fraction rises materially above the others, and report both
grids"*, cited in this file at 14:27 as the single permitted extension. `43.1%` against `0.0%` and
`0.0%` is that condition. **Committed now, before the extension runs:**

- **Pleias-1.2B only.** The other two are at `0.0%` and are not touched; their readings above stand
  as final and are not re-run at any grid.
- **Grid `k in {4.6, 5.3, 6.6}`**, obtained mechanically as the ceiling `4.2` times the same
  multipliers the KL3M-1.7B precedent used on its own ceiling (`3.2 -> 3.5, 4.0, 5.0`, i.e.
  `x1.094, x1.25, x1.5625`). Same 100 passages, same seed, same threshold, same `--modes single`.
  Run into `fineb_pleias12b_ext`, merged into `fineb_pleias12b_full`, **both grids reported**.
- **What it can change:** the no-crossing fraction, the interval, and whether the point estimate
  survives. On the one precedent, no-crossing went `4.3% -> 0.0%`, the upper end widened and the
  onset was **unmoved**; nothing here assumes that repeats.
- **What it cannot do:** it cannot rescue band 1 or band 2 by fiat. If Pleias' onset holds near
  `4.0` with the no-crossing at zero, then band 1 fails and band 2 inverts, on a clean grid, and
  they are reported as failing --- which is what the bands were written for.
- **What will not happen:** no third grid, no re-threshold, no re-seed, no raising the entry gate
  after the fact. Pleias' `k=-1` of `0.1504` clears the committed `0.10` and it enters; that it is
  the weakest of the three is reported as a caveat, never used to exclude it.

**A discrepancy recorded rather than resolved.** The fine-tuner's own post-training check put
Pleias' sampled recall at `0.457` on 24 *training* excerpts; the sweep's `k=-1` arm on the 100
swept passages reads `0.1504`. Different sample and different draw, so they are not the same
quantity, and the gate is the sweep's own `k=-1` (caution (a)). The gap is noted because a weak
memoriser needs more budget before it can leak, which is the obvious alternative explanation for
this pair sitting at the top of the grid, and it is not one the extension can settle.

### Scoring, 2026-09-15 22:05: the extension confirms rather than rescues, and TWO of three bands fail

```
.venv/bin/python analysis/composition_attack.py --safe-model PleIAs/Pleias-1.2b-Preview \
  --risky-model output/phase5/memb_Pleias-1_2b --corpus-file data/bench/bookmia100_onset100.jsonl \
  --k-values 4.6 5.3 6.6 --modes single --limit 100 --out output/phase5/fineb_pleias12b_ext
# merged into output/phase5/fineb_pleias12b_full; the scorer repointed, BOTH grids reported
.venv/bin/python analysis/onset_gutenberg.py --corpus bookmia --out results
.venv/bin/python analysis/onset_ci.py --comp output/phase5/fineb_pleias12b_full/composition.csv \
  --s-x 3.048079572669047 --label "Pleias-1.2B (BookMIA, extended grid)" --out results
```

**The extension behaved exactly as the one precedent said it would, and that settles the pair
against us.** Recall keeps climbing above the old ceiling --- `4.6:0.026`, `5.3:0.031`, `6.6:0.040`
--- so the bootstrap resamples have somewhere to cross and the conditioning artefact dissolves:

```
grid                 onset    ratio    95% CI            no-crossing
committed (top 4.2)  4.0058   1.3142   [1.004, 1.360]      43.1%
extended  (top 6.6)  4.0058   1.3142   [1.010, 1.594]       0.4%
```

The point estimate is **unmoved to four decimals**, the upper end widens, the no-crossing collapses
to `0.4%`. The reading was never a ceiling artefact; the grid was simply too short to prove it. Zero
per-trajectory violations in all 300 new queries. `results/onset_bookmia_committed_grid.csv` holds
the unextended scoring, as the rule requires.

### Band 1 (the level of Eq. (eq:req)): **FAILS**

```
pair            pred    onset    pred/meas     Gutenberg pred/meas
KL3M-520M      2.197    2.465      0.891              0.851
Pleias-1.2B    2.614    4.006      0.652              0.980
Phi-3.5-mini   2.619    2.607      1.004              1.040
```

`0.652` is outside the committed `[0.7, 1.4]`, so the third row fires: **the near-determinism
approximation does not survive a second change of corpus**, and the onset section says so rather
than Limitations alone. Two of three pairs are fine --- `0.891` and `1.004`, both inside the tight
`[0.85, 1.15]` --- which is the point: the equation puts the onset in the right place until it does
not, and nothing in it says which case you are in. **P2 fails on all three** again, as on Gutenberg.
The direction test gives `rho = -1.00` at exact `p = 0.333`: no information at `n = 3`, and the sign
is the wrong one, consistent with the seven-pair CopyBench inversion at `-0.18` that already
refuted it.

### Band 2 (the onset section's central split): **INVERTS** --- the strongest committed negative

```
pair            ratio   95% CI            tokenizer   CopyBench  Gutenberg
KL3M-520M      1.0138  [0.970, 1.131]      fine         1.053      1.102
Pleias-1.2B    1.3142  [1.010, 1.594]      coarse       0.878      0.895
Phi-3.5-mini   0.9920  [0.867, 1.370]      coarse       0.926      0.949
```

The third row fires, and it fires **twice over**:

1. **The ordering inverts.** The coarse-tokenizer Pleias pair is now the highest of the three at
   `1.3142`, above the fine-tokenizer KL3M pair at `1.0138`. On CopyBench and on Gutenberg the fine
   pair was the highest and the two coarse pairs sat below `1`.
2. **KL3M's interval no longer excludes 1.** It reads `[0.970, 1.131]` against CopyBench's
   `[1.0163, 1.2436]` and Gutenberg's `[1.0744, 1.4227]`. That interval excluding `1` is the whole
   evidential basis for "leakage begins *after* the certificate has gone vacuous", and on the third
   corpus it does not.

The committed reading is *"two agreeing corpora were a coincidence of two. The paper leads with
it."* That is what the file says and it is what the paper will do.

### Band 3 (corpus-to-corpus range against within-corpus sampling): **ONE PAIR EXCEEDS**

```
pair           CopyBench  Gutenberg   BookMIA    range   CopyBench CI width   verdict
KL3M-520M         1.0532     1.1020    1.0138   0.0882         0.2273         inside
Pleias-1.2B       0.8784     0.8947    1.3142   0.4358         0.1709         EXCEEDS
Phi-3.5-mini      0.9261     0.9489    0.9920   0.0659         0.2866         inside
```

The second row fires: the ratio is a property of the pair for KL3M-520M and Phi-3.5-mini --- whose
three-corpus ranges are `2.6x` and `4.3x` *inside* their own within-corpus sampling widths --- and
**corpus-sensitive for Pleias-1.2B, named**. Every onset ratio quoted for that pair carries the
corpus it was measured on.

### The confound, measured rather than asserted, and it does NOT rescue anything

The pre-registration flagged in advance that a weak memoriser needs more budget before it can leak.
Pleias' memoriser is by far the weakest here, and its strength falls monotonically across the three
corpora while its ratio rises monotonically:

```
pair            sampled k=-1: CopyBench  Gutenberg  BookMIA      onset ratio across the same three
KL3M-520M                        0.5188     0.5784   0.7326      1.0532  1.1020  1.0138
Pleias-1.2B                      0.9091     0.5168   0.1504      0.8784  0.8947  1.3142
Phi-3.5-mini                     0.5660     0.2696   0.4425      0.9261  0.9489  0.9920
```

Within Pleias the rank correlation is `-1.00`, but `n = 3` has a floor of `1/3` and it carries no
information. **Across all nine cells it is `rho = -0.317` at exact `p = 0.4101`** --- POST HOC, no
band, and not significant. So memoriser strength is a real caveat for this one pair and **not** a
general explanation for the ratio, and it is reported as a caveat and never used to set aside a band
that fired. Pleias' `k=-1` of `0.1504` clears the committed `0.10` gate; the gate is not raised after
the fact.

### What this arm establishes, and what it does not

It establishes that the onset section's central split --- fine-tokenizer pairs above `1`, coarse
below --- **is not a property of the pair**. It held on two corpora and fails on the third, on the
same three anchors, the same architectures, the same settings, the same seed and the same grid, with
only the protected work changed. The paper can no longer say the split survives a change of corpus.

It does not establish *why*. Three pairs cannot separate corpus from memoriser strength when the
memoriser is trained on the corpus, and this design confounds them by construction. That is a
limitation of the design, stated here and in the paper, not a reason to discount the failure: the
bands were written to be failed, two of them failed, and they are reported as failed.

