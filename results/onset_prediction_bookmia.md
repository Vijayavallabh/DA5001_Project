# A third protected corpus: does the onset split survive a second replication?

**Design, grid, entry gate and bands committed 2026-09-15 14:27, while the three memorisers were
still in epoch 1 of 40 and before any `s_s`, `s_r` or prediction for this corpus existed.** Nothing
above `## The P1 predictions, appended 2026-09-15 17:41, before any sweep decoded a token

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

## Scoring log` is edited after that line is written. The parameter-free P1 predictions are
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

## Scoring log
