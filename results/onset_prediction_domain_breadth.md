# Pre-registration: is the constructive claim anchor-bound?

Committed **before any per-domain number is computed**. Nothing above the `## Scoring log` line is
edited afterwards.

## What is already known, stated first so nothing below can be passed off as blind

The two standard-benchmark arms are already scored in aggregate and are on record:

| benchmark | n | judge B gain at `n=8` | judge C gain at `n=8` |
|---|---|---|---|
| in-house ordinary prompts | 500 | `+0.054 [+0.013, +0.095]` | — |
| AlpacaEval | 805 | `+0.031 [-0.001, +0.062]` | `+0.067 [+0.034, +0.101]` |
| MT-Bench | 80 | `+0.006 [-0.088, +0.094]` | `-0.038 [-0.131, +0.050]` |

So the constructive claim is **weaker off the in-house prompt set**, one judge does not resolve it on
AlpacaEval, and MT-Bench resolves nothing in either judge. That much is settled and is not under
review here. What is **not** computed, and is what this file registers, is any split of either
benchmark by domain.

## Why the split is worth running

Proposition~4 says `q(y) <= n p_s(y)`: a served string must be one the anchor would have drawn. So
selection cannot manufacture ability the anchor lacks — it can only reorder what the anchor already
produces. The paper's Limitations asserts this ("selection anchoring cannot exceed its anchor's
support") and has never measured it. If it is the right reading of the weakening above, then the
gain should be **larger where the anchor can already do the task and absent where it cannot**, and
the aggregate AlpacaEval and MT-Bench numbers are averages over domains that differ in exactly that.

If instead the gain is flat across domains of very different anchor competence, the ceiling is not
what is limiting the benchmark arms, the Limitations sentence is the wrong diagnosis, and the paper
must say the weakening is unexplained.

## The cells, fixed now

**AlpacaEval, its five constituent sources**, as shipped in the benchmark and read off the corpus
field, not chosen by us: `selfinstruct` (252), `oasst` (188), `koala` (156), `helpful_base` (129),
`vicuna` (80).

**MT-Bench, two families of four categories each**, grouped here **before** the eight cells are
computed, because ten prompts per category resolve nothing:

* **open-ended** — `writing`, `roleplay`, `humanities`, `stem` (40)
* **constrained** — `reasoning`, `math`, `coding`, `extraction` (40)

Seven cells in total. Both judges, `n = 8`, from the per-prompt CSVs already written
(`results/selection_scaling_per_prompt_{alpaca,mtbench}.csv`); no generation, no GPU, no re-judging.

## Bands, committed before the numbers

**D1 -- is the gain anchor-bound?** Spearman across the seven cells of the cell's `n=8` gain against
the cell's anchor-alone control level `u(n=1)`, with an exact permutation `p` over all `7! = 5040`
orderings. Computed **separately for each judge, and the reading requires both**, so that no judge
can be picked after the fact.

| reading | band |
|---|---|
| ANCHOR-BOUND | `rho >= +0.5` in **both** judges |
| UNRELATED | otherwise, with `\|rho\| < 0.5` in at least one |
| ANTI | `rho <= -0.5` in both |

The shared-noise term biases this **negative**: `u(n=1)` is subtracted inside the gain, so a cell
whose control happens to read high has a mechanically lower gain. A positive `rho` is therefore
conservative and a negative one is the cheap direction. Stated now so the sign cannot be
reinterpreted afterwards.

**D2 -- per-cell gains, descriptive.** Each cell's `n=8` gain with a paired bootstrap CI, both
judges. **Seven cells at `n = 40` to `252` cannot each resolve a gain of `0.05`** -- at `n=129` the
paired CI half-width is about `0.06` -- so no individual cell may be quoted as a positive or a null
result. This is written here rather than discovered later.

**D3 -- the MT-Bench family contrast.** Control level and gain for open-ended against constrained.
The ceiling reading predicts the **control level** is higher on open-ended; that comparison is far
better powered than the gain, since a level at `n=40` has a standard error near `0.08` while the
between-family difference the reading predicts is large.

| reading | band |
|---|---|
| CEILING VISIBLE | open-ended control exceeds constrained by more than `0.10` |
| NO FAMILY EFFECT | the two are within `0.10` |

## Excluded alternatives (named now so they cannot be adopted afterwards)

1. Regrouping the eight MT-Bench categories after seeing them. The two families above are final.
2. Dropping an AlpacaEval source, or merging two, after seeing its number.
3. Binning **per prompt** on `u(n=1)`. That is mechanically confounded --- `u(n=1)` is the quantity
   subtracted --- and would produce a strong spurious correlation. It is named here so it cannot be
   adopted as a "better powered" version of D1.
4. Changing `n`, the judge, or the reward model after seeing any cell.
5. Quoting a single cell as evidence for or against the gain (D2 forbids it).
6. Re-describing the aggregate AlpacaEval and MT-Bench readings above as anything other than what
   they are: weaker than the in-house set, one judge unresolved, and one benchmark underpowered.

## Scoring

```
.venv/bin/python analysis/domain_breadth.py --out results
```

Writes `results/domain_breadth.csv`.

---

## Scoring log (appended after the run; nothing above this line is edited)
