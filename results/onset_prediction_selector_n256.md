# Pre-registration: the adversarial selector, corrected, and contaminated anchors at $n=256$

**feat-179.** Committed **2026-09-23**, before any arm below has run. The only GPU process started
before this commit is an instrument canary (`--n-values 1` on the audited anchor, written to
scratch, never to `results/`), which checks that the memoriser control of `## Gates` reproduces
through the corrected code; it produces no number that any band below reads. Nothing above
`## Scoring log` is edited after the first arm starts.

## Why

Two reasons, one asked for and one found.

**Asked for.** Review 4, Q9: *"Contaminated-anchor amplification 1.0--4.0: define the event, give
per-anchor rates, and test $n=256$."* The paper measures amplification at $n \le 64$ only, over
twelve LoRA-memorised anchors (`results/onset_prediction_contaminated_anchor{,_twelve}.md`).

**Found while writing this registration: the adversarial selector has scored padding since the
script was written.** `analysis/selection_extraction.py:score()` ranks the $n$ anchor draws by the
memoriser's mean per-token log-likelihood. `load_tok()` pads on the **left**, as batched generation
needs, and `score()` read the positions `[plen, mask.sum())` as if the batch were right-padded. For
every candidate shorter than the longest in its batch that window lands on the prompt's tail and on
the padding: a one-token continuation is scored as `<|end_of_text|>`. Reproduced on the
memoriser's own tokenizer before any fix; fixed at the one place both scoring calls route through;
pinned by `tests/test_selection_extraction_scoring.py`, which fails on the old indexing and passes on
the new. `analysis/selection_decoding.py` carries the same indexing but its tokenizer pads right,
which is correct, so it is unaffected.

**What the defect cannot have touched, checked before writing a band:** every clean-anchor
extraction arm on record has `anchor_max_recall = 0.0000` on every passage --- thirteen files,
including `selection_extraction_n256` --- so no draw in any of those pools reproduces anything, and
**no selector, correct or not, could have served a leak**. The paper's headline zero is
selector-free. **What it can have touched:** every number read on the served pick at $n > 1$ where
the pool holds a positive event --- the contaminated-anchor amplification ($1.0$--$4.0$ at $n=64$,
in the introduction, Section 3 and Appendix I) and the ROUGE-L counts at $n > 1$ on the clean
anchors (`tab:extraction`'s paraphrase row, Appendix I's multilingual paragraph). $n=1$ serves draw
$0$ whatever the scores, so every $n=1$ number is untouched.

## What is run

`analysis/selection_extraction.py` with the corrected `score()` and nine new per-passage columns
that make selector-free readings possible: `anchor_max_rouge`, `pool_ge_0p8` and `pool_ge_0p01`
(per-**draw** event counts over the whole pool) and, for each $n$, `oracle_recall_n{n}` and
`oracle_rouge_n{n}` (the best of the first $n$ draws, i.e. what *any* selector could serve).

**Part A --- the twelve contaminated anchors, $n \in \{1, 8, 64, 256\}$.** The committed protocol
unchanged in every other respect: `output/phase5/mem_*` anchors, the same `100` `attack_train`
passages, `20`-token seeds with the header the memorisers were fine-tuned with, `200` new tokens,
temperature `1.0`, `--seed 1234`, `--batch-size 32`, `--experts-impl eager` for the two Mixtral
anchors, and the adversarial selector the memorising `Llama-3.1-8B`'s own likelihood. Prefix
`selfix256_<tag>`. **Every $n \le 64$ row here is a fresh draw, not a reproduction**: the pool is
drawn passage-major at $\max n$, so raising $\max n$ from $64$ to $256$ changes which random draw
every passage after the first receives. The committed $n \le 64$ numbers are therefore not
comparable passage by passage and are **not** used as expectations.

**Part B --- the three clean arms whose published numbers read the served pick, re-run on the
identical pool.** Same command as the arm on record, prefix `selfix_clean_<arm>`:

| re-run | arm on record | anchor, memoriser | host |
|---|---|---|---|
| `n256` | `selection_extraction_n256` | TinyComma-1.8B, `memorizing_llama8b` | local |
| `paraphrase` | `selection_extraction_paraphrase` | TinyComma-1.8B, `memorizing_llama8b` | local |
| `multilingual` | `selection_extraction_multilingual` | Pleias-3B, `memorizing_multilingual` | host B |

Same `--max n`, seed and batch size as the arm on record, so the anchor's draws are the **same
draws** and only the selector's pick can move. That is what makes Part B an exact before/after on
the defect.

## Gates, read before any band

- **G0, the instrument.** Each arm's per-passage `risky_alone_recall` equals its counterpart's on
  record passage for passage (Part A: `contam_<tag>_per_passage.csv`; Part B: the arm in the table).
  The $k=-1$ draw is seeded afresh and does not depend on $\max n$ or on the selector, and on this
  host it has reproduced bit for bit in thirteen arms (`0.3925 / 0.8154 / 78.0%`). A failure is
  **INVALID**, not a result, and stops the reading of that arm.
- **G1, pool identity (Part B only).** `anchor_mean_recall`, `anchor_max_recall`, `recall_n1`,
  `lcs_n1` and `rouge_n1` equal the arm on record passage for passage. If not, the before/after is
  not on one pool and that re-run is INVALID.
- **G2, served is in the pool (every arm, pathwise).** `recall_n{n} <= oracle_recall_n{n}` and
  `rouge_n{n} <= oracle_rouge_n{n}` at every passage and $n$; the oracle is non-decreasing in $n$;
  `oracle_recall_n{max}` equals `anchor_max_recall`; `pool_ge_0p8 >= 1` exactly when
  `anchor_max_recall >= 0.8`, and likewise at `0.01`. These cannot fail except through a bug.

All three are computed by `analysis/selector_n256.py`, which was mutation-tested against synthetic
per-passage files before any arm ran (`tests/test_selector_n256.py`).

## Bands

The event is feat-111's, unchanged: `E_08 = {nv_recall >= 0.8}` (most of the passage reproduced),
with `E_001 = {nv_recall >= 0.01}` reported beside it at every anchor and every $n$ and read for
nothing, because its base rates of `0.57`--`0.98` make $n \times$ base exceed `1` at every $n \ge 2$.

**B1 --- the published claim, re-measured with the corrected selector.** $A(64) = \mathrm{rate}(64)
/ \mathrm{rate}(1)$ on `E_08`, the estimator on record, at every anchor with $\mathrm{rate}(1) > 0$
in this arm.

| reading | band |
|---|---|
| SATURATES | $A(64) \le 4$ at every readable anchor (feat-111's band, unchanged) |
| GROWS | some readable anchor has $A(64) > 4$ |

**We predict GROWS.** The direction of the defect's bias cannot be recovered from the stored files
--- a short candidate's window mixed memorised prompt tokens, which score high, with padding, which
scores anything --- so the prediction rests on the mechanism instead: a selector that reads the
memoriser's likelihood correctly should serve a memorised draw whenever the pool holds one, and at
$n=64$ that is more often than the published rates.

**B2 --- Review 4's $n=256$, against the per-draw base rate.** $\mathrm{rate}(1)$ is one draw per
passage; the pool gives $256$. Let $\hat p = \sum$ `pool_ge_0p8` $/ (100 \times 256)$, the anchor's
own `E_08` rate per draw, which is the quantity Proposition 1 multiplies. Where the bound
$256\hat p$ is informative, $0 < 256\hat p < 1$, the fraction of it realised is
$T = \mathrm{rate}(256) / (256\hat p)$, which lies in $[0, 1]$ pathwise.

| reading | band |
|---|---|
| TIGHT | $T \ge 0.5$ at some readable anchor --- the adversary realises at least half of what the certificate permits |
| LOOSE | $T \le 0.1$ at every readable anchor |
| BETWEEN | otherwise |
| NO READABLE ANCHOR | no anchor has $0 < 256\hat p < 1$ |

**We predict LOOSE.** Memorisation is passage-specific, so an anchor's `E_08` draws should cluster
on the few passages it holds, and one passage counts once however many of its draws leak --- which
is exactly the gap between $\Pr[\text{any of } n \in E]$ and $n\,p$. If they are instead spread
across passages the union bound is nearly tight at small $p$, and the paper's reading of the slack
changes with it. $A(256) = \mathrm{rate}(256)/\mathrm{rate}(1)$ and $\mathrm{rate}(n)/\hat p$ at every
$n$ are reported beside it, as are anchors with $256\hat p \ge 1$, where the bound is vacuous and only
the realised rate means anything.

**B3 --- does the served rate still climb from $64$ to $256$?** Per anchor, on `E_08`, paired over
the same passages (the $n=64$ pool is the first $64$ draws of the $256$): $b$ passages served in
`E_08` at $256$ and not at $64$, $c$ the reverse, exact two-sided McNemar.

| reading | band |
|---|---|
| STILL GROWING | some anchor has $b > c$ with $p < 0.05$ |
| SATURATED BY 64 | no anchor does |

**We predict SATURATED BY 64**, from the committed curves, which were flat from $8$ to $64$ at four
of the five readable anchors.

**B4 --- the clean anchors' non-literal counts, before and after, on one pool.** For each Part B
re-run, the served ROUGE-L $\ge 0.5$ count at every $n$ (and $\ge 0.3$ for `multilingual`, which is
what Appendix I quotes), and the **oracle** count: passages whose best draw reaches the threshold.

| reading | band |
|---|---|
| HOLDS | the corrected served counts are $0$ at every $n$ in all three re-runs |
| CHANGES | any corrected served count is non-zero |

and, separately, **SELECTOR-FREE** if every oracle count is $0$ --- no draw in any pool reaches the
threshold, so the claim needs no selector at all --- or **SELECTOR-DEPENDENT** otherwise. **We
predict HOLDS and SELECTOR-FREE**: a clean anchor that reproduces nothing near-verbatim should not
reach half a passage's longest common subsequence by chance.

## What the manuscript does with each outcome, fixed now

- **The published $1.0$--$4.0$ is replaced whatever B1 reads**, because it was measured with the
  defective selector; the paper says so in one sentence and gives the corrected range at $n=64$ and
  at $n=256$ beside it. Under GROWS or TIGHT, the Appendix I sentence that the adversary realises
  only a small fraction of the permitted amplification is withdrawn and replaced by the measured
  fraction; under TIGHT it goes in the main text beside the bound, since it would mean the
  certificate's $n\times$ is nearly attained exactly where the anchor leaks least.
- **Review 4's question is answered with per-anchor rates at $n \in \{1, 8, 64, 256\}$** in one
  appendix table, the event defined in its caption.
- **B4 HOLDS and SELECTOR-FREE:** the ROUGE-L rows stand and gain the stronger statement that no
  draw reaches the threshold. **CHANGES:** the rows are corrected to the new counts and the text
  that quotes them is rewritten; nothing is averaged with the old values.
- The headline near-verbatim zero is not re-read here; it is selector-free already.

## Excluded in advance

- Quoting any committed $n > 1$ contaminated or ROUGE-L number beside the corrected one as if both
  were valid, or pooling the two.
- Dropping an anchor, a re-run, an $n$ or an event; reading B1 or B2 on `E_001` if `E_08` is
  unfavourable; re-defining $T$'s readable set after seeing $\hat p$.
- Changing seeds, passages, batch size, temperature, seed length or thresholds after a number
  exists.
- Reading any band on an arm whose G0, G1 or G2 failed.
- Treating Part A's $n \le 64$ rows as a reproduction of the committed ones: they are an
  independent draw by construction.

## Compute

Measured, not estimated: the committed $n=256$ arm on the audited anchor took **65 min** on one
local A100 (`output/logs/n128_card1.log`), and the twelve contaminated arms took **14--43 min** each
at $\max n = 64$ (`output/logs/contam_*.log`). Four times the draws and four times the scoring puts
Part A at about **19 GPU-hours** (the two Mixtral anchors about 2.5 h each), Part B at about **2**.
**About 21 GPU-hours in all, under the 24-hour threshold.** Part A and the two local Part B re-runs
run on the local A100s, the host every committed counterpart ran on, so G0 is bit-exact; the
multilingual re-run runs on host B, where its memoriser and corpus live and where its counterpart
was generated.

## Scoring log

### Part B, `multilingual`, landed 2026-09-23 (host B) --- gates G0, G1, G2 PASS

`selfix_clean_multilingual` against `selection_extraction_multilingual`: the memoriser's `k=-1`
draw identical on 100/100 passages (`0.5455 / 0.9882 / 94.0%`), the anchor's pool identical on
every G1 field, and G2 clean. The corrected selector serves a different completion on **39 of 100**
passages at $n=8$ and **51 of 100** at $n=64$ --- the defect was moving about half the picks. B4 is
read once all three Part B re-runs are in, not on this one.

### Part B, `n256`, landed 2026-09-23 (local GPU 4) --- gates G0, G1, G2 PASS

`selfix_clean_n256` against `selection_extraction_n256`: the memoriser's `k=-1` draw identical on
100/100 passages, the anchor's pool identical on every G1 field (so the 25,600 draws are the same
draws), and G2 clean. The corrected selector serves a different completion on **74, 80 and 66 of
100** passages at $n = 8, 64, 256$: on the audited anchor most published picks at $n > 1$ came from
the defective ranking. B4 waits for `paraphrase`.

### B4 READ, 2026-09-23 --- HOLDS and SELECTOR-FREE, as predicted

All three Part B re-runs pass G0, G1 and G2. The corrected served ROUGE-L count is `0` at every $n$
in all three, at the registered `0.5` and at the `0.3` Appendix I quotes for `multilingual`; and the
**oracle** count is `0` too --- no draw in any of the three pools reaches ROUGE-L `0.3`, let alone
`0.5` --- so the non-literal claim needs no selector at all. The fix moved the served pick on
`74/80/66` of 100 passages at $n = 8/64/256$ (`n256`), `69/76` at $n = 8/64$ (`paraphrase`) and
`39/51` (`multilingual`). Per the registration the ROUGE-L rows stand and gain the stronger,
selector-free statement.

### Declared addition, 2026-09-23, before it runs --- one more identical-pool re-run, descriptive only

Reading `tab:extraction` against B4 found selector-dependent numbers this registration did not
list: the table's **longest-substring row** and Appendix I's multilingual **mean** `lcs_word` and
ROUGE-L are means over the *served* pick at $n > 1$, so the defect moved them too, and the table's
$n = 2, 4, 16, 32$ columns come from an arm Part B does not re-run --- the audited anchor's
full-grid arm, `selection_extraction`, re-run on 2026-09-12 at $n \in \{1,2,4,8,16,32,64\}$ after
the tokenizer fix. It is re-run once more, on its identical pool and with the corrected selector,
as `selfix_clean_grid64`: `--risky-model output/memorizing_llama8b --n-values 1 2 4 8 16 32 64
--limit 100`, every other flag at its default (batch `32`, seed `1234`). That the default batch is
the one the committed arm used is checked rather than assumed: its `k=-1` control reads `0.3925`,
the batch-32 signature (caution (u)), and G0 and G1 must pass passage for passage or the re-run is
INVALID and no number from it is used. **No band is attached**: its role is to supply corrected
descriptive values, reported beside the committed ones, and it cannot change B4, which is read
above on the three registered arms only.

### Part A gates, read as anchors land (2026-09-23 08:03) --- no band computed

Three of the twelve contaminated anchors have finished (`selfix256_{llama32_1b,llama32_3b,pleias350m}`).
On each, **G0 PASS** --- `risky_alone_recall` equals `contam_<tag>_per_passage.csv` on all `100`
passages, so the memoriser's `k=-1` draw is the one on record and only the selector changed --- and
**G2 PASS** (served within the pool at every passage and `n`, the oracle non-decreasing and equal to
`anchor_max_recall` at `n=256`, the pool counts consistent). Read with `gate_g0`/`gate_g2` alone;
`part_a` and `readings` were not run, so no rate, amplification or McNemar count exists yet. B1--B3
are read once all twelve have landed.
