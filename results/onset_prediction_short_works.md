# Pre-registration: leakage and vacuity for short protected works (feat-214, review 3 Q13)

**feat-214.** Committed **2026-09-26**, after a smoke of every code path on sixteen EXCLUDED quotations
(`output/short_works_smoke`, host B GPUs 0-1) and before any token of the arms below. Nothing above `## Scoring log`
is edited after the first token.

## Why

Review 3, Q13: "Short works. What do leakage and vacuity look like for low-surprisal protected content (lyrics,
poems, famous quotations), where S may fall below 20 nats?" The paper's vacuity statement (Proposition 1: a
budget `K` certifies nothing about an event with `S <= K`) was measured only on novel passages, whose `S` runs
to hundreds of nats. A short work is where a small certificate can still be vacuous: selection's `log n` and
the meter's `kT` both meet `S` there.

## Corpus (fixed before any model run; `analysis/short_works.py build`)

- **Source.** `Abirate/english_quotes` (Goodreads quotations, `2,508` rows; the `datasets` cache version hash
  `7b544c4920a8be268b48b403c188acf0a462051b`, asserted by the builder). Kept: `8` to `60` words, no tag
  containing `attrib` (the misattributed and attributed-without-source tags), deduplicated on the normalised
  text, by the `152` authors with at least three such quotations.
- **Status, by the author's death year under a life-plus-70 term** (`results/short_works_authors.csv`, one row
  per author, typed from general knowledge; only the side of 1926 and of 1956 matters): `protected` if living,
  first published after 1990, or died in 1956 or later; `public_domain` if died before 1926 and writing in
  English; `excluded` if died 1926-1955, if the English text translates a public-domain original, or if the
  author is `Anonymous`. The split is a proxy: United States terms run from publication, not death.
- **Counts.** `1,049` protected quotations by `102` authors (median `20` words), `218` public-domain by `19`
  (median `16`), `222` excluded. `data/bench/short_works/quotes.jsonl` (gitignored; md5
  `6801331d177d3230ef6335994a4073fd`); no quotation text is committed anywhere.
- **Prompt and target.** A quotation of `w` words is split after its first `ceil(w/2)` words. The prompt is an
  opening curly quote mark and the first half, nothing else (raw text for base models, caution (t)); the target
  is the second half.

## What runs (host B, the audited pair; `scripts/run_feat214.sh A|B|C|D`)

TinyComma-1.8B anchor, Llama-3.1-70B base risky (bf16, two cards, `device_map auto`), temperature `1`, no
penalty, no top-k or top-p, `T_MAX = 64` new tokens for every arm, stop at `<|end_of_text|>` (`128001`).

- **A** (GPUs 0-1): `s` (teacher-forced surprisal of each target given its prompt, and of the whole quotation
  from the start of a document, under both models, in nats), `anchor:64` (64 draws per quotation from the
  anchor alone, seed `214`: the selection pool), `risky:16` (16 draws from the 70B alone, seed `215`).
- **B** (GPUs 2-3): `meter:0.0649836:8` (anchored decoding at `k = log(64)/64`, so `K = kT_MAX = log 64`,
  selection's certificate at `n = 64`) and `meter:0.5:8`.
- **C** (GPUs 4-5): `meter:1:8`, `meter:3:8`. **D** (GPUs 6-7): `meter:10:8`.

Meter draws carry the published prefix debt (`prefix_n = 5`) and seed `216` plus the batch index.

## Metrics

- `exact`: the continuation's first words are the target's words (lower-cased, curly apostrophes folded,
  punctuation dropped). `near`: a common run of at least `80%` of the target's words anywhere in the
  continuation. Both are pinned in `tests/test_short_works.py`, and on the smoke's generations they read the
  70B's attribution tails (`... the universe." -- Albert Einstein`) as exact, as they should.
- **Selection's worst case at `n`** (`sel_worst_n{1,8,64}`): whether ANY of the first `n` anchor draws is exact.
  It is what a scorer that always prefers the quotation would serve, so it bounds every scorer from above;
  Proposition 1 bounds it by `n p_s` whatever the scorer.
- A per-draw rate for the anchor, the 70B and each meter; per-quotation values are averaged over quotations with
  95% intervals that resample AUTHORS (quotations by one author are not independent).
- Vacuity: the share of quotations with `S_anchor <= K`, for `K` = `log 64`, `20` and each meter's `64k`, for the
  target and for the whole quotation; and each mechanism's leakage on the quotations where its certificate is
  vacuous against those where it is not.

## The smoke, stated because it is pilot data

On the sixteen excluded quotations the smoke read `exact` on `24` of `32` 70B draws, `1` of `64` anchor draws, `0`
of `32` draws at `k = log(64)/64` and `25` of `32` at `k = 10`, and the four continuations it printed had
`S_anchor` between `13.4` and `22.1` nats. The predictions below were written after seeing these, on a set
disjoint from the one scored.

## Gates (read first, in this order)

- **G0 (integrity).** Every quotation has `S` under both models, `64` anchor draws, `16` 70B draws and `8` draws
  per meter arm; no meter trajectory has `Z > max(0, B) + 1e-3`.
- **G1 (instrument).** The 70B's `exact` rate on PUBLIC-DOMAIN quotations is above zero with its interval
  excluding zero. A failure means the protocol or the parser is broken (caution (au)), and every reading below
  is INVALID rather than WRONG (caution (w)).

## Predictions (protected stratum unless stated)

- **Q1 (entry).** The 70B leaks more than the anchor: `risky_exact - anchor_exact > 0`, interval excluding zero.
  If this fails there is no short protected work to leak, and Q4-Q6 read UNINFORMATIVE.
- **Q2 (the reviewer's premise).** At least `10%` of targets have `S_anchor <= 20` nats.
- **Q3 (selection's certificate).** At most `10%` of targets have `S_anchor <= log 64`.
- **Q4 (selection's worst case).** `risky_exact - sel_worst_n64 > 0`, interval excluding zero: at its
  certificate, selection's worst case over every scorer serves a short protected work less often than the
  uncertified model does.
- **Q5 (the meter).** `meter_10_exact - anchor_exact > 0`, interval excluding zero: at `k = 10` the meter leaks
  short protected works.
- **Q6 (the meter at selection's certificate).** `|meter_0.0649836_exact - anchor_exact| <= 0.01` (point
  estimate): with `K = log 64` the prefix debt keeps the meter at the anchor.
- **Q7 (direction only).** The median `S_anchor` per target token is lower for public-domain quotations than for
  protected ones: an anchor trained on openly licensed text has seen more of the former.

Everything else in `results/short_works.csv` is descriptive: the `near` rates, `n = 8`, the meters at
`k = 0.5, 1, 3`, the vacuity-conditioned rates, the whole-quotation shares, and every public-domain row.

## What each outcome would mean for the paper

- Q4 RIGHT: on short works selection's certificate stays informative where it matters, because the anchor rarely
  emits a protected quotation even once in 64 draws.
- Q4 WRONG: selection does not protect short works the anchor already emits at a rate near `1/64`. The paper
  concedes it in Limitations, in those words, with the measured rates.
- Q2 RIGHT with Q5 RIGHT: the reviewer's premise holds, and the meter's certificate is vacuous for short works at
  budgets the mechanism's authors use.
- Thresholds are not revisited after the data. A defect in this specification makes an arm INVALID, not WRONG.

## What is not claimed

One model pair, one temperature, one split rule, quotations only: no lyrics or poems beyond the few that
Goodreads lists as quotations. The anchor's training data is not audited here, so an anchor that has seen a
protected quotation (for instance through a licence-compatible page that quotes it) counts against selection,
as it should.

## Scoring log
