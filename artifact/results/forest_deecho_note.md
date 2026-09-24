# Note: the forest plot's judged rows, re-judged on recovered text (descriptive, no bands)

Committed **2026-09-24, before any judge call below.** Nothing above `## Scoring log` is edited
afterwards.

## Why

Caution (bc): every pool recorded before 2026-09-24 carries the tail of its prompt at the start of
`generation`, and the judge reads that text. `results/deecho_rejudge_note.md` re-judged every
order-averaged pass the appendix quotes; the forest plot (`fig:breadth`) reads a different family,
the single-order two-judge sweeps of `analysis/selection_scaling.py`, and those were not re-judged:
`selection_breadth_deecho.csv` re-measured only the entry gate, and its judged gains are the committed
ones. This re-runs each of them with `--deecho` and nothing else changed: same pool, same committed
reward cache (read, never rewritten: the launcher checks its hash), so the same selection picks; same
opponent directory, same two judges (B `Phi-3.5-mini-instruct`, C `Meta-Llama-3.1-8B-Instruct`), same
`--seed 8801`, same grid. The presentation-order draw is keyed on `(prompt, pick)`, which the repair
does not change, so each item is shown in the order it was shown on record. What changes is the text
the judge reads (`served_generation`) and the prompt it is shown (the corpus prompt, caution (aa)).

## What is re-run (local GPU 1; `scripts/run_forest_rejudge.sh`)

| sweep | pool | opponent | grid | reward cache | new tag |
|---|---|---|---|---|---|
| audited TinyComma-1.8B | `sel_anchor64` | `sweep_plain` | `1..64` | `selection_rewards64.csv` | `_deecho` |
| Pleias-1.2B | `sel_pleias12b_8` | `sweep_plain` | `1..8` | `selection_rewards_pleias12b.csv` | `_pleias12b_deecho` |
| KL3M-1.7B | `sel_kl3m17b_8` | `sweep_plain` | `1..8` | `selection_rewards_kl3m17b.csv` | `_kl3m17b_deecho` |
| Comma-7B | `sel_comma7b_8` | `sweep_plain` | `1..8` | `selection_rewards_comma7b.csv` | `_comma7b_deecho` |
| Comma-7B (1T) | `sel_comma1t_8` | `sweep_plain` | `1..8` | `selection_rewards8_comma1t.csv` | `_comma1t_deecho` |
| Pleias-3B | `sel_pleias3b_8` | `sweep_plain` | `1..8` | `selection_rewards8_pleias3b.csv` | `_pleias3b_deecho` |
| AlpacaEval, TinyComma | `alpaca_anchor8` | `alpaca_risky` | `1..8` | `selection_rewards_alpaca.csv` | `_alpaca_deecho` |
| AlpacaEval, Comma-7B | `alpaca_comma7b_8` | `alpaca_risky` | `1..8` | `selection_rewards_alpaca_comma7b.csv` | `_alpaca_comma7b_deecho` |
| MT-Bench, TinyComma | `mtbench_anchor8` | `mtbench_risky` | `1..8` | `selection_rewards_mtbench.csv` | `_mtbench_deecho` |

Then `analysis/selection_breadth.py --rejudged` -> `results/selection_breadth_rejudged.csv`. The
exact-match rows of the forest (GSM8K, TriviaQA) have no judge and are not part of this note.

## Readings on record (gain of `n` over the same anchor's `n=1`, single order)

| sweep | `n` | judge | gain on record | reading |
|---|---|---|---|---|
| audited | 8 | B | `+0.0540 [+0.0130, +0.0950]` | POSITIVE |
| audited | 8 | C | `+0.0730 [+0.0270, +0.1200]` | POSITIVE |
| audited | 64 | B | `+0.1420 [+0.0970, +0.1870]` | POSITIVE |
| audited | 64 | C | `+0.1230 [+0.0760, +0.1710]` | POSITIVE |
| Pleias-1.2B | 8 | B | `+0.0290 [-0.0120, +0.0680]` | COVERS ZERO |
| Pleias-1.2B | 8 | C | `+0.0710 [+0.0270, +0.1150]` | POSITIVE |
| KL3M-1.7B | 8 | B | `+0.0390 [+0.0050, +0.0760]` | POSITIVE |
| KL3M-1.7B | 8 | C | `+0.0510 [+0.0130, +0.0900]` | POSITIVE |
| Comma-7B | 8 | B | `+0.1110 [+0.0720, +0.1480]` | POSITIVE |
| Comma-7B | 8 | C | `+0.1550 [+0.1060, +0.2000]` | POSITIVE |
| Comma-7B (1T) | 8 | B | `+0.0260 [-0.0100, +0.0630]` | COVERS ZERO |
| Comma-7B (1T) | 8 | C | `+0.1130 [+0.0690, +0.1570]` | POSITIVE |
| Pleias-3B | 8 | B | `+0.0470 [+0.0030, +0.0890]` | POSITIVE |
| Pleias-3B | 8 | C | `+0.0970 [+0.0560, +0.1400]` | POSITIVE |
| AlpacaEval, TinyComma | 8 | B | `+0.0311 [-0.0006, +0.0621]` | COVERS ZERO |
| AlpacaEval, TinyComma | 8 | C | `+0.0665 [+0.0342, +0.1006]` | POSITIVE |
| AlpacaEval, Comma-7B | 8 | B | `+0.0745 [+0.0416, +0.1075]` | POSITIVE |
| AlpacaEval, Comma-7B | 8 | C | `+0.1671 [+0.1342, +0.2006]` | POSITIVE |
| MT-Bench, TinyComma | 8 | B | `+0.0063 [-0.0875, +0.0938]` | COVERS ZERO |
| MT-Bench, TinyComma | 8 | C | `-0.0375 [-0.1313, +0.0500]` | COVERS ZERO |

## How it is read, fixed now

- A row **survives** if its sign class (POSITIVE: `lo > 0`; NEGATIVE: `hi < 0`; else COVERS ZERO) is
  unchanged. Levels are not compared: a single-order level carries the position lottery (caution (m)),
  and the gain is the within-sweep paired quantity.
- The forest is re-drawn from the re-judged rows **whatever they read**, and its caption says the
  judged rows are on recovered text. Any row whose class changes is named in the caption or the
  paragraph that reads the figure, with both readings.
- The breadth claim ("B1 GENERALISES": at least two new anchors with a judge-B interval excluding
  zero) is re-read on `selection_breadth_rejudged.csv` by the script's own rule, and the manuscript
  quotes that reading.

## Excluded in advance

- Re-selecting on de-echoed text (a new reward pass); the picks are the committed picks.
- Any other judge, seed, grid or pool, and pooling a re-judged row with its committed twin.

## Scoring log

### Scored 2026-09-24 15:35 IST --- 17 of 20 rows keep their class; KL3M-1.7B loses its interval under both judges and Comma-1T gains one under judge B

`scripts/run_forest_rejudge.sh 1` (local GPU 1, 15:04--15:27), every committed reward cache
hash-identical before and after; `analysis/forest_rejudge_score.py` -> `results/forest_rejudge.csv`;
`analysis/selection_breadth.py --rejudged` -> `results/selection_breadth_rejudged.csv`.

| sweep | `n` | judge | on record | repaired text | survives |
|---|---|---|---|---|---|
| audited | 8 | B | `+0.0540 [+0.0130, +0.0950]` | `+0.0730 [+0.0300, +0.1200]` | YES |
| audited | 8 | C | `+0.0730 [+0.0270, +0.1200]` | `+0.0960 [+0.0500, +0.1410]` | YES |
| audited | 64 | B | `+0.1420 [+0.0970, +0.1870]` | `+0.1410 [+0.0950, +0.1870]` | YES |
| audited | 64 | C | `+0.1230 [+0.0760, +0.1710]` | `+0.1170 [+0.0700, +0.1640]` | YES |
| Pleias-1.2B | 8 | B | `+0.0290 [-0.0120, +0.0680]` | `+0.0370 [-0.0060, +0.0770]` | YES |
| Pleias-1.2B | 8 | C | `+0.0710 [+0.0270, +0.1150]` | `+0.0970 [+0.0520, +0.1430]` | YES |
| **KL3M-1.7B** | 8 | B | `+0.0390 [+0.0050, +0.0760]` | `+0.0300 [-0.0030, +0.0630]` | **NO** |
| **KL3M-1.7B** | 8 | C | `+0.0510 [+0.0130, +0.0900]` | `+0.0360 [-0.0020, +0.0730]` | **NO** |
| Comma-7B | 8 | B | `+0.1110 [+0.0720, +0.1480]` | `+0.1260 [+0.0870, +0.1640]` | YES |
| Comma-7B | 8 | C | `+0.1550 [+0.1060, +0.2000]` | `+0.1410 [+0.0950, +0.1870]` | YES |
| **Comma-7B (1T)** | 8 | B | `+0.0260 [-0.0100, +0.0630]` | `+0.0360 [+0.0010, +0.0740]` | **NO** |
| Comma-7B (1T) | 8 | C | `+0.1130 [+0.0690, +0.1570]` | `+0.1160 [+0.0730, +0.1570]` | YES |
| Pleias-3B | 8 | B | `+0.0470 [+0.0030, +0.0890]` | `+0.0540 [+0.0130, +0.0940]` | YES |
| Pleias-3B | 8 | C | `+0.0970 [+0.0560, +0.1400]` | `+0.1080 [+0.0700, +0.1500]` | YES |
| AlpacaEval, TinyComma | 8 | B | `+0.0311 [-0.0006, +0.0621]` | `+0.0199 [-0.0112, +0.0516]` | YES |
| AlpacaEval, TinyComma | 8 | C | `+0.0665 [+0.0342, +0.1006]` | `+0.0739 [+0.0422, +0.1081]` | YES |
| AlpacaEval, Comma-7B | 8 | B | `+0.0745 [+0.0416, +0.1075]` | `+0.0752 [+0.0441, +0.1056]` | YES |
| AlpacaEval, Comma-7B | 8 | C | `+0.1671 [+0.1342, +0.2006]` | `+0.1516 [+0.1193, +0.1845]` | YES |
| MT-Bench, TinyComma | 8 | B | `+0.0063 [-0.0875, +0.0938]` | `+0.0000 [-0.0750, +0.0813]` | YES |
| MT-Bench, TinyComma | 8 | C | `-0.0375 [-0.1313, +0.0500]` | `-0.0437 [-0.1437, +0.0500]` | YES |

**What the manuscript now says, as fixed above.** The forest is drawn from the repaired rows and its
caption names both moved rows with their recorded readings. Counted over all six anchors under judge
B, the gain still excludes zero at **four of six** --- but the four are TinyComma, Comma-7B, Comma-1T
and Pleias-3B, **two families rather than three**, so the abstract, the introduction and Section 4
now say "two families", and the largest gain is Comma-7B's `+0.126 [+0.087, +0.164]` (was `+0.111`).
Section 4's `rho = +0.543` on the two-judge mean is unchanged by the re-judge (same ranks); under
judge B alone it moves from `-0.029` to `+0.429`. The script's own B1 rule (new anchors that pass the
entry gate on the true generations, judge-B interval above zero) reads **PARTIAL**: one of Pleias-1.2B,
KL3M-1.7B and Pleias-3B (only Pleias-3B), where the same rule on the recorded judgments with the
repaired gate read two. On non-empty prompts the gate-failing anchors' judge-B gains move by at most
`0.012` (Comma-7B `+0.126 -> +0.114`); the caption's earlier "under `0.01`" no longer held and was
changed. The limitations paragraph quotes the AlpacaEval and MT-Bench rows on the repaired text
(max MT-Bench half-width `0.097`, printed `0.10`).
