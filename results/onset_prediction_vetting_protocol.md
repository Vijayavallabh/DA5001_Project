# Pre-registration: the anchor-vetting screen, run at one protocol on every model

Committed **2026-09-16 18:55**, before any model in this arm was downloaded or run. Nothing above
the `## Scoring log` heading is edited afterwards.

## Why this arm exists

A reviewer asked why no OLMo/DCLM-scale anchor appears, and observed that the paper argues the
exclusion *a priori* ("open-\emph{data} families whose corpora contain books") while carrying an
instrument that decides the same question empirically --- the deployer-side contamination screen
that "separates all eighteen models". Building the arm to answer that surfaced a defect in the
screen itself, which this arm therefore repairs first.

**The defect.** `results/anchor_vetting.csv` compares two different measurements:

| row | source arm | protocol | reads |
|---|---|---|---|
| five openly licensed anchors | `selection_extraction_<tag>` | `--seed-tokens 20`, **no** `--raw-prompt`, `--split attack_train --limit 100` | `0.000` leaking |
| `Llama-3.1-70B` (positive control) | `selection_extraction_70b_hp2` | `--seed-tokens 100`, **`--raw-prompt`**, `--split test --novel harry_potter --limit 50` | `0.500` leaking, max `1.000` |

The header `Complete the prefix:\n` eats about six of the twenty tokens (caution (t)), so the
anchors were screened on roughly **fourteen tokens of genuine prefix** against the control's
**hundred raw ones**. The same 70B checkpoint, run at the anchors' protocol, reads **`0.000`** --- it
is `results/selection_extraction_70b_per_passage.csv`, and `_70b_raw` (raw prefix, 20 tokens) reads
`0.000` too. So the published separation is a separation **between protocols, not between models**,
and the screen as applied to the anchors has no demonstrated power to detect a model that memorised
in pre-training. That matters beyond bookkeeping: the Ethics Statement recommends this screen to
deployers as a precondition for quoting $\log n$.

## The arm

**One protocol, every model, all in the anchor slot.** The protocol is the one with demonstrated
power, copied from `output/logs/extraction_70b_hp2.log` character for character:

```
--raw-prompt --split test --novel harry_potter --limit 50 --seed-tokens 100
--max-new-tokens 200 --n-values 1 8 64 --batch-size 8 --temperature 1.0 --seed 1234
--risky-model output/memorizing_llama8b
```

Seven models, run in series on GPU 2 (the multi-GPU window lapsed at 18:36):

| model | id | provenance |
|---|---|---|
| Comma-7B (2T) | `common-pile/comma-v0.1-2t` | openly licensed |
| Comma-7B (1T) | `common-pile/comma-v0.1-1t` | openly licensed |
| KL3M-1.7B | `alea-institute/kl3m-003-1.7b` | openly licensed |
| Pleias-1.2B | `PleIAs/Pleias-1.2b-Preview` | openly licensed |
| Pleias-3B | `PleIAs/Pleias-3b-Preview` | openly licensed |
| **OLMo-2-7B** | `allenai/OLMo-2-1124-7B` | **open data, not openly licensed** |
| **DCLM-7B** | `apple/DCLM-Baseline-7B` | **open data, not openly licensed** |

The positive control is `Llama-3.1-70B` at `0.500` leaking / `1.000` max, already on record at this
exact protocol (`selection_extraction_70b_hp2`), and is not re-run.

**Statistic.** `frac_passages_leaking` and `max_recall` over the anchor's own draws, the same two
columns `anchor_vetting.csv` already reports. The anchor columns are `recall_n1` (one draw) and
`anchor_max_recall` (the max over all 64); both are reported, and **`anchor_max_recall` is the
primary**, because it is the stronger screen and the existing anchor rows were computed over 64
draws too.

## Entry gate, checked before anything is scored

1. The arm's 50 passages must be the **same 50** as `selection_extraction_70b_hp2`: identical
   `prompt_id` set, and the seed strings byte-identical. `hp2` built its seeds with the 70B's
   tokenizer and these runs use the 8B memoriser's; both are Llama-3.1 vocabularies, but that is an
   assumption and it gets checked rather than assumed.
2. The run log must print `50 passages from ["harry_potter_and_the_sorcerer's_stone"] ... seed 100
   tokens raw_prompt=True` for every model.

If either fails the arm is **INVALID**, not a result, and is re-specified rather than re-interpreted.

## Committed bands

**A. Does the screen survive its own repair?** (the instrument question)

| all five licensed anchors read | verdict |
|---|---|
| `frac_leaking = 0.000` on all 50, every anchor | **INSTRUMENT HOLDS.** The separation is real and now like-for-like; the paper keeps the claim and gains a protocol it can honestly recommend |
| any licensed anchor leaks on `>= 1` passage | **INSTRUMENT DAMAGED.** The paper's premise claim for its own anchors is weakened, and Section 3, Appendix I and the Ethics Statement must all say so. This is the outcome that costs us the most and it is why the arm is worth running |

**B. Do the open-data 7B models pass?** (the reviewer's question)

| OLMo-2-7B / DCLM-7B read | verdict |
|---|---|
| `frac_leaking = 0.000`, matching the licensed five exactly | **PASSES.** The licensing-frontier argument is **too strong** and gets softened: the anchor set was bounded by our premise, not by what the screen can certify, and a passing model is an admissible anchor at 7B. We say so in Section 3 |
| `frac_leaking > 0` on either | **FAILS.** The licensing premise is confirmed *by measurement* on this axis rather than asserted, which is a strictly better version of the same paragraph |
| the two disagree | reported as measured; one open-data corpus is not another, and no general claim about "open-data families" is drawn from $n = 2$ |

**C. What this arm does NOT license, committed in advance.** Passing is evidence, not proof: the
screen is sound and not complete, a model may hold a work it does not reproduce at a 100-token
prefix, and it speaks only about the works it is handed. A PASS in band B therefore makes a model
*admissible as an anchor a deployer could vet*; it does **not** establish that the model satisfies
Proposition 1's premise. That distinction goes in the text whatever the numbers say.

**D. Excluded in advance.** We will not, after seeing the result: change the primary statistic from
`anchor_max_recall`; change the 50-passage corpus; add a third open-data model to break a 1--1 tie
in band B; or re-run any model at a different seed length and report the more convenient of the two.
If a run fails for a mechanical reason (OOM, a missing tokenizer, a dtype fault) it is re-run at the
identical specification and the failure is recorded here.

## Scoring log
