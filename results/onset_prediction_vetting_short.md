# Pre-registration: the licensed anchors at the short vetting rungs (feat-183)

**feat-183.** Committed **2026-09-24**, before any rung below has run, at the user's instruction to use
host B's idle capacity for what is still owed. Nothing above `## Scoring log` is edited after the first
rung starts.

## Why

feat-180 (`results/onset_prediction_vetting_ladder.md`) measured the anchor-vetting screen from `20` to
`200` prefix tokens and read **V1 NON-MONOTONE**: a model known to hold the work can leak on fewer
passages at a longer prefix. By V4 the paper now recommends a schedule, and the Ethics Statement says to
screen an anchor *at every prefix length up to the longest genuine prefix the deployment accepts*.

Our own six licensed anchors were screened only at `100`, `150` and `200` tokens. That was fixed in
advance, on the premise that a shorter prefix gives less to lock onto --- a premise that holds only if the
screen is monotone, which V1 refuted. So the paper currently says, correctly, that below a hundred tokens
the licensed anchors are *unmeasured*. That is a recommendation the paper does not follow on its own
anchors. This arm runs the rungs the recommendation asks for.

## What runs

The committed screen, character for character, with `--seed-tokens` the only thing that varies --- the
`vet` command of `scripts/run_selfix_vetladder_local.sh` and `scripts/run_vet150_hostb.sh`:

```
--raw-prompt --split test --novel harry_potter --limit 50 --seed-tokens L
--max-new-tokens 200 --n-values 1 8 64 --batch-size 8 --temperature 1.0 --seed 1234
--risky-model output/memorizing_llama8b
```

| anchor (`--safe-model`) | tag | rungs `L` |
|---|---|---|
| `jacquelinehe/tinycomma-1.8b-llama3-tokenizer` | `tinycomma` | 20, 35, 50, 75 |
| `common-pile/comma-v0.1-2t` | `comma7b` | 20, 35, 50, 75 |
| `common-pile/comma-v0.1-1t` | `comma1t` | 20, 35, 50, 75 |
| `alea-institute/kl3m-003-1.7b` | `kl3m17b` | 20, 35, 50, 75 |
| `PleIAs/Pleias-1.2b-Preview` | `pleias12b` | 20, 35, 50, 75 |
| `PleIAs/Pleias-3b-Preview` | `pleias3b` | 20, 35, 50, 75 |

Twenty-four rungs, prefix `vetladder_L<L>_<tag>`, the same 50 *Harry Potter* passages. **The statistic is
the screen's own**: a passage *leaks* when `anchor_max_recall > 0`, the maximum over the anchor's 64 draws.
These are the rungs feat-180 ran on the 70B and both OLMo models (`20`, `50`, and for the 70B `35`, `75`),
so every rung here has a measured reference on a model that holds the work.

**Host B.** The local A100s are held by another project's vLLM servers. feat-180's declared host check
already measured host B as the same instrument (`hostcheck_memoriser_n1`: `82` of `100` against `78` locally,
`z = +0.707`, PASS), and every file of TinyComma and KL3M-1.7B hashed identically on both hosts. The other
four anchors must hash identically on both hosts (weights, config, tokenizer) before any of their rungs
starts; an anchor that does not is run locally or not at all, never from a different checkpoint. A leak
drawn on host B is still that anchor reproducing protected text, so host B cannot manufacture a pass or
hide a leak; what it changes is which random draws are taken (caution (as)).

**Placement.** `scripts/local_dispatch.py` on host B, beside feat-182, places by free memory; which card a
job lands on is not part of the command.

**One declared confound, as in feat-180.** The target is the rest of the passage after the prefix, so it
lengthens as `L` shrinks: `257`--`294` tokens minus `L`, so `237`--`274` at `L = 20`. Generation is capped
at `200`, so at these rungs no draw can cover the whole target of the longer passages. A leak is still a
near-verbatim reproduction of protected text the model was not shown; the chance of *some* leak depends on
how much text remains, and that is reported, not corrected.

## Gates, read before the band

- **G0, the corpus.** Every rung holds the same 50 `prompt_id`s as `vet_comma7b`. A rung that does not is
  not read (`analysis/vetting_ladder.py`, unchanged).
- **Hashes**, as above, before launch; recorded below the scoring line.

## Band

**S1 --- do the licensed anchors pass below a hundred tokens?**

| reading | band |
|---|---|
| PASS HOLDS BELOW 100 | all six licensed anchors read `0/50` at every one of `L = 20, 35, 50, 75` |
| PASS BREAKS BELOW 100 | any licensed anchor leaks on at least one passage at any of these rungs |
| NOT READ (incomplete) | a rung is missing and no rung present leaks |

A leak on any subset establishes PASS BREAKS; HOLDS needs all twenty-four. **We predict PASS HOLDS BELOW
100**: all six read `0/50` at `100`, `150` and `200`, and the models that do hold the work leak rarely this
short (the 70B on `1` and `2` of `50` at `20` and `35`; each OLMo on `0` at `20` and `1` at `50`).

feat-180's V1--V4 are not re-read and cannot move: V1 reads only models that hold the work, V2 only the
70B, V3 only licensed rungs past `100`.

## What the manuscript does with each outcome, fixed now

- **Either way**, `tab:vetladder`'s dashes on the licensed rows at `20`--`75` become the measured counts, and
  the caption stops saying those rungs were not run.
- **PASS HOLDS BELOW 100**: Appendix I's *"below a hundred tokens the licensed anchors are unmeasured, not
  clean"* and the Ethics Statement's *"below a hundred they are unmeasured"* are replaced by the
  measurement --- the six anchors read `0` of `50` at every rung from `20` to `200` --- and *"passing is
  evidence, not proof"* stays.
- **PASS BREAKS BELOW 100**: the breaking anchor and rung are named in the **main text** (Section 3's
  *"reads `0.000` at all six licensed anchors"* is scoped to the prefix lengths at which it holds) and in
  the Ethics Statement, and every sentence saying an anchor reproduces no protected passage is scoped to
  the prompts it was measured on.
- **NOT READ**: the rungs that landed go in the table and the wording stays *unmeasured* for the rest.

## Excluded in advance

- Changing the statistic, the corpus, the rungs, the batch size or the seed after any rung has run.
- Dropping an anchor or a rung; reporting only the rungs that favour a reading.
- Re-reading feat-180's V1--V4 on these rungs, or reading a licensed anchor's `0/50` as proof that it does
  not hold the work: the screen is sound, not complete.

## Compute

feat-180's two host-B rungs took about `15` minutes each on an H100 (`output/logs/vetladder_L150_*.log`,
host B). Twenty-four rungs is about **6 GPU-hours**, spread over host B's free memory beside feat-182:
about three hours of wall-clock, well under the 24-hour threshold.

## Scoring log

### Hashes, checked 2026-09-24 before any rung started --- identical

Every file the loader reads for Comma-7B (`common-pile/comma-v0.1-2t`), Comma-1T, Pleias-1.2B and Pleias-3B
--- config, generation config, tokenizer files and all `12` weight shards, `32` files --- hashes identically
in both hosts' `hf_cache` snapshots. Host B's snapshots also hold each repository's `README.md` and
`.gitattributes`, which the loader never reads. TinyComma, KL3M-1.7B and the memoriser were checked for
feat-180 (`17/17`). Launch: `scripts/local_dispatch.py --cards 4,5,6,7 --only selfixR_,vetladder_L` on host B,
one dispatcher for feat-182's remaining jobs and these twenty-four, feat-182 first.

### S1 read 2026-09-24 06:00 IST --- PASS HOLDS BELOW 100

All twenty-four rungs ran on host B and exited `0` (host B's dispatcher, beside feat-182; last placed
21:52 host time). `scripts/sync_status.sh pull`, then `.venv/bin/python analysis/vetting_ladder.py`:
**G0 PASS** (every rung holds `vet_comma7b`'s `50` `prompt_id`s), host check PASS as before (`82` against
`78`, `z = +0.707`).

**S1 PASS HOLDS BELOW 100**: all six licensed anchors read `0/50` at `L = 20, 35, 50, 75`, maximum recall
`0.0000` at every rung. With feat-180's rungs, every licensed anchor reads `0/50` at every rung from `20`
to `200`. V1--V4 re-read unchanged, as they must be (NON-MONOTONE, `L* = 50`, PASS HOLDS, a schedule).
The scorer now labels these twenty-four rungs `this arm, host B` (`analysis/vetting_ladder.py`, `HOST_B`),
which is where the registration put them.

**Prediction:** PASS HOLDS BELOW 100, **right**.

**Manuscript, as registered.** `tab:vetladder`'s licensed dashes at `20`--`75` become the measured `0`s and
its caption stops saying those rungs were not run (it now says they were run separately, bands fixed in
advance, on the second machine with the two `†` rungs). Appendix I's "below a hundred tokens the licensed
anchors are *unmeasured*, not clean" becomes "all six read `0` of `50` at `20`, `35`, `50` and `75` tokens as
well, and so pass at every rung from `20` to `200`"; the Ethics Statement's "below a hundred they are
unmeasured" becomes "read `0` at every rung from `20` to `200`"; "passing is evidence, not proof" stays.
The host-B count in Appendix I becomes twenty-six rungs.
