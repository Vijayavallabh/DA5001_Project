# Pre-registration: extraction onset in the chat serving configuration (feat-206)

**feat-206.** Committed **2026-09-25**, before any token is decoded. Nothing above `## Scoring log` is
edited after the first query.

## Why

Review 4 (sixth round, Q2): the chat-templated meter overtakes selection from `k=3` (`K/S_w = 3.75`,
feat-196). "Please also measure extraction onset in the same serving configuration. If the meter wins
at a budget below onset, it is empirically leak-free but formally vacuous." Onset on record is measured
only on plain text: the audited pair (TinyComma-1.8B with the LoRA memoriser) reaches mean near-verbatim
recall `0.01` at `k = 2.873` `[2.717, 3.451]` (`results/onset_table.csv`).

## What runs

The audited pair's onset sweep exactly (`analysis/composition_attack.py --safe-model
jacquelinehe/tinycomma-1.8b-llama3-tokenizer --risky-model output/memorizing_llama8b --modes single
--windows 50 --limit 100`, split `attack_train`, `20`-token seeds, temperature `1.0`), with one change,
a new flag `--use-chat-template`: every query is wrapped as one user turn of the Llama-3.1 chat
template (`dap.shared.wrap_chat`, what `h1.py --use-chat-template` serves), with the memoriser's
`Complete the prefix:` header inside the turn, and decoding stops at `<|eot_id|>`. Grid: `k = -1, 0`
and the plain sweep's `1.5, 2, 2.6, 3.2, 3.8, 4.5`. Onset and its interval by `analysis/onset_ci.py
--thresh 0.01 --s-x 3.239` (the pair's `s(x)`). Host B, one card -> `output/feat206/chat_onset/`,
`results/chat_onset.csv`.

## Gates

- **G0.** `100` passages at every `k`; no query's spend exceeds its budget (the script's own count).
- **G1, entry (caution (a)).** The memoriser served through the chat template reproduces its passages:
  sampled `k=-1` mean recall at least `0.10`. If it does not, the pair does not enter, no onset is
  quoted, and the manuscript says that this memoriser does not leak when chat-served, so the question
  needs a chat-format memoriser this arm did not train.

## Readings and predictions

- **C1.** G1 passes. *Predicted: passes, with recall below the plain configuration's `0.4921`.*
- **C2.** The chat onset's interval against the crossover `k=3`: ABOVE (lower end above `3`: the meter's
  win at `k=3` is at a budget below onset), BELOW (upper end below `3`), or STRADDLES. *Predicted:
  STRADDLES*, because the plain onset's interval `[2.717, 3.451]` contains `3`.
- **Descriptive.** Recall at every `k`; `k=0` recall (the anchor alone through the template).

## What the manuscript does with each outcome, fixed now

Section 4's chat sentence (the meter overtakes selection from `k=3`, `K/S_w = 3.75`) adds where onset
sits in the same configuration, reading C2's label: ABOVE -- the meter's win is empirically leak-free
for this memoriser and formally vacuous, and the certificate does not say so; BELOW -- the meter wins
only at budgets where this memoriser already leaks; STRADDLES -- the two coincide within the
interval. If G1 fails, that is said instead.

## Excluded in advance

- Any other grid, seed length, temperature, memoriser or template chosen after a query is read.

## Scoring log
