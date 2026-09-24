# Empty answers in the headline head-to-head — a POST-HOC check, with no committed bands

This is **not** a pre-registration and there are **no committed bands**; it is named `*_note.md` so it
cannot inflate the pre-registration count. Two referee reports (2026-09-24, sixth round) observed that
the reward prefers empty drafts, that the judge scores an empty answer near parity against the risky
model's full text, and that the audited anchor's first draw is empty on `64` of `500` prompts, and asked
whether the headline `D3 = +0.0505 [+0.0155, +0.0860]` (feat-184 A1) survives counting an empty answer
as a loss.

## 1. Empty answers scored as losses (no judge call)

`.venv/bin/python analysis/empty_as_loss.py --out results` -> `results/empty_as_loss.csv`. It re-reads
the committed per-prompt levels of the repaired-text pass
(`results/order_averaged_h2h_per_prompt_deecho.csv`) and sets the level of every empty served text to
`0` (`0.5` if the opponent's is empty too). Served empties: selection `n=64` `41`, its `n=1` control
`64`, the meter at `k=10` `0`, its anchor control `62`, the opponent `0`.

| rule | D1 selection gain | D2 metered gain | D3 |
|---|---|---|---|
| as judged | `+0.1015` | `+0.0510` | `+0.0505 [+0.0155, +0.0860]` |
| empty is a loss | `+0.1190` | `+0.1035` | `+0.0155 [-0.0245, +0.0555]` |

The headline does not survive this rule: the interval covers zero. The paper says so in Section 4.

## 2. A selection rule that never serves an empty draw

The remedy belongs in the rule rather than the judge: serve the highest-scoring **non-empty** draw.
It is still a selection rule over the same `n` draws, so Proposition 1 certifies it at `log n`
unchanged. `OMP_NUM_THREADS=32 .venv/bin/python analysis/nonempty_rule.py --out results` ->
`results/nonempty_rule.csv`. Only the `41` prompts whose committed pick is empty change pick, and only
they are judged, by judge B (Phi-3.5-mini) on **CPU in float32**, because every GPU on both hosts was
held by another job. The other `459` prompts keep their committed bf16 GPU levels.

**Control, read before any new level was used:** the CPU judge re-scored the `41` committed picks and
reproduced the committed level on `37` of `41`, so about one near-tie verdict in ten moves between bf16
on a GPU and float32 on a CPU (caution (as)); the new picks' levels carry that noise.

| rule | D1 | D2 | D3 | served levels, selection minus meter |
|---|---|---|---|---|
| as judged | `+0.1015` | `+0.0510` | `+0.0505 [+0.0155, +0.0860]` | `+0.0655 [+0.0390, +0.0915]` |
| empty is a loss | `+0.1595` | `+0.1035` | `+0.0560 [+0.0195, +0.0930]` | `+0.0655 [+0.0390, +0.0915]` |

Under the non-empty rule, with empty answers scored as losses, the difference is back above zero at
`+0.056`. The paper reports both readings side by side (Section 4, Appendix `app:empties`) and does
not replace the headline with the rule chosen after seeing the first reading.
