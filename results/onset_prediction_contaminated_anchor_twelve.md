# Pre-registration: the contamination axis at twelve anchors

Committed at **12:25 on 2026-09-14**, while feat-111's arms were still sampling and **before any
`results/contam_*.csv` existed** --- verified by `ls results/contam_*` returning nothing at that
time, recorded in the commit that adds this file. Nothing above the `## Scoring log` line is edited
afterwards.

## Why this extends feat-111 rather than replacing it

`results/onset_prediction_contaminated_anchor.md` registered five contaminated anchors and made
**N3 descriptive with no band**, in those words, because five points cannot support a coefficient
--- which is what C2 had just taught us at six
(`results/onset_prediction_selection_breadth_six.md`).

Seven further LoRA-memorised models are already on disk from the onset work. Running them costs no
new training and roughly two hours on cards that are otherwise idle, and it takes the axis from
five points to twelve. **This is not anchor-shopping**: no number from any of the five had been
read when this was written, and N1 and N2 below are feat-111's bands unchanged. What changes is
that N3 becomes readable.

| added anchor | base | architecture |
|---|---|---|
| `mem_kl3m-002-170m` | `alea-institute/kl3m-002-170m` | GPT-NeoX |
| `mem_kl3m-003-3_7b` | `alea-institute/kl3m-003-3.7b` | Mixtral, 4 experts |
| `mem_llama32-1b` | `meta-llama/Llama-3.2-1B` | Llama |
| `mem_llama32-3b` | `meta-llama/Llama-3.2-3B` | Llama |
| `mem_opencalm3b` | `cyberagent/open-calm-3b` | GPT-NeoX |
| `mem_Pleias-350m-Preview` | `PleIAs/Pleias-350m-Preview` | Llama |
| `mem_qwen25-7b` | `Qwen/Qwen2.5-7B-Instruct` | Qwen2 |

Everything else is feat-111's protocol unchanged: the same `100` `attack_train` passages and seeds,
`20`-token seeds, `200`-token cap, temperature `1.0`, `n \in \{1, 8, 64\}`, the adversarial
selector (the memorising `Llama-3.1-8B`'s own likelihood), and every band read against **each
arm's own `n=1` row**.

## The one protocol departure, declared here

`kl3m-002-520m` and `kl3m-003-3.7b` are **Mixtral** mixtures of experts, and the expert gather
`down_proj[expert_ids]` exhausts an idle 80 GB card at batch `32`: the first `kl3m-002-520m` arm
died at 12:20 with the allocator assert that this box's broken NVML reports in place of an OOM.
**Both MoE anchors run at batch `8`**; the other ten stay at the default `32`.

This is **not** the situation caution (u) is about. That caution is one model measured twice and
compared across runs, where a different batch size silently gives a different draw of the same
quantity. Here each anchor is a different model, every band is a ratio **inside** one arm against
its own `n=1` row, and no number is compared across arms except `A(64)`, which is itself such a
ratio. The departure is declared before the run and is reported in the scoring log.

## Bands

**N1 and N2 are feat-111's, unchanged**, now read over twelve anchors instead of five: `rate(n) <=
n * rate(1)` at both events and every `n` (HOLDS / VIOLATED, a violation being an implementation
bug), and `A(64) = rate(64)/rate(1)` on `E_08` (SATURATES if `<= 4` at every anchor, GROWS
otherwise). **The prediction stays SATURATES.**

**N3 -- does an adversarial selector amplify a contaminated anchor more, or less, the more
contaminated it is?** Spearman of `A(64)` against `rate(1)` on `E_08`, over every anchor with
`rate(1) > 0`.

| reading | band |
|---|---|
| AMPLIFIES LESS | `\rho <= -0.6` --- a badly contaminated anchor leaves selection less to add |
| NO TREND | `-0.6 < \rho < +0.6` |
| AMPLIFIES MORE | `\rho >= +0.6` |

**We predict AMPLIFIES LESS**, committed so the arm can refute it: at a high base rate the anchor
already emits the passage on most draws and there is nothing for a selector to find, while at a low
base rate the occasional lucky draw is exactly what an adversarial scorer exists to pick out. If
the prediction is wrong, the mechanism is more dangerous at the contaminated end than we expect.

**N3 is confounded and the paper must say so.** Twelve models differ in family, size and tokenizer
as well as contamination, exactly as C2's six did. What is *not* confounded is N2, which is a ratio
computed inside each anchor; N2 is primary and N3 is read beside it.

**N4 -- the manuscript consequence**, on top of feat-111's, fixed now:

- Under **AMPLIFIES LESS**: Limitations states the shape --- selection adds most where the anchor
  leaks least, so the danger from a mildly contaminated anchor is larger \emph{relative} to its own
  rate than from a badly contaminated one, and the absolute rate is what a deployer must bound.
- Under **NO TREND**: the coefficient is reported and no shape is claimed, as C2 was.
- Under **AMPLIFIES MORE**: this goes in the **main text**. It would mean the mechanism is worst
  exactly where the premise fails worst, which qualifies the constructive claim and must sit beside
  it.

**N5 -- what may not be claimed**, extending feat-111's. Four of the twelve
(`llama32-1b`, `llama32-3b`, `phi35mini`, `qwen25-7b`) are instruction-tuned or
undisclosed-corpus models and are **not** legitimate safe models; they appear here only as
contamination levels, and no certificate, leakage, `s(x)` or utility claim about a deployable
anchor comes from any of them.

## Excluded alternatives

- Adding a thirteenth anchor after reading any of these twelve.
- Dropping an anchor whose `n=1` rate is inconvenient, or whose architecture forced batch `8`.
- Reading N3 on `E_001` if `E_08` is unfavourable, or on the mean recall, which Proposition 4 does
  not bound. All are reported.
- Treating N3 as causal evidence about contamination when family, size and tokenizer vary with it.
- Re-reading feat-111's N1/N2 on five anchors and these on twelve, whichever is more favourable.
  The twelve-anchor reading supersedes and both are printed by the same script.

## Scoring log
