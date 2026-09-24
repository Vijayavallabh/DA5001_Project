# Note: the breadth entry gate re-measured on de-echoed text (post hoc, 2026-09-24)

**Post hoc, no band.** The registered entry gate for the six-anchor breadth sweep reads the n=1
arm's empty-completion fraction. It was measured on `generation` fields that carried the prompt's
tail (caution (bc)), so an empty draw looked non-empty. Re-measured on the true generations
(`dap.shared.served_generation`) with every judged number unchanged:

`.venv/bin/python analysis/selection_breadth.py --out results --deecho` -> `results/selection_breadth_deecho.csv`

| anchor | empty, as gated | empty, de-echoed | gate | judge-B gain at `n=8` | on non-empty prompts only |
|---|---|---|---|---|---|
| TinyComma-1.8B (audited) | 6.8% | 14.6% | FAIL | +0.054 [+0.013, +0.095] | +0.062 [+0.016, +0.108] (427) |
| Pleias-1.2B | 0.0% | 0.0% | PASS | +0.029 [-0.012, +0.068] | same |
| KL3M-1.7B | 0.2% | 0.2% | PASS | +0.039 [+0.005, +0.076] | +0.039 (499) |
| Comma-7B | 3.0% | 18.4% | **FAIL** | +0.111 [+0.072, +0.148] | +0.103 [+0.063, +0.145] (408) |
| Comma-7B (1T tokens) | 16.6% | 31.6% | FAIL | +0.026 [-0.010, +0.063] | +0.019 [-0.032, +0.072] (342) |
| Pleias-3B | 0.0% | 0.0% | PASS | +0.047 [+0.003, +0.089] | same |

Comma-7B moves from PASS to FAIL; nothing moves the other way. B1 still reads GENERALISES (2 of the
3 admissible new anchors exclude zero on the registered scorer), and at every anchor the gain on
non-empty prompts is within `0.01` of the gain on all prompts except Comma-1T under judge C (`+0.113`
-> `+0.162`), so the gains are not a degeneracy filter. The paper reports the de-echoed gate.
