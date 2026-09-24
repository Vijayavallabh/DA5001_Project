# Post-hoc note: served lengths, a length-controlled head-to-head, and the non-empty restriction

**Post hoc, 2026-09-24. No band was registered for anything here**, so it is a note and never an
`onset_prediction_*.md` (caution (as)). Nothing new is judged: it reads the de-echoed headline pass
(feat-184 Part A, `results/order_averaged_h2h_per_prompt_deecho.csv`) and the texts that pass judged.

Command: `.venv/bin/python analysis/h2h_length_control.py --out results` -> `results/h2h_length_control.csv`.

**Served lengths (words, median [IQR], 500 prompts).** Selection `n=64`: `75 [33, 136]`, empty on
`41` prompts. Its `n=1` draw: `54 [13, 126]`, empty on `64`. The `k=10` meter: `151 [142, 160]`,
empty on none. Its anchor control: `63 [16, 134]`, empty on `62`. The opponent: `154 [144, 162]`.
The meter writes about twice selection's length.

**Length control** (after length-controlled AlpacaEval, Dubois et al. 2024): regress each
order-averaged score on arm indicators plus `beta * tanh(dlen / 62.7)`, `dlen` the arm's words minus
the opponent's, and recompute the difference of gains from the four arm coefficients; prompts
resampled and refitted, 2,000 times. `D3` as judged `+0.0505 [+0.0165, +0.0855]` (it reproduces the
pass); length-controlled `+0.0605 [+0.0236, +0.0971]`; `beta = +0.0179 [-0.0085, +0.0450]`. The judge
shows no length preference this regression can resolve, and at equal length the difference is, if
anything, larger: the meter's longer answers are not what selection is judged against.

**Non-empty restriction.** On the `414` prompts where both anchor-drawn controls (`n=1` and `k=0`)
are non-empty, `D3 = +0.0441 [+0.0024, +0.0864]`. Part of selection's gain is replacing an empty
draw with a non-empty one; what remains on the non-empty prompts still excludes zero, narrowly.
