# Post-hoc note: the realised divergence of majority vote

**Post hoc, 2026-09-24, descriptive, no band.** Proposition 1's `log n` is a bound. For majority
vote the served law can be computed rather than bounded, because the rule reads only the extracted
answers: given its answer class the served string is distributed as `p_s(. | class)`, so
`q(y)/p_s(y) = q_ans(a(y))/pi(a(y))`, `D_inf(q||p_s)` is the largest of those ratios and
`D_KL(q||p_s) = KL(q_ans||pi)`, both exactly. `pi` is the plug-in over the `64` cached Comma-7B draws
per question, and `q_ans` is simulated (`20,000` runs) under the committed rule.

Command: `.venv/bin/python analysis/majority_realised_divergence.py --out results` ->
`results/majority_realised_divergence.csv`.

| task | n | `log n` | KL bound | `D_inf` median / p90 / max | KL median / p90 |
|---|---|---|---|---|---|
| GSM8K | 8 | 2.079 | 1.204 | 0.506 / 0.712 / 0.841 | 0.188 / 0.330 |
| GSM8K | 64 | 4.159 | 3.175 | 0.924 / 1.547 / 1.967 | 0.734 / 1.203 |
| TriviaQA | 8 | 2.079 | 1.204 | 0.475 / 0.748 / 1.633 | 0.142 / 0.346 |
| TriviaQA | 64 | 4.159 | 3.175 | 1.022 / 1.701 / 2.426 | 0.678 / 1.282 |

At `n=64` the realised pathwise divergence is about one nat at the median and under `2.5` at the
worst of `500` questions, against a certificate of `4.16`: the rule concentrates on the modal class,
whose ratio is at most `1/pi(modal)`, so the realised figure is set by how often the anchor already
agrees with itself. The plug-in `pi` cannot see classes absent from `64` draws; the maximum is
carried by the modal class, which the plug-in estimates well.
