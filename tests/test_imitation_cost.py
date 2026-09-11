"""Proposition 3's empirical signature, checked against the CSVs it is quoted from.

The proposition says a bucket-metered decoder's realised spend is the cost of imitating p_r, not of
buying utility, so it must saturate once the cap stops binding and then stop tracking the cap. That
is a falsifiable prediction about data this repo already has, and it is the only evidence in the
paper that the proposition describes the deployed decoder rather than an idealisation.

results/utility_price.csv carries the arm means the body quotes; results/imitation_cost.csv carries
the per-step measurement of the proposition's two factors. The two are the same trajectories, which
is why the spend columns must agree -- they did not while one reported medians and the other means,
and a reader would have met 4.2% and 4.3% for one quantity."""
import csv
import re

from tests.manuscript import tex

ROWS = {r["k"]: r for r in csv.DictReader(open("results/utility_price.csv"))}
IMIT = {(r["prompt_class"], r["k"]): r
        for r in csv.DictReader(open("results/imitation_cost.csv"))}
TEX = tex("sections/frontier.tex")
APX = tex("sections/appendix_proofs.tex")


def test_the_spend_saturates_while_the_cap_keeps_doubling():
    """k = 10 -> 20 doubles the cap and moves the realised spend by under a tenth of a nat."""
    lo, hi = ROWS["10.0"], ROWS["20.0"]
    assert float(hi["budget_K"]) == 2 * float(lo["budget_K"])
    assert abs(float(hi["mean_spend_nats"]) - float(lo["mean_spend_nats"])) < 0.1
    # and it is monotone up to there, i.e. the cap binds at small k and stops binding
    ks = sorted(ROWS, key=float)
    spend = [float(ROWS[k]["mean_spend_nats"]) for k in ks]
    assert spend[0] < spend[-1], spend
    assert max(spend) - spend[-1] < 0.1, "the largest budget should be at the saturated spend"


def test_the_manuscript_quotes_the_saturated_spend_and_the_unused_allowance():
    body = open(TEX, encoding="utf-8").read().replace("\n", " ")
    sat = float(ROWS["20.0"]["mean_spend_nats"])
    frac = 100 * sat / float(ROWS["20.0"]["budget_K"])
    m = re.search(r"stalls at \$([\d.]+)\$ nats", body)
    assert m and float(m.group(1)) == round(sat, 2), (m.group(1) if m else None, sat)
    m = re.search(r"\$([\d.]+)\\%\$ of the allowance", body)
    assert m and abs(float(m.group(1)) - frac) < 0.05, (m.group(1) if m else None, frac)


def test_the_two_csvs_describe_the_same_arm():
    """The per-step scan must reproduce utility_price's spend, or the appendix table and the body
    sentence are summarising different trajectories at the same k."""
    for k in ("3", "5", "10", "20"):
        a = float(ROWS[f"{float(k)}"]["mean_spend_nats"])
        b = float(IMIT[("ordinary", k)]["spend_nats"])
        assert abs(a - b) < 0.05, (k, a, b)
        assert int(IMIT[("ordinary", k)]["n_trajectories"]) == int(ROWS[f"{float(k)}"]["n_trajectories"])


def test_the_binding_fraction_the_proposition_leans_on_is_measured():
    """beta is measured, not assumed, and the body quotes it at the two ends and at k=3."""
    body = open(TEX, encoding="utf-8").read().replace("\n", " ")
    m = re.search(r"\$\\beta\$ falls from \$([\d.]+)\$ at \$k=([\d.]+)\$ to \$([\d.]+)\$ at\s*"
                  r"\$k=(\d+)\$ and \$([\d.]+)\$ at \$k=(\d+)\$", body)
    assert m, "the beta sentence has moved"
    lo_b, lo_k, mid_b, mid_k, hi_b, hi_k = m.groups()
    for b, k in ((lo_b, lo_k), (mid_b, mid_k), (hi_b, hi_k)):
        want = float(IMIT[("ordinary", k.rstrip("."))]["beta_binding_frac"])
        assert abs(float(b) - want) < 5e-4, (k, b, want)
    assert float(lo_b) > float(mid_b) > float(hi_b), m.groups()


def test_the_saturated_imitation_rate_in_the_body_comes_from_the_per_step_scan():
    body = open(TEX, encoding="utf-8").read().replace("\n", " ")
    m = re.search(r"saturates at \$([\d.]+)\$ nats per token", body)
    rate = float(IMIT[("ordinary", "20")]["imitation_rate_nats_per_token"])
    assert m and abs(float(m.group(1)) - rate) < 5e-4, (m.group(1) if m else None, rate)
    # saturation means the last two budgets agree while the cap doubles
    prev = float(IMIT[("ordinary", "10")]["imitation_rate_nats_per_token"])
    assert abs(rate - prev) < 0.01, (prev, rate)


def test_the_appendix_table_rounds_from_the_csv_once():
    apx = open(APX, encoding="utf-8").read()
    for k in ("0.1", "0.5", "3", "20"):
        r = IMIT[("ordinary", k)]
        row = "${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ \\\\".format(
            k, int(float(r["budget_K"])),
            f'{float(r["beta_binding_frac"]):.4f}',
            f'{float(r["imitation_rate_nats_per_token"]):.3f}',
            f'{float(r["realised_rate_nats_per_token"]):.3f}',
            f'{float(r["spend_nats"]):.1f}',
            f'{100 * float(r["spend_over_cap"]):.1f}\\%',
            f'{float(r["median_cum_spend_vs_step_r2"]):.4f}')
        assert row in apx, row


def test_the_interior_blend_fraction_is_distinguished_from_beta():
    """beta counts the prefix-debt opening too. Conflating the two would put 0.4% where 3.4%
    belongs, which is the mistake the body sentence was rewritten to avoid."""
    r = IMIT[("ordinary", "3")]
    beta, active = float(r["beta_binding_frac"]), float(r["active_frac"])
    assert beta > 5 * active, (beta, active)
    assert abs(beta - (active + float(r["forced_safe_frac"]))) < 1e-6
    apx = open(APX, encoding="utf-8").read().replace("\n", " ")
    m = re.search(r"strictly interior blends alone\s+are \$([\d.]+)\\%\$ of steps at \$k=3\$", apx)
    assert m and abs(float(m.group(1)) - 100 * active) < 0.05, (m.group(1) if m else None, active)


def test_the_protected_passages_are_charged_a_higher_rate_than_ordinary_prompts():
    """The claim in the appendix: the meter charges most where the risky model is most distinctive
    from the anchor. If this ever inverted, the sentence would have to go."""
    o = float(IMIT[("ordinary", "20")]["imitation_rate_nats_per_token"])
    p = float(IMIT[("protected", "20")]["imitation_rate_nats_per_token"])
    assert p > o, (o, p)
    apx = open(APX, encoding="utf-8").read().replace("\n", " ")
    m = re.search(r"same rate is higher, \$([\d.]+)\$ nats per token", apx)
    assert m and abs(float(m.group(1)) - p) < 5e-4, (m.group(1) if m else None, p)


def test_the_running_spend_is_linear_in_the_step_index_at_every_budget():
    for (cls, k), r in IMIT.items():
        assert float(r["median_cum_spend_vs_step_r2"]) > 0.9, (cls, k, r)
    body = open(TEX, encoding="utf-8").read().replace("\n", " ")
    m = re.search(r"linear in the step index at every budget \(median \$R\^2 \\ge ([\d.]+)\$\)", body)
    assert m, "the linearity claim has moved"
    # the sentence is about the ordinary-prompt arms it quotes beta and the rate from; the
    # protected arms are weaker at the smallest budget (0.93) and are reported separately.
    worst = min(float(r["median_cum_spend_vs_step_r2"])
                for (cls, _), r in IMIT.items() if cls == "ordinary")
    assert round(worst, 2) >= float(m.group(1)), (worst, m.group(1))


def test_the_sparsity_proposition_is_stated_and_its_one_number_is_measured():
    """Proposition 5 narrows what the paper used to call open: a causal policy on a budget that
    does not grow with the work MUST be the anchor almost everywhere. Its only empirical claim is
    the contrast -- the deployed rule serves p_r unchanged at 99.95% of steps at k=20."""
    apx = open(APX, encoding="utf-8").read()
    assert r"\label{prop:sparse}" in apx, "the proposition has moved"
    assert r"\mathbb{E}_q[N_\varepsilon] \le K/\varepsilon" in apx, "the bound has changed"
    slack = 100 * (1 - float(IMIT[("ordinary", "20")]["beta_binding_frac"]))
    m = re.search(r"serves \$p_\{r,t\}\$ unchanged at \$([\d.]+)\\%\$ of steps", apx.replace("\n", " "))
    assert m and abs(float(m.group(1)) - slack) < 0.005, (m.group(1) if m else None, slack)


def test_the_limitations_no_longer_call_the_shape_question_open():
    """It was 'could in principle concentrate its spend'; Proposition 5 makes that a requirement,
    and only the quantitative half stays open. If the proposition were ever removed, this sentence
    would be an overclaim."""
    closing = open(tex("sections/iclr_closing.tex"), encoding="utf-8").read().replace("\n", " ")
    assert r"\ref{prop:sparse}" in closing, "the limitation no longer cites the proposition"
    assert "could in principle concentrate" not in closing, "the weaker claim is back"
    assert "quantitative half" in closing, "the limitation must say what is still open"


def test_the_spend_concentration_beside_the_sparsity_proposition_rounds_from_the_csv():
    """Proposition 5 needs a policy on an O(1) budget to put its spend on O(1) steps. The deployed
    rule is at the other extreme, and the appendix quotes two numbers for how far."""
    r = IMIT[("ordinary", "20")]
    top1 = 100 * float(r["top1pct_of_steps_share_of_spend"])
    cov = 100 * float(r["frac_of_steps_for_90pct_of_spend"])
    apx = open(APX, encoding="utf-8").read().replace("\n", " ")
    m = re.search(r"busiest \$1\\%\$ of steps carry \$([\d.]+)\\%\$", apx)
    assert m and abs(float(m.group(1)) - top1) < 0.05, (m.group(1) if m else None, top1)
    m = re.search(r"covering \$90\\%\$ of it takes \$(\d+)\\%\$ of the sequence", apx)
    assert m and abs(float(m.group(1)) - cov) < 0.5, (m.group(1) if m else None, cov)
    # the claim is that it is NOT sparse: most of the sequence is needed to cover most of the spend
    assert cov > 50, cov


def test_the_appendix_figure_exists_and_its_caption_numbers_come_from_the_csv():
    """A missing \\includegraphics halts tectonic and leaves the PREVIOUS pdf in place, which then
    measures as if nothing were wrong (AGENTS caution (f)). Check the file, not just the caption."""
    import os
    from tests.manuscript import tex
    fig = os.path.join(os.path.dirname(tex("iclr_2027.tex")), "figures", "imitation_cost.pdf")
    assert os.path.exists(fig), fig
    apx = open(APX, encoding="utf-8").read().replace("\n", " ")
    assert r"\label{fig:imitation}" in apx
    sat = float(IMIT[("ordinary", "20")]["imitation_rate_nats_per_token"])
    m = re.search(r"imitation rate \$([\d.]+)\$ nats per token", apx)
    assert m and abs(float(m.group(1)) - sat) < 5e-4, (m.group(1) if m else None, sat)


def test_the_lorenz_curves_end_at_one_and_lie_above_the_diagonal():
    """The claim the figure makes with them: the spend is spread over the sequence, barely above
    uniform, which is the opposite of what Proposition 5 needs."""
    import csv as _csv
    from collections import defaultdict
    curves = defaultdict(list)
    for r in _csv.DictReader(open("results/imitation_lorenz.csv")):
        curves[(r["prompt_class"], r["k"])].append(
            (float(r["frac_of_steps"]), float(r["frac_of_spend"])))
    assert curves
    for key, v in curves.items():
        v.sort()
        assert abs(v[-1][1] - 1.0) < 1e-6, (key, v[-1])
        assert all(y >= x - 1e-9 for x, y in v), key          # above the uniform diagonal
        assert all(b >= a - 1e-9 for (_, a), (_, b) in zip(v, v[1:])), key
    half = dict(curves[("ordinary", "20")])[0.5]
    assert 0.5 < half < 0.9, half     # spread, not concentrated: half the steps carry most of it
