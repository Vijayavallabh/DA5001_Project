"""Four numbers in Appendix C are prose, not a table, and prose is where stale values survive:
`analysis/audit_numbers.py` only asks whether a literal appears in SOME CSV, and 0.165 and 0.173
did -- they had been the metric-block values two pair-counts earlier and stayed in the manuscript
after the CSV moved to 0.106 and 0.109. Pin them against the block they claim to come from."""
import csv
import os
import re

from tests.manuscript import tex

CSV = "results/collapse_robustness.csv"
TEX = tex("sections/appendix_robustness.tex")


def _blocks():
    out = {}
    for r in csv.DictReader(open(CSV)):
        out[(r["block"], r["setting"])] = r
    return out


def test_the_metric_sentence_matches_the_metric_block():
    b = _blocks()
    body = open(TEX, encoding="utf-8").read()
    lcs = float(b[("metric", "lcs_word")]["value_relative_to_metric_range"])
    nv = float(b[("metric", "nv_recall")]["value_relative_to_metric_range"])
    m = re.search(r"the spread is \$([\d.]+)\$ of that metric's range against\s*\n?\$([\d.]+)\$ "
                  r"for the thresholded recall", body)
    assert m, "the metric sentence has moved"
    assert (float(m.group(1)), float(m.group(2))) == (round(lcs, 3), round(nv, 3)), m.groups()


def test_the_threshold_sentence_matches_the_threshold_block():
    b = _blocks()
    body = open(TEX, encoding="utf-8").read()
    spreads = [round(float(b[("threshold", f"onset at {t}")]["value"]), 3)
               for t in ("0.005", "0.01", "0.02")]
    levels = [float(b[("threshold", f"onset at {t}")]["value_relative_to_metric_range"])
              for t in ("0.005", "0.01", "0.02")]
    m = re.search(r"is \$([\d.]+)\$, \$([\d.]+)\$ and \$([\d.]+)\$ at\s*\n?thresholds", body)
    assert m and [float(x) for x in m.groups()] == spreads, (m.groups() if m else None, spreads)
    m = re.search(r"the absolute level runs from \$([\d.]+)\$ to \$([\d.]+)\$", body)
    assert m, "the level sentence has moved"
    assert [float(x) for x in m.groups()] == [round(min(levels), 3), round(max(levels), 3)], m.groups()


def test_the_k_crit_prediction_sentence_is_a_mean_and_says_so():
    """It read 'predicts every other arm to within 5.1%', which claims a bound. 5.1% is the MEAN
    over the five seed arms; the largest single error is 10.4%. What IS true of every arm is that
    the prediction falls inside its measured interval, which the CSV records per row."""
    import statistics as st
    rows = [r for r in csv.DictReader(open("results/seed_effect_summary.csv"))
            if not r["pair"].endswith(" tau")]
    err = [abs(float(r["predicted_ratio_K"]) - float(r["measured_ratio"])) / float(r["measured_ratio"])
           for r in rows]
    assert all(r["prediction_in_ci"] == "True" for r in rows), \
        [r["arm"] for r in rows if r["prediction_in_ci"] != "True"]
    body = open(tex("sections/onset.tex"), encoding="utf-8").read()
    assert "to within $5.1\\%$" not in body, "the bound claim is back"
    m = re.search(r"at a mean error of \$([\d.]+)\\%\$", body)
    assert m and abs(float(m.group(1)) - 100 * st.mean(err)) < 0.05, (m.group(1) if m else None,
                                                                     100 * st.mean(err))
    assert max(err) > float(m.group(1)) / 100, "if the max were below it, 'within' would be fine"


def test_the_top_of_the_utility_scale_is_priced_from_the_same_law_as_the_rest():
    """Section 3 prices winning every judged comparison. It read 1.19 nats and 144x, which come
    from results/utility_v4_summary.csv (180 judged pairs per arm, anchor win 30.5%); every other
    number in the same paragraph comes from v5 (600 pairs, 27.3%), where the same quantity is 1.30
    and 132x. AGENTS.md caution (d) is about exactly that 180-against-600 difference."""
    rows = {r["k"]: r for r in csv.DictReader(open("results/utility_price.csv"))}
    body = open(tex("sections/frontier.tex"), encoding="utf-8").read()
    lam = float(rows["3.0"]["lambda_star_u_max"])           # a property of the safe law, same on every row
    assert len({r["lambda_star_u_max"] for r in rows.values()}) == 1
    m = re.search(r"would cost an optimal policy \$([\d.]+)\$ nats,\s*\n?and the decoder spends "
                  r"\$(\d+)\$ times that", body)
    assert m, "the ceiling sentence has moved"
    assert float(m.group(1)) == round(lam, 2), (m.group(1), lam)
    best = max(float(r["spend_over_lambda_star_u_max"]) for r in rows.values())
    assert float(m.group(2)) == round(best), (m.group(2), best)


def test_the_two_judge_sigmas_in_the_introduction_come_from_the_v6_separation_csvs():
    """Caution (e): a judged sigma is fragile and must be quoted with its budget. The intro quotes
    four, two judges at two budgets; at k = 0.5 they are -1.03 and +1.52 and at k = 10 they are
    -6.97 and -3.29. The abstract makes the same claim at k = 0.6, where the values are different
    numbers with the same reading, so both budgets are checked."""
    def z(path, k):
        for r in csv.DictReader(open(path)):
            if r["decoder"] == "KL" and float(r["k"]) == k:
                return float(r["z_loss_vs_anchor"])
        raise AssertionError((path, k))
    q, p = "results/judge_separation_v6.csv", "results/judge_separation_v6_judge2.csv"
    body = open(tex("sections/iclr_intro.tex"), encoding="utf-8").read().replace("\n", " ")
    for k, pair in ((0.5, (z(q, 0.5), z(p, 0.5))), (10.0, (z(q, 10.0), z(p, 10.0)))):
        a, b = pair
        assert f"(${a:+.2f}\\sigma$, ${b:+.2f}\\sigma$)".replace("+-", "-") in body \
            or f"(${a:.2f}\\sigma$, ${b:+.2f}\\sigma$)" in body, (k, a, b)
    # the abstract's budget is 0.6, where neither judge separates: |z| < 2 for both
    assert abs(z(q, 0.6)) < 2 and abs(z(p, 0.6)) < 2
    assert "at $k=0.6$" in open(tex("iclr_2027.tex"), encoding="utf-8").read()
