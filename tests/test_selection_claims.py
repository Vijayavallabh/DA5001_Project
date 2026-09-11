"""The numbers Sections 5 and 6 rest on, checked against the CSVs that produced them.

The selection sections are new in v6 and carry the paper's constructive claim, so every load-bearing
literal in them is pinned here rather than read by eye. The odometer arithmetic is included because
it is the one place the paper does a division in prose."""
import csv
import math
import re

from tests.manuscript import tex

SEL = tex("sections/selection.tex")
EXP = tex("sections/experiments.tex")
APP = tex("sections/appendix_selection.tex")


def _rows(path):
    return list(csv.DictReader(open(path)))


def test_the_median_protected_target_is_quoted_from_the_odometer_csv():
    """849 is S_total_median in results/odometer.csv. It was quoted as 850 in four places until
    2026-09-11, which is a second rounding of an integer the CSV already stores exactly."""
    vals = {r["S_total_median"] for r in _rows("results/odometer.csv")}
    assert len(vals) == 1, vals
    s_tot = float(vals.pop())
    body = "".join(open(f, encoding="utf-8").read() for f in (SEL, EXP, APP,
                                                              tex("iclr_2027.tex"),
                                                              tex("sections/frontier.tex"),
                                                              tex("sections/iclr_closing.tex")))
    assert f"${s_tot:.0f}$" in body, f"the paper no longer quotes S(x) = {s_tot:.0f}"
    assert "$850$" not in body, "the rounded 850 is back"


def test_the_selection_budget_arithmetic_is_exact():
    """log n - (n-1)/n at n = 8, and log 8 for the pathwise budget. Both are quoted to 2 or 3 dp."""
    kl8 = math.log(8) - 7 / 8
    assert f"${kl8:.3f}$" == "$1.204$"
    body = open(SEL, encoding="utf-8").read() + open(EXP, encoding="utf-8").read()
    assert "$1.204$" in body or "$1.20$" in body
    assert f"$\\log 8 = {math.log(8):.2f}$" in open(SEL, encoding="utf-8").read()


def test_the_composition_count_divides_out():
    """332 queries at n = 8 and 2 at k = 3, both from a 400-nat odometer over the measured spend."""
    spend = {r["k"]: float(r["mean_spend_nats"]) for r in _rows("results/utility_price.csv")}
    kl8 = math.log(8) - 7 / 8
    assert int(400 / kl8) == 332, 400 / kl8
    assert int(400 / spend["3.0"]) == 2, 400 / spend["3.0"]
    body = open(SEL, encoding="utf-8").read().replace("\n", " ")
    assert "$332$ queries at $n=8$ against $2$" in body, body[body.find("composes"):][:220]
    m = re.search(r"spends a measured \$([\d.]+)\$ nats", body)
    assert m and float(m.group(1)) == spend["3.0"], (m.group(1) if m else None, spend["3.0"])


def test_the_cross_judge_gain_and_its_interval_come_from_the_csv():
    rows = _rows("results/selection_crossjudge.csv")
    g = {r["gain"] for r in rows}
    lo = {r["gain_lo95"] for r in rows}
    hi = {r["gain_hi95"] for r in rows}
    assert len(g) == len(lo) == len(hi) == 1
    body = "".join(open(f, encoding="utf-8").read() for f in (EXP, tex("sections/iclr_closing.tex")))
    assert f"$+{float(g.pop()):.3f}$" in body
    assert f"$[+{float(lo.pop()):.3f}, +{float(hi.pop()):.3f}]$" in body


def test_extraction_is_zero_at_every_n_in_the_table():
    """The safety claim. Every n up to 64 must read 0.0000, and the risky baseline must not."""
    rows = {r["n"]: r for r in _rows("results/selection_extraction.csv")}
    for n in ("1", "2", "4", "8", "16", "32", "64"):
        assert float(rows[n]["nv_recall_mean"]) == 0.0, n
        assert float(rows[n]["nv_recall_max"]) == 0.0, n
    assert float(rows["-1"]["nv_recall_mean"]) > 0.4
    body = open(EXP, encoding="utf-8").read()
    assert f"${float(rows['-1']['nv_recall_mean']):.4f}$" in body
    assert f"${float(rows['-1']['nv_recall_max']):.4f}$" in body


def test_the_n_sweep_rows_in_the_table_round_from_selection_scaling_csv():
    """feat-088's arms reach the main table; a stale row here would be a claim about a run that
    never happened. Checked mechanically against the CSV, not by eye (caution (j))."""
    import csv as _csv
    from tests.manuscript import tex as _tex
    rows = list(_csv.DictReader(open("results/selection_scaling.csv")))
    body = open(_tex("sections/experiments.tex"), encoding="utf-8").read()
    want = [("Phi-3.5-mini-instruct", 8, "judge B"), ("Phi-3.5-mini-instruct", 64, "judge B"),
            ("Meta-Llama-3.1-8B-Instruct", 64, "judge C")]
    for judge, n, label in want:
        r = next(x for x in rows if judge in x["judge"] and int(float(x["n"])) == n)
        cell = (f"pointwise reward            & {label} & ${n}$ & "
                f"${float(r['kl_nats']):.3f}$ & ${float(r['mean_words']):.1f}$ & "
                f"${float(r['u']):.3f}$ & $[{float(r['u_lo95']):.3f}, {float(r['u_hi95']):.3f}]$")
        assert cell in body, cell


def test_the_reversal_claim_is_true_of_the_csvs_it_cites():
    """The paper says the comparison is 'not a tie but a reversal'. That is only allowed while
    selection's u at n=64 actually exceeds the metered decoder's, at a far smaller budget."""
    import csv as _csv
    import re as _re
    from tests.manuscript import tex as _tex
    sel = next(r for r in _csv.DictReader(open("results/selection_scaling.csv"))
               if "Phi-3.5" in r["judge"] and int(float(r["n"])) == 64)
    dec = next(r for r in _csv.DictReader(open("results/selection_crossjudge.csv"))
               if "metered" in r["selector"])
    assert float(sel["u"]) > float(dec["u"]), (sel["u"], dec["u"])
    ratio = float(dec["kl_nats"]) / float(sel["kl_nats"])
    body = open(_tex("sections/experiments.tex"), encoding="utf-8").read().replace("\n", " ")
    m = _re.search(r"selection reaches \$([\d.]+)\$ for \$([\d.]+)\$", body)
    assert m, "the reversal sentence has moved"
    assert float(m.group(1)) == round(float(sel["u"]), 3), (m.group(1), sel["u"])
    assert abs(float(m.group(2)) - float(sel["kl_nats"])) < 0.005, (m.group(2), sel["kl_nats"])
    assert "fifty-fourth" in body and 53.0 < ratio < 55.0, ratio


def test_the_sweep_is_monotone_in_log_n_on_both_judges():
    """O1 read SCALES. If a rerun ever made it non-monotone the paragraph would be wrong, and the
    Spearman the paper quotes is the thing to check."""
    import csv as _csv
    rows = list(_csv.DictReader(open("results/selection_scaling.csv")))
    for judge in {r["judge"] for r in rows}:
        arms = sorted((int(float(r["n"])), float(r["u"])) for r in rows if r["judge"] == judge)
        assert arms[-1][1] > arms[0][1], (judge, arms)
        assert float(next(r for r in rows if r["judge"] == judge)["spearman_u_logn"]) > 0.95, judge
    b = {int(float(r["n"])): r for r in rows if "Phi-3.5" in r["judge"]}
    gain8 = float(b[8]["u"]) - float(b[1]["u"])
    gain64 = float(b[64]["u"]) - float(b[1]["u"])
    assert gain64 >= gain8 + 0.05, (gain8, gain64)      # the committed SCALES band
