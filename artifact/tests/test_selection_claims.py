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
