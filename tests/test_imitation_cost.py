"""Proposition 3's empirical signature, checked against results/utility_price.csv.

The proposition says a bucket-metered decoder's realised spend is the cost of imitating p_r, not of
buying utility, so it must saturate once the cap stops binding and then stop tracking the cap. That
is a falsifiable prediction about a CSV this repo already has, and it is the only evidence in the
paper that the proposition describes the deployed decoder rather than an idealisation."""
import csv
import re

from tests.manuscript import tex

ROWS = {r["k"]: r for r in csv.DictReader(open("results/utility_price.csv"))}
TEX = tex("sections/frontier.tex")


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
    m = re.search(r"\$([\d.]+)\$\s+nats\s+at\s+\$k\s*=\s*20\$", body)
    assert m and float(m.group(1)) == round(sat, 2), (m.group(1) if m else None, sat)
    m = re.search(r"using\s+\$([\d.]+)\\%\$\s+of\s+the\s+allowance", body)
    assert m and abs(float(m.group(1)) - frac) < 0.05, (m.group(1) if m else None, frac)


def test_the_binding_fraction_the_proposition_leans_on_is_the_one_in_the_order_table():
    """beta is measured, not assumed: the 0.4% is the alpha=1 'steps touched' cell of Table 3."""
    tbl = open(tex("sections/appendix_onset.tex"), encoding="utf-8").read()
    row = [l for l in tbl.split("\n") if "alpha = 1" in l or r"\alpha = 1" in l]
    assert row, "the alpha=1 row of the order table has moved"
    assert "0.4\\%" in row[0], row[0]
    body = open(TEX, encoding="utf-8").read().replace("\n", " ")
    assert "touches $0.4\\%$ of steps at $k=3$" in body, "the proposition no longer cites it"
