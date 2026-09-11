"""The second-anchor appendix opens with a count of budgets, strategies and budgeted queries. The
query count reproduced from no subset of results/composition_comma7b.csv -- plan v5 recorded that as
an open erratum on 2026-09-08 and it was still in the manuscript on 2026-09-11. Counts are the
easiest number to write from memory and the easiest to check."""
import csv
import re

from tests.manuscript import tex

CSV = "results/composition_comma7b.csv"
TEX = tex("sections/appendix_second_anchor.tex")


def test_the_opening_counts_match_the_run():
    rows = list(csv.DictReader(open(CSV)))
    budgets = sorted({float(r["k"]) for r in rows if float(r["k"]) > 0})
    baselines = sorted({float(r["k"]) for r in rows if float(r["k"]) <= 0})
    strategies = {(r["mode"], r["L"]) for r in rows}
    queries = round(sum(float(r["n_queries"]) for r in rows if float(r["k"]) > 0))
    body = open(TEX, encoding="utf-8").read().replace("\n", " ")
    m = re.search(r"Across \$(\d+)\$ budgets, their (\w+) baselines, \$(\d+)\$ strategies and "
                  r"\$([\d,{}]+)\$ budgeted queries", body)
    assert m, "the opening sentence has moved"
    assert int(m.group(1)) == len(budgets), (m.group(1), budgets)
    assert m.group(2) == ["zero", "one", "two"][len(baselines)], (m.group(2), baselines)
    assert int(m.group(3)) == len(strategies), (m.group(3), strategies)
    assert int(re.sub(r"[^0-9]", "", m.group(4))) == queries, (m.group(4), queries)


def test_there_really_are_no_invariant_violations_to_report():
    rows = list(csv.DictReader(open(CSV)))
    assert sum(float(r["invariant_violations"]) for r in rows) == 0
