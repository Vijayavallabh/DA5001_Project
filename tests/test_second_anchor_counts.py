"""The second-anchor appendix opens with a count of budgets, strategies and budgeted queries. The
query count reproduced from no subset of results/composition_comma7b.csv -- plan v5 recorded that as
an open erratum on 2026-09-08 and it was still in the manuscript on 2026-09-11. Counts are the
easiest number to write from memory and the easiest to check."""
import csv
import re

from tests.manuscript import tex

CSV = "results/composition_comma7b.csv"
# appendix_second_anchor.tex was retired in v10 (2026-09-24); its paragraph is now "A pair where
# memorisation is the only variable" in Appendix G, appendix_onset.tex.
TEX = tex("sections/appendix_onset.tex")


def test_the_opening_counts_match_the_run():
    rows = list(csv.DictReader(open(CSV)))
    budgets = sorted({float(r["k"]) for r in rows if float(r["k"]) > 0})
    baselines = sorted({float(r["k"]) for r in rows if float(r["k"]) <= 0})
    strategies = {(r["mode"], r["L"]) for r in rows}
    queries = round(sum(float(r["n_queries"]) for r in rows if float(r["k"]) > 0))
    body = " ".join(open(TEX, encoding="utf-8").read().split())
    # v10 wording: "Across $7$ budgets, both baselines, $5$ strategies and ..." ("both" is two).
    m = re.search(r"Across \$(?P<b>\d+)\$ budgets, (?:their (?P<base>\w+)|(?P<both>both)) baselines, "
                  r"\$(?P<s>\d+)\$ strategies and \$(?P<q>[\d,{}]+)\$ budgeted queries", body)
    assert m, "the opening sentence has moved"
    assert int(m["b"]) == len(budgets), (m["b"], budgets)
    base = "two" if m["both"] else m["base"]
    assert base == ["zero", "one", "two"][len(baselines)], (base, baselines)
    assert int(m["s"]) == len(strategies), (m["s"], strategies)
    assert int(re.sub(r"[^0-9]", "", m["q"])) == queries, (m["q"], queries)


def test_there_really_are_no_invariant_violations_to_report():
    rows = list(csv.DictReader(open(CSV)))
    assert sum(float(r["invariant_violations"]) for r in rows) == 0
