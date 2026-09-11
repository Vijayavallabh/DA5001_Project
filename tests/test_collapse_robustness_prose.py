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
