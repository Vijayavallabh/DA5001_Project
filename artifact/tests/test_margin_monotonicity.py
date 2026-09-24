"""'Rises monotonically with capability' is true per family and FALSE pooled -- so the scope is load-bearing.

Section 4 said "Over ten safe models the margin s(x)/c_use rises monotonically with capability"
while the appendix figure it cites says "rises monotonically with capability IN EVERY FAMILY".
Pooled over all ten models ordered by parameter count the series is 1.94, 3.45, 2.05, 4.14, 2.31,
4.07, 4.19, 2.37, 4.63, 5.02 -- it zig-zags, because the three families sit at different levels and
KL3M's largest model is below Pleias' smallest. Caution (af): a claim correct in one place and
unscoped in another.

This test asserts the shape BOTH ways, so the scope can never be dropped again without a failure:
within family the series must be monotone, pooled it must not be, and the main text must carry the
qualifier.
"""
import csv
import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import body, tex  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rows():
    p = os.path.join(ROOT, "results", "anchor_scaling_summary.csv")
    assert os.path.exists(p), p
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    assert len(rows) == 10, f"the claim says ten safe models; the CSV has {len(rows)}"
    return rows


def _by_family():
    fam = {}
    for r in _rows():
        fam.setdefault(r["corpus"], []).append((float(r["params"]), float(r["margin"]), r["model"]))
    for v in fam.values():
        v.sort()
    return fam


def test_within_every_family_the_margin_is_monotone_in_capability():
    fam = _by_family()
    assert len(fam) == 3, sorted(fam)
    for corpus, series in fam.items():
        m = [x for _, x, _ in series]
        assert m == sorted(m), f"{corpus} is not monotone: {series}"


def test_pooled_over_ten_models_it_is_NOT_monotone_which_is_why_the_scope_matters():
    """If this ever starts passing as monotone, the qualifier could be dropped -- and until then
    dropping it states something the data refutes."""
    pooled = sorted((float(r["params"]), float(r["margin"])) for r in _rows())
    m = [x for _, x in pooled]
    assert m != sorted(m), \
        "pooled monotonicity now holds; the scope is no longer load-bearing, so revisit the claim"


def test_the_main_text_carries_the_family_scope():
    """v10 (2026-09-24) took the margin trend out of the main text; it lives in Appendix G with both
    scopes (next test). Pooled across families the series is NOT monotone (test above), so if any
    body sentence brings the trend back it must carry the within-family scope in the same breath.
    Retired placement, kept property: see tests/RETIRED_2026-09-24.md."""
    txt = body("iclr_intro.tex", "selection.tex", "frontier.tex", "experiments.tex",
               "related_work_v4.tex", "iclr_closing.tex")
    for m in re.finditer(r"(?:margin[^.]{0,80})?rises (?:monotonically )?with capability", txt):
        window = txt[m.start(): m.end() + 60]
        assert "every family" in window, \
            f"an unscoped margin trend is back in the main text: {window!r}"


def test_the_appendix_caption_still_says_the_same_thing():
    """v10 (2026-09-24) removed Figure fig:frontier, whose caption carried this claim, and retired
    appendix_robustness.tex. The claim now lives in Appendix G's 'A better anchor widens the margin'
    paragraph (appendix_onset.tex), which states the scope both ways -- "rises with capability within
    every family" and "Pooled across families it is not monotone" -- and the per-family sign test is
    one row per family of Table tab:marginspan. The sign test's p is derived from the CSV."""
    txt = " ".join(open(tex("sections/appendix_onset.tex"), encoding="utf-8").read().split())
    i = txt.index("\\label{app:scaling}")
    para = txt[i: txt.index("\\end{table}", i)]
    assert "rises with capability within every family" in para, \
        "the appendix lost the scope the main text is checked against"
    assert "Pooled across families it is not monotone" in para, \
        "the appendix no longer says the pooled series is not monotone"
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "anchor_scaling_paired.csv"),
                                    encoding="utf-8")))
    assert len(rows) == 3, "one paired row per family"
    cells = []
    for r in rows:
        mant, exp = f"{float(r['sign_test_p']):.1e}".split("e")
        cells.append(f"${r['novels_margin_up']}/{r['n_novels']}$, $p = {mant}\\times10^{{{int(exp)}}}$")
    for c in set(cells):
        assert para.count(c) == cells.count(c), (c, para.count(c), "the per-family sign test was trimmed")


def test_the_sign_test_is_per_family_and_all_sixteen_novels():
    p = os.path.join(ROOT, "results", "anchor_scaling_paired.csv")
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    assert len(rows) == 3, "one paired row per family"
    for r in rows:
        assert int(r["n_novels"]) == 16 and int(r["novels_margin_up"]) == 16, r
        assert abs(float(r["sign_test_p"]) - 3.0517578125e-05) < 1e-12, r["sign_test_p"]
