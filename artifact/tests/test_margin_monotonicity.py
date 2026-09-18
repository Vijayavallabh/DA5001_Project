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
    txt = body("frontier.tex")
    i = txt.find("rises monotonically with capability")
    assert i != -1, "the margin trend sentence is gone from Section 4"
    assert "rises monotonically with capability in every family" in txt, \
        "the main text dropped 'in every family' -- pooled, the series is not monotone"


def test_the_appendix_caption_still_says_the_same_thing():
    txt = " ".join(open(tex("sections/appendix_robustness.tex"), encoding="utf-8").read().split())
    assert "rises monotonically with capability in every family" in txt, \
        "the appendix caption lost the scope the main text is checked against"
    assert "$p = 3.1\\times10^{-5}$ per family" in txt, "the per-family sign test was trimmed"


def test_the_sign_test_is_per_family_and_all_sixteen_novels():
    p = os.path.join(ROOT, "results", "anchor_scaling_paired.csv")
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    assert len(rows) == 3, "one paired row per family"
    for r in rows:
        assert int(r["n_novels"]) == 16 and int(r["novels_margin_up"]) == 16, r
        assert abs(float(r["sign_test_p"]) - 3.0517578125e-05) < 1e-12, r["sign_test_p"]
