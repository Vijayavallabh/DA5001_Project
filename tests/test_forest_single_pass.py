"""The forest plot must not mix judging passes, which caution (ap) makes an error of construction.

Every judged row in Figure 2 comes from the committed pass (`selection_scaling*.csv`). Adding the
n=128 row from `selection_scaling_n128.csv` beside the n=64 row from `selection_scaling.csv` would
set a number from one sweep against a number from another whose grid differs -- and that is exactly
the comparison feat-129 showed is invalid: identical generations, 1,954 vs 2,219 distinct served
completions, every presentation flip re-rolled, and judge B moving 0.435 -> 0.478 on the same text.

The saturation result therefore belongs where it is, as a PAIRED difference within one pass in the
appendix, and must never be drawn as an extra forest row. This test exists so that a later session
reaching for the obvious figure update fails instead of silently making the comparison.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import _forest, body  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_no_forest_row_comes_from_the_n128_pass():
    fig = os.path.join(ROOT, "figures", "make_figures_v4.py")
    src = open(fig, encoding="utf-8").read()
    assert "selection_scaling_n128.csv" not in src, (
        "the forest plot now reads the n=128 pass; its other judged rows come from the committed "
        "pass, and the two are not comparable (caution (ap)). Report the saturation as the paired "
        "within-pass difference it was registered as, not as a figure row.")


def test_the_forest_still_carries_the_committed_n64_row():
    labels = [lbl for lbl, _b, _c, _cert, _g in _forest()]
    assert any("n=64" in lbl or "$n=64$" in lbl or "n = 64" in lbl for lbl in labels), labels


def test_the_saturation_is_reported_as_a_paired_within_pass_difference():
    txt = body("appendix_selection.tex")
    assert "paired" in txt.lower(), "the appendix must say the read is paired"
    assert "$g(128) - g(64)$" in txt, "the paired quantity must be named as such"
