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
    """v10 (2026-09-24) moved the read into Table tab:ladder, whose row spells it `$g(128)-g(64)$`
    (no spaces) and whose caption says the rows are paired differences within one pass. Matched
    whitespace-tolerantly (guard the quantity, not the spacing) and the 'paired ... one pass' claim
    is now checked in the caption of the table that carries the row, not anywhere in the appendix."""
    import re
    from tests.manuscript import caption_of
    txt = body("appendix_selection.tex")
    assert "paired" in txt.lower(), "the appendix must say the read is paired"
    assert re.search(r"\$g\(128\)\s*-\s*g\(64\)\$", txt), "the paired quantity must be named as such"
    cap = caption_of("tab:ladder").lower()
    assert "paired" in cap and "one pass" in cap, cap


def test_a_breadth64_anchor_may_not_be_drawn_beside_its_old_n8_row():
    """Decided BEFORE feat-130's numbers existed, so the rule is not chosen to suit them.

    The three feat-130 anchors already have an n=8 row in the forest, taken from a grid-1..8 sweep
    (`selection_scaling_<tag>.csv`). Their new n=64 rows come from a grid-1..64 sweep. Putting the
    two side by side invites the reader to see a climb, and part of any such climb is the
    grid-dependence feat-129 measured -- the same anchor's n=8 gain is not the same number under a
    longer grid (Comma-7B: +0.111 on grid 1..8, +0.072 on grid 1..64).

    So if the figure ever shows an anchor at n=64 from a `*64.csv`, its n=8 row must come from THAT
    file too. The registered read for feat-130 is already the paired g(64) - g(8) WITHIN the new
    pass, which is immune to this, and that is where the comparison belongs.
    """
    fig = os.path.join(ROOT, "figures", "make_figures_v4.py")
    src = open(fig, encoding="utf-8").read()
    for name in ("pleias12b", "kl3m17b", "pleias3b"):
        new = f"selection_scaling_{name}64.csv"
        old = f"selection_scaling_{name}.csv"
        if new in src:
            assert old not in src, (
                f"the forest reads both {new} (grid 1..64) and {old} (grid 1..8) for the same "
                f"anchor. Those two sweeps do not produce comparable single-order gains "
                f"(caution (ap)). Take both rows from {new}, or show only the paired difference.")
