"""Every placed figure must print at >= 0.70 of the size it was drawn at.

Nothing in the build catches a figure that is shrunk too far: tectonic exits 0, the overfull count
is 0, `??` is 0, and the page silently carries 3.5pt type. It has bitten twice --- frontier_scaling
was drawn at 6.9in and placed at 0.62\\textwidth, a shrink of 0.494, so its 7pt legend printed at
3.5pt (caution (af)); and the forest plot saved 9.09in wide because `bbox_inches="tight"` expands
the canvas around anything drawn outside the axes, so a 7pt label printed at 5.0 (caution (ak)).

The shrink must be computed as `\\textwidth * placement_fraction / SAVED FILE WIDTH`, read out of
`pdfinfo` -- never from the `figsize` argument, which is not what was written to disk. That is the
whole lesson of caution (ak), and it is why this test shells out to pdfinfo instead of importing
the figure code.
"""
import os
import re
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import DIR, tex  # noqa: E402

TEXTWIDTH_PT = 430.9   # ICLR 2027, measured off the compiled page (caution (ak))
FLOOR = 0.70
PLACE = re.compile(r"includegraphics\[width=([0-9.]*)\\textwidth\]\{(?:figures/)?([A-Za-z0-9_]+)\.pdf\}")


def _live_files():
    main = open(tex("iclr_2027.tex"), encoding="utf-8").read()
    files = [tex(m + ".tex") for m in re.findall(r"\\input\{(sections/[a-z_0-9]+)\}", main)]
    return files + [tex("iclr_2027.tex")]


def _placed():
    out = {}
    for f in _live_files():
        if not os.path.exists(f):
            continue
        for m in PLACE.finditer(open(f, encoding="utf-8").read()):
            out[m.group(2)] = float(m.group(1)) if m.group(1) else 1.0
    return out


def _width_pt(name):
    p = os.path.join(DIR, "figures", name + ".pdf")
    assert os.path.exists(p), f"{name}.pdf is placed in the document but not in figures/"
    out = subprocess.run(["pdfinfo", p], capture_output=True, text=True).stdout
    m = re.search(r"Page size:\s+([0-9.]+)", out)
    assert m, out
    return float(m.group(1))


@pytest.mark.skipif(shutil.which("pdfinfo") is None, reason="pdfinfo not available")
def test_the_document_still_places_figures():
    """A test that finds nothing to check is a test that passes by never running (caution (aj))."""
    placed = _placed()
    assert len(placed) >= 8, f"only {len(placed)} placed figures found -- the scan is broken"
    # the path form is easy to miss: the forest plot is included as figures/<name>.pdf
    assert "selection_breadth_forest" in placed, "the forest plot was not seen by the scan"
    assert "selection_frontier" in placed, "Figure 1 was not seen by the scan"


@pytest.mark.skipif(shutil.which("pdfinfo") is None, reason="pdfinfo not available")
def test_every_placed_figure_clears_the_shrink_floor():
    bad = []
    for name, frac in sorted(_placed().items()):
        s = TEXTWIDTH_PT * frac / _width_pt(name)
        if s < FLOOR:
            bad.append((name, round(s, 3), frac))
    assert not bad, (
        f"figures printing below {FLOOR} of their drawn size: {bad}. Either widen the "
        f"\\includegraphics or shrink the figsize -- and re-measure the SAVED file, not figsize.")


def test_figure_1a_label_placement_and_the_data_condition_it_rests_on():
    """Two rendering collisions in the paper's only main-text figure, and the layout that fixed them.

    Found 2026-09-18 by rendering the compiled page and looking (caution (ad): grep cannot see any
    of this). `$S(x) = s(x)T$` anchored at T[26] ran straight into `$k=3$` at T[40] -- the two rays
    are nearly collinear, because s(x) is about 3 nats/token, which is the panel's whole point -- so
    the page printed the two labels as one smear. Moving k=3 to T[62] then put it on the vertical
    median-target rule, caution (ad)'s "one collision traded for another"; T[52] is the gap.

    The second is subtler and was measured, not guessed. Every ray has slope 1 on log-log axes, so
    the gap between rays is constant, while a label rotated to 31 degrees RISES about 0.67 decades
    over its own length. k=0.5 sits log10(3/0.5) = 0.78 decades below the S(x)/k=3 pair, so its
    label, drawn above its own ray, put its top edge into both. Anchors were exact -- reading the
    word boxes out of the figure PDF reproduces the axis transform to 0.05pt -- so what collided was
    the extent. It is now the one label drawn BELOW its ray.

    A pairwise bbox-overlap test over the figure is not the guard: it reports 27 pairs, nearly all
    of them two-line annotations whose boxes legitimately share rows. What is worth pinning is the
    placement and the data condition that made it necessary.
    """
    import csv as _csv
    import os as _os
    import re as _re
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    src = open(_os.path.join(root, "figures", "make_figures_v4.py"), encoding="utf-8").read()

    m = _re.search(r"for k, style, xi, dy in \((.*?)\):", src, _re.S)
    assert m, "the k-ray label loop was restructured; re-check the layout against a rendered page"
    placed = _re.findall(r"\(([\d.]+), \"[^\"]*\", (\d+), (-?\d+)\)", m.group(1))
    assert [(float(k), int(xi), int(dy)) for k, xi, dy in placed] == \
        [(10.0, 11, 4), (3.0, 52, 4), (0.5, 85, -4)], placed

    # the condition the layout rests on: S(x)'s slope sits between the k=0.5 and k=10 rays and close
    # enough to k=3 that their labels cannot share an x. If the data ever moves it out of this band,
    # the panel needs looking at again rather than silently re-rendering.
    onset = list(_csv.DictReader(
        open(_os.path.join(root, "results", "onset_table.csv"), encoding="utf-8")))
    ss = sorted(float(r["s_safe"]) for r in onset)
    s_med = ss[len(ss) // 2]
    assert 2.0 < s_med < 4.0, (s_med, "S(x)'s ray has moved; re-check Figure 1(a)'s label layout")
    import math as _m
    assert _m.log10(3.0 / 0.5) > 0.67, "the k=0.5 label would no longer clear the S(x)/k=3 pair"
