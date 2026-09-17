"""No two series in a figure may share a colour AND a marker.

Found 2026-09-17 by looking at the rendered appendix figure, which is the only way this is
findable: `onset_collapse` styled its series with `list[i % len(list)]` over an eight-colour,
eight-marker list, and results/onset_pairs.tsv now holds NINE pairs. The ninth wrapped to index 0,
so TinyComma-1.8B and open-calm-3b were the same blue circle -- two indistinguishable curves in
both panels. The figure's own comment said the pair set is "data, not code, so a new admissible
pair appears in the figure without editing it", which is exactly the mechanism. `seed_effect` uses the same construct over a four-marker list but filters
its temperature arms out first and draws only three, so it had not yet collided -- it is
pinned here because the next arm added is what would tip it.

The style capacity is now tied to the data here, so adding a tenth pair fails this test instead of
silently colliding in a figure nobody re-renders.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "figures"))
from make_figures_v4 import distinct_styles  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_styles_are_distinct_and_never_wrap():
    for n in range(1, 11):
        st = distinct_styles(n)
        assert len(st) == n and len(set(st)) == n, (n, st)
        assert len({c for c, _ in st}) == n, ("colours repeat", n, st)
        assert len({m for _, m in st}) == n, ("markers repeat", n, st)


def test_past_ten_the_pair_still_separates_them():
    """Colour alone runs out at ten. Beyond that the (colour, marker) PAIR must stay unique --
    that is what a reader uses -- and the function must refuse rather than reuse a pair."""
    for n in (11, 12, 20, 100):
        st = distinct_styles(n)
        assert len(st) == n and len(set(st)) == n, (n, len(set(st)))
        assert len({c for c, _ in st}) <= 10, n     # colour necessarily repeats
    try:
        distinct_styles(101)
    except ValueError as e:
        assert "distinct styles" in str(e), e
    else:
        raise AssertionError("distinct_styles(101) must refuse, not wrap")


def test_the_collapse_figure_has_a_style_for_every_pair_it_draws():
    path = os.path.join(ROOT, "results", "onset_pairs.tsv")
    pairs = [l for l in open(path, encoding="utf-8").read().splitlines() if l.strip()]
    assert pairs, path
    distinct_styles(len(pairs))          # raises if a pair was added past the palette
    assert len(pairs) == 9, (len(pairs), "the pair count moved; check the figure still reads")


def test_the_seed_effect_figure_has_a_marker_for_every_series_it_draws():
    path = os.path.join(ROOT, "results", "seed_effect.csv")
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if not r["pair"].endswith(" tau")]      # the figure drops these before styling
    drawn = [p for p in {r["pair"] for r in rows}
             if len([r for r in rows if r["pair"] == p]) >= 2]
    assert len(drawn) == 3, (sorted(drawn), "the series count moved; re-render and look")
    distinct_styles(len(drawn))


def test_the_order_figure_has_a_style_for_every_pair_it_draws():
    """Twelve series, ten colours, one marker: matplotlib wraps C10 to C0, so Comma-7B and
    Qwen2.5-7B were the same blue circle and KL3M-1.7B and TinyComma-1.8B the same orange one,
    across all three panels. `% len(...)` appears nowhere in that code -- the wrap is inside
    matplotlib's own property cycle, which is why grepping for the explicit form missed it and
    only rendering the figure and looking at it found it (2026-09-17)."""
    path = os.path.join(ROOT, "results", "order_law.csv")
    if not os.path.exists(path):
        path = os.path.join(ROOT, "results", "order_law_summary.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    pairs = sorted({r["pair"] for r in rows if r.get("pair")})
    assert len(pairs) >= 11, (len(pairs), "the figure this guards draws twelve")
    st = distinct_styles(len(pairs))
    assert len(set(st)) == len(pairs)
    # and past ten it must be the PAIR that separates them, not the colour alone
    assert len({c for c, _ in st}) < len(pairs), st
