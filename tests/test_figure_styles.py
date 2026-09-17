"""No two series in a figure may share a colour AND a marker.

Found 2026-09-17 by looking at the rendered appendix figure, which is the only way this is
findable: `onset_collapse` styled its series with `list[i % len(list)]` over an eight-colour,
eight-marker list, and results/onset_pairs.tsv now holds NINE pairs. The ninth wrapped to index 0,
so TinyComma-1.8B and open-calm-3b were the same blue circle -- two indistinguishable curves in
both panels. The figure's own comment said the pair set is "data, not code, so a new admissible
pair appears in the figure without editing it", which is exactly the mechanism. `seed_effect` drew
five series from four markers the same way.

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


def test_asking_for_more_than_there_are_raises_rather_than_wrapping():
    """The defect was silent reuse. Refusing is the whole point."""
    try:
        distinct_styles(11)
    except ValueError as e:
        assert "wrap" in str(e), e
    else:
        raise AssertionError("distinct_styles(11) must refuse, not wrap")


def test_the_collapse_figure_has_a_style_for_every_pair_it_draws():
    path = os.path.join(ROOT, "results", "onset_pairs.tsv")
    pairs = [l for l in open(path, encoding="utf-8").read().splitlines() if l.strip()]
    assert pairs, path
    distinct_styles(len(pairs))          # raises if a pair was added past the palette
    assert len(pairs) == 9, (len(pairs), "the pair count moved; check the figure still reads")


def test_the_seed_effect_figure_has_a_marker_for_every_series_it_draws():
    path = os.path.join(ROOT, "results", "seed_effect.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    drawn = [p for p in {r["pair"] for r in rows}
             if len([r for r in rows if r["pair"] == p]) >= 2]
    assert drawn, path
    distinct_styles(len(drawn))
