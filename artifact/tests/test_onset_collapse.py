"""Plan v5: onset.collapse must compare EVERY pair, not the first two.

The original implementation computed abs(vals[0] - vals[1]), so a third pair that disagreed
wildly left the reported agreement untouched. These tests fail against that version.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.onset import collapse, crossing, load_pairs  # noqa: E402


def _curve(scale):
    """Recall rising linearly in k, so k/s(x) = x maps to a known value."""
    return {round(x * scale, 6): x for x in (0.6, 0.8, 1.0, 1.2, 1.4)}


def test_third_pair_disagreement_is_not_hidden():
    two = {("a", "single"): (2.0, _curve(2.0)), ("b", "single"): (3.0, _curve(3.0))}
    three = {**two, ("c", "single"): (4.0, {k: v + 0.5 for k, v in _curve(4.0).items()})}
    s2 = [r["spread"] for r in collapse(two)]
    s3 = [r["spread"] for r in collapse(three)]
    assert max(s2) < 1e-9, "identical rescaled curves must collapse exactly"
    assert min(s3) > 0.4, "an outlying third pair must raise the reported spread"
    assert collapse(three)[0]["n_pairs"] == 3


def test_spread_is_order_invariant():
    a, b, c = (2.0, _curve(2.0)), (3.0, _curve(3.0)), (4.0, {k: v + 0.3 for k, v in _curve(4.0).items()})
    fwd = collapse({("a", "single"): a, ("b", "single"): b, ("c", "single"): c})
    rev = collapse({("c", "single"): c, ("b", "single"): b, ("a", "single"): a})
    assert [r["spread"] for r in fwd] == [r["spread"] for r in rev]


def test_single_pair_yields_no_collapse_row():
    assert collapse({("a", "single"): (2.0, _curve(2.0))}) == []


def test_crossing_interpolates_inside_the_bracket():
    lo, hi, est = crossing({1.0: 0.0, 2.0: 0.02}, 0.01)
    assert (lo, hi) == (1.0, 2.0) and abs(est - 1.5) < 1e-9


def test_manifest_round_trips():
    """The manifest is shared with analysis/onset_units.py (4th field, tokenizer) and
    analysis/collapse_robustness.py (5th field, the pair's name in onset_theory_per_work.csv).
    onset.py must tolerate those and hand back exactly the three it uses."""
    pairs = load_pairs("results/onset_pairs.tsv")
    assert len(pairs) >= 2 and all(len(p) == 3 for p in pairs)


def test_manifest_rejects_a_truncated_row():
    import pytest
    tmp = "/tmp/onset_pairs_short.tsv"
    open(tmp, "w").write("only a name\tand one path\n")
    with pytest.raises(SystemExit):
        load_pairs(tmp)


def test_missing_manifest_falls_back_to_builtin():
    assert len(load_pairs("results/does_not_exist.tsv")) >= 2


def test_collapse_skips_a_pair_swept_in_one_mode_only():
    """Phi-3.5 was swept single-query only. Its oracle curve is empty, and interpolating it used
    to raise IndexError on ks[0] and take the whole refresh down with it."""
    from analysis.onset import collapse
    data = {
        ("a", "single"): (3.0, {2.0: 0.0, 3.0: 0.05, 4.0: 0.2}),
        ("b", "single"): (2.0, {1.5: 0.0, 2.0: 0.04, 3.0: 0.2}),
        ("a", "oracle"): (3.0, {2.0: 0.0, 3.0: 0.1, 4.0: 0.3}),
        ("b", "oracle"): (2.0, {}),          # swept in one mode only
    }
    rows = collapse(data)
    assert rows and all(r["mode"] == "single" for r in rows)   # oracle drops to one series
    assert all(r["n_pairs"] == 2 for r in rows)
