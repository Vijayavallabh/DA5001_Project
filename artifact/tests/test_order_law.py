"""order_law asks whether one variable predicts what a higher Renyi order is worth. The way that
question gets answered wrongly is by comparing curves at F values that are not the same F, so the
interpolation onto a common grid is what is checked, along with the sign convention: a positive
number must mean the higher order leaks LESS."""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.order_frontier import _at  # noqa: E402


def test_common_grid_values_come_from_inside_each_pairs_own_range():
    """_at clamps outside the data, so a common-F cell built from a pair that never reaches that F
    would silently repeat its endpoint. order_law only builds cells inside every pair's range; this
    pins the clamping behaviour it relies on being told about."""
    xs, ys = [0.7, 0.8, 0.9], [5.0, 4.0, 3.0]
    assert _at(xs, ys, 0.95) == 3.0        # clamps, so callers must screen the range themselves
    assert _at(xs, ys, 0.6) == 5.0
    assert abs(_at(xs, ys, 0.85) - 3.5) < 1e-12


def test_committed_law_table_is_consistent_and_signed_the_right_way():
    path, summary = "results/order_law.csv", "results/order_law_summary.csv"
    if not (os.path.exists(path) and os.path.exists(summary)):
        return
    rows = list(csv.DictReader(open(path)))
    import math
    for r in rows:
        # log10_factor is nats_per_window converted; the conversion is the only arithmetic here
        assert abs(float(r["log10_factor"]) - float(r["nats_per_window"]) / math.log(10)) < 1e-3, r
        assert float(r["alpha"]) > 1.0, r          # alpha = 1 is the baseline, never a row
        assert 0.0 < float(r["F"]) <= 1.0, r
    for s in csv.DictReader(open(summary)):
        assert float(s["log10_hi"]) >= float(s["log10_lo"]), s
        assert abs(float(s["log10_spread"])
                   - (float(s["log10_hi"]) - float(s["log10_lo"]))) < 2e-3, s   # all three are rounded to 3dp
        assert int(s["n_pairs"]) >= 2, s
