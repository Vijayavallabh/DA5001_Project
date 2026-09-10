"""The matched-utility claim is an interpolation on a k grid, so the interpolation is what has to be
right: reading the matched budget off the wrong side of a bracket, or silently clamping when the
target is off the end of the grid, would manufacture a dominance result out of arithmetic."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.order_frontier import _at, interp  # noqa: E402


def test_interp_finds_the_crossing_and_is_exact_on_a_line():
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [10.0, 20.0, 30.0, 40.0]
    assert abs(interp(xs, ys, 25.0) - 2.5) < 1e-12
    assert abs(interp(xs, ys, 10.0) - 1.0) < 1e-12
    assert abs(interp(xs, ys, 40.0) - 4.0) < 1e-12


def test_interp_returns_none_off_the_grid_instead_of_clamping():
    """A budget the order cannot reach must read as 'off grid', never as the top of the grid --
    clamping would report a matched point that does not exist and compare leakage there."""
    xs, ys = [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]
    assert interp(xs, ys, 55.0) is None      # more utility than any budget on the grid buys
    assert interp(xs, ys, 5.0) is None       # below the grid's own floor
    assert interp(xs, ys, 30.0) == 3.0       # exactly the top of the grid is still on it


def test_at_is_piecewise_linear_and_hits_the_knots():
    xs, ys = [1.0, 2.0, 4.0], [0.0, -10.0, -14.0]
    for x, y in zip(xs, ys):
        assert abs(_at(xs, ys, x) - y) < 1e-12
    assert abs(_at(xs, ys, 1.5) - (-5.0)) < 1e-12
    assert abs(_at(xs, ys, 3.0) - (-12.0)) < 1e-12


def test_committed_sweep_is_monotone_in_k_and_bracketed():
    """Fidelity must rise with the budget at a fixed order -- that monotonicity is the only reason a
    matched budget exists and is unique -- and every cell must sit inside the two models' own
    log-probabilities of the protected tokens."""
    import csv
    for path in ("results/order_frontier_kl3m.csv", "results/order_frontier_pleias.csv"):
        if not os.path.exists(path):
            continue
        rows = list(csv.DictReader(open(path)))
        hi, lo = float(rows[0]["logp_target_risky"]), float(rows[0]["logp_target_safe"])
        tol = 1e-6 * max(1.0, abs(lo))
        by_alpha = {}
        for r in rows:
            by_alpha.setdefault(r["alpha"], []).append((float(r["k"]), float(r["price_nats"]),
                                                        float(r["logp_target"])))
            assert lo - tol <= float(r["logp_target"]) <= hi + tol, (path, r)
        for alpha, cells in by_alpha.items():
            cells.sort()
            for (_, f0, _), (_, f1, _) in zip(cells, cells[1:]):
                assert f1 >= f0 - 1e-6, (path, alpha)
