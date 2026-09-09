"""feat-053: the natural (unfine-tuned) 70B pair is compared to the built pairs by interpolating
THEM onto ITS grid, so the interpolation and the coarse-grid calibration both have to be right."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.natural_pair import coarse_onset, interp, per_passage


def test_interp_returns_none_outside_the_grid():
    c = {2.0: 0.0, 4.0: 0.10}
    assert interp(c, 2.0, 0.5) is None      # below k/s = 1.0
    assert interp(c, 2.0, 2.5) is None      # above k/s = 2.0
    assert abs(interp(c, 2.0, 1.5) - 0.05) < 1e-12


def test_interp_endpoints_are_inclusive():
    c = {2.0: 0.02, 4.0: 0.10}
    assert interp(c, 2.0, 1.0) == 0.02
    assert interp(c, 2.0, 2.0) == 0.10


def test_coarse_onset_widens_the_bracket_and_lowers_the_estimate():
    """A convex recall curve read on a coarse grid puts the crossing too low; the calibration block
    exists to measure that, so it must reproduce it on a curve where the answer is known."""
    fine = {1.0: 0.0, 2.0: 0.0, 2.5: 0.0, 3.0: 0.004, 3.2: 0.02}
    coarse = coarse_onset(fine, 1.0, 3.2)
    exact = 3.0 + 0.2 * (0.01 - 0.004) / (0.02 - 0.004)
    assert coarse is not None and coarse < exact
    assert abs(coarse - (1.0 + 2.2 * 0.01 / 0.02)) < 1e-12


def test_baselines_are_not_budget_points(tmp_path):
    p = tmp_path / "composition.csv"
    p.write_text("k,mode,nv_recall\n-1,single,0.9\n0,single,0.0\n3,single,0.02\n3,oracle,0.5\n")
    got = per_passage(str(p))
    assert list(got) == [3.0] and got[3.0] == [0.02]
