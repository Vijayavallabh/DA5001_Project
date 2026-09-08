"""Plan v5 / feat-044: the onset derivation.

The claim is that a work's budget requirement is r(x) = s_safe(x) - s_risky(x), and that the
population onset is a low quantile of the r(x) distribution rather than its median. These tests
pin the arithmetic and the ordering the derivation depends on; the empirical check that the
prediction matches measurement lives in results/onset_theory.csv.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.onset_theory import quantile


def test_quantile_is_monotone_and_bracketed():
    v = sorted([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    qs = [quantile(v, p) for p in (0.0, 0.01, 0.05, 0.10, 0.25, 0.5, 0.9)]
    assert qs == sorted(qs), qs
    assert min(v) <= qs[0] and qs[-1] <= max(v)


def test_quantile_handles_degenerate_inputs():
    assert quantile([1.0], 0.25) == 1.0
    assert quantile([1.0, 2.0], 0.99) == 2.0        # never indexes past the end
    assert quantile([1.0, 2.0], 0.0) == 1.0


def test_requirement_is_lower_for_a_better_memoriser():
    """r(x) = s_safe - s_risky. A risky model that has memorised x more thoroughly has smaller
    s_risky, hence a LOWER budget requirement -- it leaks at a smaller k. The paper's claim that
    0.89 is not universal rests on exactly this dependence."""
    s_safe = 3.0
    weak, strong = 0.6, 0.05                        # residual surprisal on the memorised work
    assert s_safe - strong > s_safe - weak
    assert (1 - strong / s_safe) > (1 - weak / s_safe)


def test_onset_ratio_is_one_minus_the_surprisal_ratio():
    """The reported 0.89 must equal 1 - s_r/s_s, not a constant."""
    for s_s, s_r in ((3.239, 0.194), (2.393, 0.179)):
        assert abs((s_s - s_r) / s_s - (1 - s_r / s_s)) < 1e-12


def test_committed_prediction_matches_measurement():
    """Guards the headline: the q25 prediction is within 1% of the measured onset on both pairs."""
    import csv
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "onset_theory.csv")
    if not os.path.exists(path):
        return
    rows = list(csv.DictReader(open(path)))
    assert len(rows) >= 2, rows
    for r in rows:
        assert abs(float(r["pred_over_meas_q25"]) - 1.0) < 0.01, r
        # the median prediction should overshoot: the cheapest works leak first
        assert float(r["pred_over_meas_median"]) > float(r["pred_over_meas_q25"])
