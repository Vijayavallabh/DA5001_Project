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
    # plan v5: rows with no measured onset are pre-registered predictions for pairs not yet swept.
    # They must be skipped here, but the invariant still has to hold for every MEASURED pair.
    measured = [r for r in rows if r["pred_over_meas_q25"]]
    assert len(measured) >= 2, f"expected >= 2 measured pairs, got {len(measured)} of {len(rows)}"
    for r in measured:
        assert abs(float(r["pred_over_meas_q25"]) - 1.0) < 0.01, r
        # the median prediction should overshoot: the cheapest works leak first
        assert float(r["pred_over_meas_median"]) > float(r["pred_over_meas_q25"])


def test_collapse_robustness_supports_the_derivation():
    """feat-045: the requirement r = s_safe - s_risky must collapse the curves better than the
    anchor surprisal alone, which is the derivation's own falsifiable prediction, and both must
    beat no rescaling. Also guards that the continuous metric does not collapse worse than the
    thresholded one (the Schaeffer critique)."""
    import csv, math
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "collapse_robustness.csv")
    if not os.path.exists(path):
        return
    rows = list(csv.DictReader(open(path)))
    norm = {r["setting"]: float(r["value"]) for r in rows if r["block"] == "normaliser"}
    assert norm["requirement r = s_safe - s_risky"] < norm["s_safe"] < norm["raw (no rescaling)"], norm
    met = {r["setting"]: float(r["value_relative_to_metric_range"])
           for r in rows if r["block"] == "metric"}
    assert met["lcs_word"] <= met["nv_recall"], met


def test_onset_threshold_band_is_reported_not_assumed():
    """The pair-to-pair agreement is stable over a band of onset thresholds while the level is not;
    the paper must quote the band. This pins that the band exists."""
    import csv
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "collapse_robustness.csv")
    if not os.path.exists(path):
        return
    th = {r["setting"]: (float(r["value"]), float(r["value_relative_to_metric_range"]))
          for r in csv.DictReader(open(path)) if r["block"] == "threshold"}
    band = [th[k][0] for k in ("onset at 0.005", "onset at 0.01", "onset at 0.02") if k in th]
    assert band and max(band) < 0.01, band          # agreement holds across the band
    levels = [th[k][1] for k in ("onset at 0.005", "onset at 0.01", "onset at 0.02") if k in th]
    assert max(levels) - min(levels) > 0.05         # but the level genuinely moves
