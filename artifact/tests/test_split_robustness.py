"""feat-060: the tokenizer split must be a fact about the pairs, not about how recall is
normalised, so the sweep has to group by characters-per-token and report overlaps honestly."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.split_robustness import GROUP_SPLIT_CHARS_PER_TOKEN, METRICS, PAIRS, crossing


def test_the_three_metrics_are_normalised_differently():
    """The whole point: a denominator of reference words, no denominator, and a denominator of
    passages. If they ever became the same measure the sweep would prove nothing."""
    assert set(METRICS) == {"nv_recall", "lcs_word", "any_span"}


def test_both_groups_are_populated():
    wide = [p for p in PAIRS if p[3] > GROUP_SPLIT_CHARS_PER_TOKEN]
    narrow = [p for p in PAIRS if p[3] <= GROUP_SPLIT_CHARS_PER_TOKEN]
    assert len(wide) == 4 and len(narrow) == 2


def test_crossing_interpolates_inside_its_bracket():
    c = {1.0: 0.0, 2.0: 0.02}
    assert abs(crossing(c, 0.01) - 1.5) < 1e-12
    assert crossing({1.0: 0.0, 2.0: 0.0}, 0.01) is None      # never reaches the threshold


def test_a_curve_starting_above_threshold_is_not_a_crossing():
    """The first grid point already above the threshold means the onset is below the grid, which
    is not a measurement; reporting it as one would put an onset at the smallest budget probed."""
    assert crossing({1.0: 0.5, 2.0: 0.6}, 0.01) is None


def test_recorded_verdicts_match_the_committed_csv():
    import csv
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "results", "split_robustness.csv")
    if not os.path.exists(p):
        return
    rows = [r for r in csv.DictReader(open(p)) if r["verdict"] in ("split", "overlap")]
    # 9 of 10 usable cells split; the exception is the passage-fraction metric at one threshold,
    # and the paper says so rather than quoting only the metrics that agree.
    assert sum(r["verdict"] == "split" for r in rows) >= len(rows) - 1, rows
    assert all(r["verdict"] == "split" for r in rows if r["metric"] == "lcs_word")
