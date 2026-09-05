"""feat-019/021: helpers of the composition attack driver (exact-window matching for the retry variant)."""
from analysis.composition_attack import window_matches


def test_window_matches_ignores_whitespace_only():
    assert window_matches("the  quick\nbrown fox ", "the quick brown fox")
    assert not window_matches("the quick brown fox", "the quick brown dog")
    assert not window_matches("", "the")
