"""feat-054: the distribution-free restatement of the law. rates() must split a passage the way
the oracle attack does, and cdf_at() must report the fraction a budget rate actually covers."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.surprisal_cdf import cdf_at, rates


def test_windows_are_consecutive_and_non_overlapping():
    nats = list(range(10))
    assert rates(nats, 5) == [2.0, 7.0]          # mean of 0..4 and of 5..9


def test_a_trailing_partial_window_is_dropped():
    """The oracle attack queries whole windows; a 3-token remainder is not one."""
    assert rates(list(range(13)), 5) == [2.0, 7.0]


def test_zero_length_means_the_whole_passage():
    assert rates([1.0, 3.0], 0) == [2.0]
    assert rates([], 0) == []


def test_cdf_counts_the_budget_itself_as_covered():
    vals = [1.0, 2.0, 3.0, 4.0]
    assert cdf_at(vals, 2.0) == 0.5              # rate exactly k is payable
    assert cdf_at(vals, 0.5) == 0.0
    assert cdf_at(vals, 9.0) == 1.0
