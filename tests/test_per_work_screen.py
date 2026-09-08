"""Plan v5: the AUC used to test r(x) as a per-work screen must be correct, including ties.

A silently wrong AUC would have produced the headline finding of this analysis (that s(x) screens
individual works better than the derived r(x)), so it is pinned here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.per_work_screen import auc  # noqa: E402


def test_perfect_separation():
    assert auc([3.0, 2.0, 1.0, 0.0], [True, True, False, False]) == 1.0


def test_perfect_inversion():
    assert auc([0.0, 1.0, 2.0, 3.0], [True, True, False, False]) == 0.0


def test_all_ties_is_a_coin_flip():
    assert auc([1.0, 1.0, 1.0, 1.0], [True, False, True, False]) == 0.5


def test_partial_ties_count_half():
    # one positive above one negative, one exact tie -> (1 + 0.5) / 2
    assert auc([2.0, 1.0, 1.0], [True, True, False]) == 0.75


def test_single_class_returns_none():
    assert auc([1.0, 2.0], [True, True]) is None
    assert auc([1.0, 2.0], [False, False]) is None


def test_matches_a_brute_force_reference():
    import itertools
    import random
    rng = random.Random(0)
    for _ in range(20):
        n = rng.randint(4, 9)
        sc = [rng.choice([0.0, 1.0, 2.0]) for _ in range(n)]
        lb = [rng.random() < 0.5 for _ in range(n)]
        if not any(lb) or all(lb):
            continue
        pos = [s for s, y in zip(sc, lb) if y]
        neg = [s for s, y in zip(sc, lb) if not y]
        ref = sum((p > q) + 0.5 * (p == q) for p, q in itertools.product(pos, neg)) / (len(pos) * len(neg))
        assert abs(auc(sc, lb) - ref) < 1e-12
