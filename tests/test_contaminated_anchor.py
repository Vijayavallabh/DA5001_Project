"""feat-111: the arm that tests Proposition 4 where it can actually fail.

Every other leakage arm has a clean anchor and a base rate of zero, so `n x 0 = 0` holds whatever
is measured. These tests pin the two things that make this arm mean something: that the base rate
really is non-zero, and that the multiplication bound is checked as an EVENT probability rather
than on a mean recall, which Proposition 4 says nothing about.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.contaminated_anchor import (EVENTS, N_GRID, SATURATION_BAND,  # noqa: E402
                                          rate, spearman)

CSV = "results/contaminated_anchor.csv"


def test_rate_is_an_event_probability_with_its_binomial_error():
    p, se = rate([0.9, 0.85, 0.0, 0.5], 0.8)
    assert p == 0.5
    assert se == pytest.approx((0.5 * 0.5 / 4) ** 0.5)
    assert rate([0.0, 0.0], 0.01) == (0.0, 0.0)
    # the threshold is inclusive: a passage exactly at the line counts as the event
    assert rate([0.8], 0.8)[0] == 1.0


def test_the_two_events_are_the_registered_ones_and_the_grid_is_nested():
    assert EVENTS == (("E_08", 0.8), ("E_001", 0.01))
    assert N_GRID == (1, 8, 64)
    assert SATURATION_BAND == 4.0


def test_spearman_is_exact_on_a_monotone_and_a_reversed_series():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def _rows():
    import csv
    with open(CSV, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.mark.skipif(not os.path.exists(CSV), reason="the contaminated-anchor arm has not run")
def test_the_base_rate_is_non_zero_or_the_arm_proves_nothing():
    """The whole point. If every anchor came back at rate(1) = 0 the bound would be vacuous again
    and N1 would be unreadable, exactly as it is for the clean anchors."""
    bases = {r["anchor"]: float(r["base_rate"]) for r in _rows() if r["event"] == "E_08"}
    assert bases, "no arms"
    assert sum(1 for v in bases.values() if v > 0) >= 3, bases


@pytest.mark.skipif(not os.path.exists(CSV), reason="the contaminated-anchor arm has not run")
def test_proposition_4_holds_at_every_cell():
    """A theorem with a two-line proof. A violation is an implementation bug, and the
    pre-registration says so; this fails loudly rather than letting one reach the paper."""
    bad = [r for r in _rows() if r["prop4_holds"] != "yes"]
    assert not bad, bad


@pytest.mark.skipif(not os.path.exists(CSV), reason="the contaminated-anchor arm has not run")
def test_amplification_is_reported_wherever_the_base_rate_allows_it():
    for r in _rows():
        if float(r["base_rate"]) > 0:
            assert r["amplification"] != "", r
            assert float(r["amplification"]) >= 0
        else:
            assert r["amplification"] == "", r


@pytest.mark.skipif(not os.path.exists(CSV), reason="the contaminated-anchor arm has not run")
def test_both_events_are_reported_at_every_anchor():
    """Excluded alternative: reading N2 on whichever event is favourable."""
    seen = {}
    for r in _rows():
        seen.setdefault(r["anchor"], set()).add(r["event"])
    assert seen
    for anchor, ev in seen.items():
        assert ev == {"E_08", "E_001"}, (anchor, ev)
