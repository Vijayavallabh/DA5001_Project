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
from tests.manuscript import tex as _tex  # noqa: E402

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
def test_proposition_4_holds_wherever_the_test_has_power():
    """A theorem with a two-line proof, so a violation at a cell that can actually test it is an
    implementation bug and fails loudly here.

    Where `rate(1)` is exactly 0 the empirical bound `n * rate(1)` collapses to 0 and ANY non-zero
    rate registers as a violation at any n -- the estimator has no power, not the theorem no
    validity. Seven of twelve anchors are in that state and phi35mini trips it on ONE passage
    (0.010 against a bound of 0.000, SE 0.00995). Those cells are excluded here and disclosed in
    the scoring log; the band itself is reported VIOLATED at the letter of what was registered and
    is deliberately NOT re-specified."""
    powered = [r for r in _rows() if float(r["base_rate"]) > 0]
    assert len(powered) >= 8, "no cell can test the bound; the arm is vacuous again"
    bad = [r for r in powered if r["prop4_holds"] != "yes"]
    assert not bad, bad


@pytest.mark.skipif(not os.path.exists(CSV), reason="the contaminated-anchor arm has not run")
def test_the_degenerate_cells_are_disclosed_rather_than_filtered():
    """The exclusion above is only honest while the scoring log says it happened, names the count
    and reports the band as VIOLATED. If that disappears, the filter becomes a way of hiding it."""
    log = " ".join(open("results/onset_prediction_contaminated_anchor.md",
                        encoding="utf-8").read().split())
    assert "VIOLATED" in log
    assert "no power" in log and "seven of twelve" in log.lower()
    assert "do not re-specify the band" in log.lower()
    zero = {r["anchor"] for r in _rows() if r["event"] == "E_08" and float(r["base_rate"]) == 0}
    assert len(zero) == 7, sorted(zero)


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


def test_no_markdown_bold_survives_into_the_manuscript():
    """`**text**` is markdown, not LaTeX: it renders as literal asterisks. Five reached the PDF
    before 2026-09-14 -- two in appendix_proofs, three in appendix_selection -- because these
    scoring logs are written in markdown and the prose is moved across by hand. Caution (y); the
    same class as (k)'s hyphen check, and the same one-line grep."""
    import glob
    import re
    bad = []
    for p in glob.glob(_tex("sections/*.tex")) + [_tex("iclr_2027.tex")]:
        if re.search(r"_v\d", p):
            continue                       # kept-verbatim predecessors are not compiled
        for i, line in enumerate(open(p, encoding="utf-8"), 1):
            if "**" in line:
                bad.append(f"{p}:{i}")
    assert not bad, bad


@pytest.mark.skipif(not os.path.exists(CSV), reason="the contaminated-anchor arm has not run")
def test_the_appendix_quotes_the_amplification_range_and_its_caveats():
    apx = " ".join(open(_tex("sections/appendix_selection.tex"), encoding="utf-8").read().split())
    amps = [float(r["amplification"]) for r in _rows()
            if r["event"] == "E_08" and r["n"] == "64" and r["amplification"]]
    assert amps
    lo, hi = min(amps), max(amps)
    assert f"${lo:.1f}$ to ${hi:.1f}\\times$" in apx, (lo, hi)
    # the three limits must stay with the number they qualify
    for phrase in ("seven of twelve anchors sit at exactly zero",
                   "violated rather than re-specifying it",
                   "eight passages over two"):
        assert phrase in apx, phrase
    # and the Ethics Statement must carry the measurement, not the old assertion
    eth = " ".join(open(_tex("iclr_2027.tex"), encoding="utf-8").read().split())
    assert "worthless if the safe model is itself contaminated" not in eth
    assert "multiplier on the anchor's own leakage" in eth
    assert "vet the anchor" in eth
