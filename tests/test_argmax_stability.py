"""Guards for the exploratory argmax-stability arm (no committed bands; see its note).

The point of these is caution (ai)/(ao): a number is checked against its CSV, and the CLAIM ABOUT a
set of numbers is checked against nothing. Two shape claims are made in the note --- the margin falls
monotonically, and the agreement falls and then PLATEAUS rather than falling monotonically --- so both
are rebuilt from the CSV and asserted, and the non-monotone one is asserted in the direction that
fails if it ever becomes monotone, which is what says to revisit the wording rather than quietly
permitting the stronger claim.
"""
import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_argmax_stability import analyse, margin, pick  # noqa: E402

CSV = "results/selection_argmax_stability.csv"


def _rows():
    with open(CSV, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _series(comparison, field):
    rs = sorted((r for r in _rows() if r["comparison"] == comparison),
                key=lambda r: int(r["n"]))
    return [(int(r["n"]), float(r[field])) for r in rs if r[field] != ""]


def test_the_nesting_rule_is_selection_scalings_own():
    r = {("a", 0): 1.0, ("a", 1): 5.0, ("a", 2): 2.0, ("a", 3): 9.0}
    assert pick(r, "a", 1) == 0 and pick(r, "a", 2) == 1 and pick(r, "a", 4) == 3


def test_margin_is_the_gap_to_the_runner_up_and_undefined_at_one():
    r = {("a", 0): 1.0, ("a", 1): 5.0, ("a", 2): 2.0}
    assert margin(r, "a", 1) is None
    assert margin(r, "a", 2) == pytest.approx(4.0)
    assert margin(r, "a", 3) == pytest.approx(3.0)


def test_analyse_reports_a_perfect_agreement_and_zero_loss_against_itself():
    """A cache compared with itself must serve the same candidate everywhere and give up nothing.
    Without this, a bug that always reported agreement would look like a finding."""
    ref = {(f"p{p}", j): -30.0 + p + j * 0.5 for p in range(4) for j in range(64)}
    res = analyse(ref, dict(ref))
    assert res and all(r["agree_frac"] == 1.0 for r in res)
    assert all(r["reward_loss_all"] == 0.0 for r in res)
    assert all(r["n_disagree"] == 0 for r in res)


def test_analyse_detects_a_flipped_winner_and_prices_it():
    """And the loss must be measured on the REFERENCE cache's scale, not the other one's."""
    ref = {("p", j): -30.0 + j * 0.5 for j in range(64)}          # rank 63 wins, by 0.5
    oth = dict(ref)
    oth[("p", 0)] = 100.0                                          # rank 0 wins in the other cache
    res = {r["n"]: r for r in analyse(ref, oth)}
    assert res[64]["agree_frac"] == 0.0
    # ref says rank 63 is worth -30+31.5 = 1.5 and rank 0 is worth -30.0, so the pick gives up 31.5
    assert res[64]["reward_loss_on_disagreements"] == pytest.approx(31.5, abs=1e-6)


def test_analyse_refuses_caches_that_share_no_complete_prompt():
    with pytest.raises(AssertionError):
        analyse({("a", 0): 1.0}, {("b", 0): 1.0})


@pytest.mark.skipif(not os.path.exists(CSV), reason="the stability arm has not been run")
def test_the_margin_really_does_fall_monotonically():
    """Claim 2 of the note, and the mechanism the other two readings rest on."""
    for comp in {r["comparison"] for r in _rows()}:
        s = [v for _, v in _series(comp, "mean_margin")]
        assert len(s) >= 5, (comp, s)
        assert all(b < a for a, b in zip(s, s[1:])), f"{comp} margin is no longer monotone: {s}"


@pytest.mark.skipif(not os.path.exists(CSV), reason="the stability arm has not been run")
def test_agreement_falls_with_n_but_is_NOT_monotone_and_the_note_says_plateau():
    """Claim 1, guarded BOTH ways (caution (ao)). If agreement ever does become monotone in n this
    fails, and the right response is to re-read the table and reword the note -- not to relax it."""
    for comp in {r["comparison"] for r in _rows()}:
        s = dict(_series(comp, "agree_frac"))
        assert s[1] == 1.0, comp
        assert s[64] < s[8] - 0.02, f"{comp}: agreement no longer falls with n: {s}"
        vals = [s[n] for n in sorted(s)]
        assert not all(b <= a for a, b in zip(vals, vals[1:])), \
            f"{comp} agreement is now monotone in n; the note says it plateaus -- reword it: {vals}"
    note = open("results/selection_argmax_stability_note.md", encoding="utf-8").read()
    assert "no committed bands" in note
    assert "it is not monotone" in note


@pytest.mark.skipif(not os.path.exists(CSV), reason="the stability arm has not been run")
def test_the_quality_served_is_reproducible_even_where_the_text_is_not():
    """Claim 3, the one that keeps this from being overstated. The reward given up must stay small
    against the margin it is measured in; if it ever grows comparable, this is a utility defect and
    not merely a reproducibility one, and the note's framing has to change."""
    for comp in {r["comparison"] for r in _rows()}:
        loss = dict(_series(comp, "reward_loss_all"))
        marg = dict(_series(comp, "mean_margin"))
        assert loss[64] < 0.10, f"{comp}: mean reward given up at n=64 is {loss[64]}"
        assert loss[64] < 0.05 * marg[64], \
            f"{comp}: the pick now gives up {loss[64]} against a margin of {marg[64]}"


@pytest.mark.skipif(not os.path.exists(CSV), reason="the stability arm has not been run")
def test_precision_alone_is_at_least_as_disruptive_as_changing_hosts():
    """The note's closing claim, and the reason feat-136's gate was repaired rather than its arms
    discarded. Asserted at n=16, where all three series have already fallen off their plateau."""
    rows = {r["comparison"]: r for r in _rows() if int(r["n"]) == 16}
    if not {"precision", "host"} <= set(rows):
        pytest.skip("both comparisons are needed")
    assert float(rows["precision"]["agree_frac"]) <= float(rows["host"]["agree_frac"]), \
        "changing hosts now disagrees MORE than changing precision; the note's claim needs revisiting"
