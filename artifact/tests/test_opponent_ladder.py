"""feat-175: the opponent ladder's gate and bands, MUTATION-TESTED BEFORE THE DATA EXIST.

Caution (ap): a gate must be repaired before the result it gates is seen, and caution (aq): the
cheapest honest check is to break each thing the change claims and confirm a named test fails.
Every case here is synthetic. None of the three new opponents had been generated when this was
written, so nothing below can have been tuned to an answer.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.opponent_strength import FLOOR, h1_h2, spearman, exact_p  # noqa: E402


def row(name, strength, lo, hi, status="new", g3="PASS"):
    d3 = (lo + hi) / 2
    verdict = ("REVERSAL CONFIRMED" if lo > 0 else
               "REVERSAL REFUTED" if hi < 0 else "REVERSAL UNRESOLVED")
    return dict(opponent=name, status=status, strength=strength, d3=d3,
                d3_lo95=lo, d3_hi95=hi, verdict=verdict, g3=g3)


# The committed pass, as it measures today. H1's threshold comes from this row and nowhere else.
REF = row("llama8b", 0.555, 0.0300, 0.0995, status="committed")


def test_the_threshold_comes_from_the_committed_pass_and_not_from_a_constant():
    h1, _h2, thr, _s, _b = h1_h2([REF, row("a", 0.40, 0.01, 0.05)])
    assert thr == 0.555 and h1 == "STRENGTH SUPPORTED"
    moved = dict(REF, strength=0.30)
    # With the committed opponent measuring 0.30, a 0.40 opponent is no longer BELOW it, so the
    # same new row stops being a test of H1. The threshold tracks the data, not a literal.
    h1b, _h2b, thrb, _sb, _bb = h1_h2([moved, row("a", 0.40, 0.01, 0.05)])
    assert thrb == 0.30 and h1b == "NOT TESTED"


def test_one_unresolved_opponent_below_the_threshold_refutes_strength():
    ok = [REF, row("a", 0.35, 0.01, 0.05), row("b", 0.45, 0.02, 0.06)]
    assert h1_h2(ok)[0] == "STRENGTH SUPPORTED"
    # MUTATION: widen one interval so it straddles zero. Exactly one row changes.
    bad = [REF, row("a", 0.35, -0.01, 0.05), row("b", 0.45, 0.02, 0.06)]
    assert sum(1 for x, y in zip(ok, bad) if x != y) == 1
    assert h1_h2(bad)[0] == "STRENGTH REFUTED"


def test_an_inverted_opponent_below_the_threshold_also_refutes_it():
    assert h1_h2([REF, row("a", 0.40, -0.06, -0.02)])[0] == "STRENGTH REFUTED"


def test_an_opponent_above_the_threshold_is_excluded_rather_than_counted():
    # A new opponent that turns out STRONGER than the committed one is not a test of "weaker
    # opponents restore the reversal", so it may neither confirm nor refute H1.
    h1, _h2, _thr, scored, below = h1_h2([REF, row("a", 0.90, -0.05, 0.02)])
    assert h1 == "NOT TESTED" and scored == [] and below == []


def test_a_g3_failure_is_excluded_from_h1_in_both_directions():
    # A degenerate run reads as a weak opponent. It must not be able to refute H1 ...
    assert h1_h2([REF, row("a", 0.20, -0.01, 0.05, g3="FAIL")])[0] == "NOT TESTED"
    # ... nor to confirm it.
    assert h1_h2([REF, row("a", 0.20, 0.01, 0.05, g3="FAIL")])[0] == "NOT TESTED"


def test_the_floor_reading_fires_only_with_a_mid_range_confirmation_beside_it():
    lo_unres = row("tiny", FLOOR - 0.05, -0.01, 0.03)
    mid_conf = row("mid", 0.45, 0.02, 0.06)
    h1, h2, _t, scored, _b = h1_h2([REF, lo_unres, mid_conf])
    assert h2 == "COMPRESSION AT BOTH ENDS"
    # and the floor row is then reported INSTEAD of refuting H1, which is the whole point of
    # registering it: a ceiling artefact is not evidence against the strength account.
    assert h1 == "STRENGTH SUPPORTED" and scored == ["mid"]
    # Without the mid-range confirmation it is just a refutation and must be read as one.
    h1b, h2b, _t2, _s2, _b2 = h1_h2([REF, lo_unres])
    assert h2b == "NOT FIRED" and h1b == "STRENGTH REFUTED"


def test_the_floor_reading_cannot_rescue_an_inverted_result():
    # H2 is about an interval with no room, not about one that points the other way.
    inverted = row("tiny", FLOOR - 0.05, -0.08, -0.02)
    h1, h2, _t, _s, _b = h1_h2([REF, inverted, row("mid", 0.45, 0.02, 0.06)])
    assert h2 == "NOT FIRED" and h1 == "STRENGTH REFUTED"


def test_the_committed_row_must_be_present():
    with pytest.raises(AssertionError):
        h1_h2([row("a", 0.40, 0.01, 0.05)])


def test_the_exact_permutation_p_is_right_at_the_sizes_this_arm_uses():
    # A perfect descending rank order over five points is the strongest result available, and its
    # exact two-sided p is 2/120. Quoting anything smaller would be a bug.
    xs = [0.1, 0.2, 0.3, 0.4, 0.5]
    ys = [0.5, 0.4, 0.3, 0.2, 0.1]
    assert spearman(xs, ys) == pytest.approx(-1.0)
    assert exact_p(xs, ys) == pytest.approx(2 / 120)
    # and over three points it is 2/6, which is why H3 is exploratory and not a band.
    assert exact_p([0.1, 0.2, 0.3], [0.3, 0.2, 0.1]) == pytest.approx(2 / 6)


def test_g3_fires_on_both_shapes_it_was_written_for():
    from analysis.opponent_strength import g3
    ref = 154.08                       # the committed opponent's measured mean, derived at run time
    assert g3(0.00, 150.0, ref) == "PASS"
    assert g3(0.09, 150.0, ref) == "PASS"
    assert g3(0.11, 150.0, ref) == "FAIL"          # empties
    assert g3(0.00, ref / 3.1, ref) == "FAIL"      # truncation
    assert g3(0.00, ref * 3.1, ref) == "FAIL"      # runaway repetition
    assert g3(0.00, ref / 3.0, ref) == "PASS"      # the boundary is inclusive, as registered
    # And the reference is a PARAMETER: halve it and a run that passed now fails, which is what
    # stops the threshold from quietly becoming a literal.
    assert g3(0.00, 150.0, ref / 4) == "FAIL"


def test_g1_excludes_an_arm_judged_on_a_different_prompt_set():
    # An opponent run that lost prompts is not a weaker opponent, it is a smaller comparison, and
    # G3 cannot see it: a 400-prompt subset can have a perfectly ordinary empty rate and length.
    ok = [REF, dict(row("a", 0.40, 0.01, 0.05), g1="PASS")]
    assert h1_h2(ok)[0] == "STRENGTH SUPPORTED"
    bad = [REF, dict(row("a", 0.40, 0.01, 0.05), g1="FAIL")]
    assert h1_h2(bad)[0] == "NOT TESTED"
