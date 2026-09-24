"""The scorer-free cost column prices a different anchor than its accuracy columns.

serving_cost.P_ANCHOR is 1.7586 (TinyComma, the audited anchor); every accuracy in the judge-free
table is Comma-7B's, because that is the anchor that clears the floor on GSM8K and TriviaQA. Drawing
is linear in the anchor, so the majority-vote cells -- whose whole cost IS the anchor -- are 2.59x
cheaper in the table than in the system that produced the accuracies, while the reward cells barely
move because the 7.6B scorer dominates them either way.

The ordering survives, so the paper's claims survive; the SIZE of the scorer-free saving does not,
and that size is quoted. Caution (ae)/(v): the number is correctly computed and is not the quantity
the row implies -- here in the direction that flatters the paper, which is the direction that must
be disclosed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.serving_cost import P_ANCHOR, P_RISKY, P_SCORER  # noqa: E402
from tests.manuscript import body  # noqa: E402

P_COMMA7B = 7.0     # common-pile/comma-v0.1-2t, the anchor the judge-free accuracies come from


def _cost(n, pf, pa):
    return n * (pa + pf) / (pa + P_RISKY)


def test_the_two_bases_are_what_the_appendix_says_they_are():
    assert abs(P_ANCHOR - 1.7586) < 1e-6, P_ANCHOR
    assert abs(_cost(32, 0.0, P_ANCHOR) - 5.75) < 5e-3
    assert abs(_cost(32, 0.0, P_COMMA7B) - 14.90) < 5e-3
    assert abs(_cost(64, P_SCORER, P_ANCHOR) - 61.29) < 5e-3
    assert abs(_cost(64, P_SCORER, P_COMMA7B) - 62.23) < 5e-3


def test_the_majority_vote_cells_move_by_the_disclosed_factor():
    f = _cost(32, 0.0, P_COMMA7B) / _cost(32, 0.0, P_ANCHOR)
    assert abs(f - 2.59) < 5e-3, f
    # the reward cells barely move -- the scorer dominates them at either anchor
    g = _cost(64, P_SCORER, P_COMMA7B) / _cost(64, P_SCORER, P_ANCHOR)
    assert g < 1.05, g


def test_the_matched_n_saving_is_not_anchor_free():
    a = _cost(32, 0.0, P_ANCHOR) / _cost(32, P_SCORER, P_ANCHOR)
    c = _cost(32, 0.0, P_COMMA7B) / _cost(32, P_SCORER, P_COMMA7B)
    assert abs(a - 0.188) < 5e-4, a
    assert abs(c - 0.479) < 5e-4, c
    assert abs(c / a - 2.55) < 0.02, c / a


def test_the_ordering_survives_at_the_measured_anchor():
    """Why the paper's claims are unaffected: both rules draw from the same anchor."""
    mv = {n: _cost(n, 0.0, P_COMMA7B) for n in (8, 16, 32, 64)}
    rw = {n: _cost(n, P_SCORER, P_COMMA7B) for n in (2, 4, 8, 64)}
    # every table-listed majority-vote cell is cheaper than the best reward cell, and more accurate
    acc_mv = {8: 0.466, 16: 0.500, 32: 0.546, 64: 0.542}
    best_rw_acc = 0.386
    for n, c in mv.items():
        assert c < rw[64], (n, c, rw[64])
        assert acc_mv[n] > best_rw_acc, (n, acc_mv[n])


def test_the_appendix_discloses_the_mismatch_and_both_figures():
    txt = body("appendix_selection.tex")
    # v10 (2026-09-24) words the mismatch "The cost column prices the audited $1.8$B anchor while the
    # accuracies are Comma-7B's" (v9: "... are not the same system"); either spelling is the disclosure.
    assert ("not the same system" in txt or
            "The cost column prices the audited $1.8$B anchor while the accuracies are Comma-7B's" in txt), \
        "the anchor mismatch disclosure was trimmed"
    # match the text as written: 62.23 sits inside the same math group as 61.29, so it has no
    # "$" of its own -- the delimiter trap that hid a defect from an earlier guard this session.
    assert "$14.90\\times$" in txt, "the corrected majority-vote cost was trimmed"
    assert "61.29\\times \\to 62.23" in txt, "the reward-cell comparison was trimmed"
    assert "$47.9\\%$" in txt and "$18.8\\%$" in txt, "both matched-n figures must be reported"
    assert "direction that flatters the scorer-free rule" in txt, \
        "the disclosure must say which direction the error ran in"
