"""The paper's two claims about reward overoptimisation must not contradict each other.

Appendix I said "the reward overoptimisation that turns such curves over is not observed anywhere
on the grid" and, 580 lines later IN THE SAME FILE, its own judge-free table reported the pointwise
reward's TriviaQA accuracy falling with n to an interval excluding zero at n=16. Both sentences were
true of their own arm and the first was false as written, because "anywhere on the grid" is not
scoped to the arm it is about.

This is caution (ai) exactly: a number is checked against its CSV, and the CLAIM ABOUT a set of
numbers is checked against nothing. So rebuild both series from their CSVs and assert the paper's
adjectives match the shapes -- monotone where it says monotone, turning over where it says turning
over -- and that the absence claim carries its qualifier.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import DIR, tex  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JUDGE_B = "Phi-3.5-mini-instruct"


def _rows(name):
    p = os.path.join(ROOT, "results", name)
    assert os.path.exists(p), p
    return list(csv.DictReader(open(p, encoding="utf-8")))


def _app():
    return " ".join(open(tex("sections/appendix_selection.tex"), encoding="utf-8").read().split())


def test_the_judged_comma7b_grid_really_does_not_turn_over():
    """The 'still climbing, no overoptimisation' half, rebuilt as a SERIES from its CSV."""
    rows = [r for r in _rows("selection_scaling_comma7b64.csv") if JUDGE_B in r["judge"]]
    series = [(int(float(r["n"])), float(r["gain"])) for r in rows]
    series.sort()
    gains = [g for _, g in series]
    assert len(gains) >= 7, series
    assert gains == sorted(gains), f"the judged grid is NOT monotone: {series}"
    # and nothing on it is negative with its interval on the wrong side
    for r in rows:
        assert float(r["gain_hi95"]) >= 0, ("a judged arm turns over", r["n"], r["gain_hi95"])


def test_the_judge_free_pointwise_reward_really_does_turn_over():
    """The other half. If this ever stops being true the appendix paragraph must change."""
    rows = [r for r in _rows("selection_verifiable_tqa_comma7b.csv")
            if r["arm"].startswith("pointwise reward")]
    bad = [r for r in rows if float(r["gain_hi95"]) < 0]
    assert bad, "TriviaQA's pointwise reward no longer turns over -- the appendix claim is stale"
    assert any(int(float(r["n"])) == 16 for r in bad), \
        f"n=16 is the one the appendix names; turnover is now at {[r['n'] for r in bad]}"
    sp = {float(r["spearman_acc_logn"]) for r in rows}
    assert all(s < 0 for s in sp), sp


def test_the_absence_claim_is_scoped_and_the_contradiction_is_named():
    """An unqualified 'not observed anywhere' is the defect; the qualifier is the repair."""
    txt = _app()
    assert "not observed anywhere on the grid" not in txt, \
        "the unqualified absence claim is back, and the paper's own TriviaQA table refutes it"
    assert "anywhere on \\emph{this} grid" in txt, "the absence claim lost its scope"
    # and the place it IS observed must still be pointed at from that paragraph
    head = txt[:txt.find("The arm's own nested check")]
    assert "TriviaQA" in head and "tab:judgefree" in head, \
        "the scoped claim no longer says where overoptimisation does occur"


def _judgefree_caption():
    """The tab:judgefree caption ALONE.

    Scoped deliberately. The first version of this check asked whether the phrase appeared
    anywhere in the file, and the scoped-claim paragraph added above now contains the same phrase
    -- so deleting it from the caption left the check passing on the other occurrence. A guard
    satisfied by a different sentence than the one it is about is not guarding that sentence.
    """
    txt = _app()
    i = txt.find("\\label{tab:judgefree}")
    assert i != -1, "the judge-free table lost its label"
    j = txt.rfind("\\caption{", 0, i)
    assert j != -1, "the judge-free table lost its caption"
    return txt[j:i]


def test_the_judge_free_table_still_reports_the_turnover():
    """Caution (ag): a concession is the first thing a length edit deletes."""
    cap = _judgefree_caption()
    assert "excludes zero on the wrong side" in cap, \
        "the turnover concession was trimmed from the judge-free caption"
    assert "$-0.607$" in cap, "the TriviaQA Spearman was trimmed from the judge-free caption"
    txt = _app()
    rows = [r for r in _rows("selection_verifiable_tqa_comma7b.csv")
            if r["arm"].startswith("pointwise reward")]
    sp = float(rows[0]["spearman_acc_logn"])
    assert abs(sp - (-0.607)) < 5e-4, (sp, "the quoted Spearman no longer rounds from the CSV")
