"""feat-108: the gate that refused to let a table be built.

The paper would be stronger with one leakage table covering both mechanisms on one passage set.
It cannot have one: the two pipelines' shared control differs by more than the registered 0.02
even after the generation lengths are matched. What this file pins is the refusal -- that no
cross-arm ratio appears, and that the residual stays disclosed -- because the temptation to quote
it comes back every time someone reads the two CSVs side by side.
"""
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import tex  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "results", "h2h", "composition_summary.csv")
B = os.path.join(ROOT, "results", "leakage_len300.csv")
LOG = os.path.join(ROOT, "results", "onset_prediction_leakage_headtohead.md")
BAND = 0.02


def controls():
    a = next(r for r in csv.DictReader(open(A)) if float(r["k"]) == -1.0)
    b = next(r for r in csv.DictReader(open(B)) if r["n"] == "-1")
    return float(a["nv_recall_mean"]), float(b["nv_recall_mean"])


def test_the_gate_still_reads_unmatched():
    """If a rerun ever brought these within 0.02, the joint table becomes allowed and Appendix J's
    paragraph becomes false. Either way this is the thing to check first."""
    a, b = controls()
    assert abs(a - b) > BAND, (a, b, "the controls now agree; G0 would read MATCHED")


def test_the_length_hypothesis_moved_it_halfway_and_that_is_what_is_claimed():
    """The 200-token arm read 0.3925 and the 300-token arm reads 0.4428, closing about half the
    gap. Appendix J says 'half'; if the numbers move, so must the word."""
    a, b = controls()
    old = next(r for r in csv.DictReader(open(os.path.join(
        ROOT, "results", "selection_extraction.csv"))) if r["n"] == "-1")
    closed = b - float(old["nv_recall_mean"])
    assert 0.4 < closed / (a - float(old["nv_recall_mean"])) < 0.6, closed


def test_no_cross_arm_ratio_reaches_the_manuscript():
    """The excluded alternative, enforced. A sentence dividing one mechanism's leakage by the
    other's is exactly what G0 refused to license."""
    for f in ("sections/experiments.tex", "sections/appendix_limitations.tex",
              "sections/appendix_selection.tex", "sections/iclr_closing.tex"):
        body = " ".join(open(tex(f), encoding="utf-8").read().split())
        assert not re.search(r"leaks \$0\.4\d+\$ where selection", body), f
        assert "0.4759" not in body, (f, "the metered arm's recall is quoted across pipelines")


def test_appendix_j_discloses_the_residual_and_both_candidate_causes():
    body = " ".join(open(tex("sections/appendix_limitations.tex"), encoding="utf-8").read().split())
    a, b = controls()
    # v10 (2026-09-24) prints the pair as "$0.4921$ against $0.4428$" and opens the next sentence
    # with "No joint table is built"; the connector and the capital are the only changes.
    assert f"${a:.4f}$ against ${b:.4f}$" in body, (a, b)
    assert f"residual of ${abs(a - b):.4f}$" in body, abs(a - b)
    assert "no joint table is built" in body.lower()
    assert "changing the token budget changes the draw" in body, "cause 1 is not disclosed"
    assert "flat $300$" in body, "cause 2 is not disclosed"


def test_the_scoring_log_reports_the_gate_failing_rather_than_the_arm_succeeding():
    t = " ".join(open(LOG, encoding="utf-8").read().split())
    assert "UNMATCHED" in t and "no joint table is built" in t
    assert "half right" in t or "half" in t
    assert "0.0493" in t
