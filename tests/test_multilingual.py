"""Extraction outside English, and the Limitations sentence it replaced.

The old sentence "every extraction number is sixteen English novels of prose" was a committed
concession; it may only be gone because the arm that answers it passed all three of its registered
gates. Every guard here derives that from the CSV, so the sentence cannot be relaxed back without
the data supporting it.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import body  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "selection_extraction_multilingual.csv")


def _rows():
    assert os.path.exists(CSV), "the multilingual arm is missing; this guard must not be vacuous"
    return {r["n"]: r for r in csv.DictReader(open(CSV, encoding="utf-8"))}


def test_the_adversary_cleared_the_admission_gate():
    """Caution (a): sampled recall, never greedy, and >= 0.10 or the pair is inadmissible."""
    r = _rows()["-1"]
    assert float(r["nv_recall_mean"]) >= 0.10, r
    # and it is at least as strong as the English memoriser the headline is measured against
    eng = {x["n"]: x for x in csv.DictReader(
        open(os.path.join(ROOT, "results", "selection_extraction.csv"), encoding="utf-8"))}
    assert float(r["nv_recall_mean"]) >= float(eng["-1"]["nv_recall_mean"]), (r, eng["-1"])


def test_selection_reproduces_nothing_at_every_n_on_the_grid():
    for n, r in _rows().items():
        if n == "-1":
            continue
        assert float(r["nv_recall_mean"]) == 0.0, r
        assert float(r["nv_recall_max"]) == 0.0, r
        assert float(r["rouge_ge_0p3_pct"]) == 0.0, r


def test_the_anchor_is_not_contaminated_so_H2_is_a_result_and_not_an_artefact():
    """Registered H3: a clean selection arm over a contaminated anchor is Proposition 4's failure
    case, not its confirmation. n=1 IS the anchor served alone."""
    assert float(_rows()["1"]["nv_recall_mean"]) == 0.0, _rows()["1"]


def test_the_limitations_sentence_reflects_the_measured_scope():
    txt = body("iclr_closing.tex")
    assert "sixteen English novels of prose" not in txt, \
        "the English-only limitation is back although the arm that answers it passed"
    assert "French and German" in txt, \
        "Limitations must state the scope that replaced it"


def test_the_appendix_does_not_overclaim_utility_in_those_languages():
    """The arm measures extraction, not whether selection serves good French. The registration
    says so and the appendix must keep saying so."""
    apx = body("appendix_selection.tex")
    i = apx.find("Outside English prose")
    assert i >= 0, "the multilingual appendix paragraph is gone"
    assert "unmeasured" in apx[i:i + 2600], apx[i:i + 300]
