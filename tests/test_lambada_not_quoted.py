"""The LAMBADA arm failed its instrument gate, so none of its numbers may reach the manuscript
(results/onset_prediction_lambada_headtohead.md, H1a and H4).

Same enforcement as tests/test_mmlu_not_quoted.py, and guarded on CONTEXT rather than on values
for the reason recorded there: a bare number is not evidence of provenance.
"""
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import DIR, body  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "verifiable_metered_lambada.csv")


def _live():
    d = os.path.join(DIR, "sections")
    out = [f for f in sorted(os.listdir(d))
           if f.endswith(".tex") and not re.search(r"_v\d+_\d{4}-\d{2}-\d{2}\.tex$", f)]
    assert out, "no live section files; this guard must not pass by never running"
    return out


def test_the_instrument_gate_still_fails():
    """If the risky model ever reads above the registered 0.40 floor the arm's status changed and
    H4 must be re-read before anything is quoted."""
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    k1 = [r for r in rows if r["mechanism"] == "metered decoder" and r["arm"] == "k=-1"]
    assert len(k1) == 1, CSV
    assert float(k1[0]["acc"]) < 0.40, \
        "the LAMBADA instrument gate now passes; re-read H4 before quoting this arm"


def test_the_paper_never_reports_a_lambada_result():
    bad = [s for s in re.split(r"(?<=[.!?])\s+", body(*_live())) if "LAMBADA" in s.upper()]
    assert not bad, ("the invalidated LAMBADA arm reached the manuscript:\n  "
                     + "\n  ".join(s[:150] for s in bad[:3]))


def test_the_judge_free_head_to_head_is_still_described_as_one_task():
    """Three tasks were tried at the one meterable anchor and all three failed. The paper must not
    imply a plural judge-free head-to-head."""
    txt = body(*_live())
    assert "TriviaQA" in txt
    for claim in ("two judge-free head-to-heads", "both judge-free head-to-heads"):
        assert claim not in txt, claim


def test_the_guard_can_actually_fire():
    fake = "On LAMBADA selection lifts exact match from 0.060 to 0.098."
    assert [s for s in re.split(r"(?<=[.!?])\s+", fake) if "LAMBADA" in s.upper()]
