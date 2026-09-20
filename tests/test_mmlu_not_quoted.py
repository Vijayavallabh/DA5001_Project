"""The MMLU-at-TinyComma arm read BELOW CHANCE at its registered gate, so no result from it may
appear in the manuscript (results/onset_prediction_mmlu_rescore.md, H4).

feat-132's precedent: when a gate fails, its numbers must not leak into the paper later, and the
cheapest enforcement is a test that fails if they do.

WHY THIS GUARDS CONTEXT AND NOT VALUES. The first version compared every accuracy and interval end
against the manuscript text and fired on `0.154`, `0.168`, `0.0500`, `0.630` -- none of them this
arm's, all of them legitimate numbers for unrelated quantities elsewhere in the paper (`0.168` is
judge F's selection gain). Filtering by "does this literal occur in another committed CSV" did not
fix it either, because the CSVs store `0.168` and the paper prints `0.1680`. A bare value is simply
not evidence of provenance -- which is the weakness audit_numbers.py has always had (caution (aq)).

The commitment is precise and so is its guard: **the paper may not attribute an MMLU result to the
audited anchor.** A future arm at a capable anchor (results/onset_prediction_mmlu_comma7b.md) may
legitimately put MMLU in the paper, and this permits that.
"""
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import DIR, body  # noqa: E402

CSV = "results/verifiable_metered_mmlu.csv"
ANCHOR = re.compile(r"TinyComma|tinycomma", re.I)


def _live_sections():
    d = os.path.join(DIR, "sections")
    out = [f for f in sorted(os.listdir(d))
           if f.endswith(".tex") and not re.search(r"_v\d+_\d{4}-\d{2}-\d{2}\.tex$", f)]
    assert out, "no live section files found; this guard must not pass by never running"
    return out


def _sentences(txt):
    return re.split(r"(?<=[.!?])\s+", txt)


def test_the_gate_still_reads_below_chance():
    """If this stops being true the arm's status changed and H4 must be re-read before quoting."""
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    k0 = [r for r in rows if r["mechanism"] == "metered decoder" and r["arm"] == "k=0"]
    assert len(k0) == 1, CSV
    assert float(k0[0]["acc_hi95"]) < 0.25, \
        "the anchor's n=1 interval no longer lies below chance; re-read H4 before quoting anything"


def test_the_paper_never_attributes_an_mmlu_result_to_the_audited_anchor():
    bad = [s for s in _sentences(body(*_live_sections()))
           if "MMLU" in s and ANCHOR.search(s)]
    assert not bad, ("the invalidated MMLU-at-TinyComma arm reached the manuscript:\n  "
                     + "\n  ".join(s[:150] for s in bad[:3]))


def test_the_guard_can_actually_fire():
    """Caution (p): a gate that fires on nothing is not a gate."""
    fake = "Selection lifts MMLU at TinyComma-1.8B from 0.188 to 0.300."
    assert [s for s in _sentences(fake) if "MMLU" in s and ANCHOR.search(s)]
