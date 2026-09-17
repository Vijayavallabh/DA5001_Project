"""The contaminated-anchor amplification range is a min/max over a set -- caution (ai)'s class.

"across twelve deliberately contaminated anchors the realised amplification at n = 64 runs between
1.0 and 4.0, against the 64 the certificate permits" is three claims about a set: the anchor count,
the endpoints, and that every value is under the certificate. Rebuild all three from the CSV.

This one IS a realisation, unlike the selection bounds guarded in test_selection_claims.py:
analysis/contaminated_anchor.py computes rate / base_rate for a named event. The column is empty for
a clean anchor because the anchor's own base rate is 0 there, which is why the clean arms have a
bound and no measured amplification.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import body  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rows():
    p = os.path.join(ROOT, "results", "contaminated_anchor.csv")
    return list(csv.DictReader(open(p, encoding="utf-8")))


def _measured_at_64():
    return [r for r in _rows() if int(float(r["n"])) == 64 and r["amplification"] not in ("", None)]


def test_twelve_distinct_contaminated_anchors():
    assert len({r["anchor"] for r in _rows()}) == 12, sorted({r["anchor"] for r in _rows()})


def test_the_quoted_endpoints_are_the_actual_min_and_max():
    vals = [float(r["amplification"]) for r in _measured_at_64()]
    assert vals, "no measured amplification at n=64"
    assert abs(min(vals) - 1.0) < 5e-4, min(vals)
    assert abs(max(vals) - 4.0) < 5e-4, max(vals)


def test_every_measured_amplification_is_under_the_certificate():
    """The claim is 'against the 64 the certificate permits' -- so none may exceed 64."""
    for r in _measured_at_64():
        assert float(r["amplification"]) <= 64.0 + 1e-9, r
        assert r["prop4_holds"] == "yes", r


def test_the_intro_quotes_the_range_and_the_anchor_count():
    txt = body("iclr_intro.tex")
    assert "twelve deliberately contaminated anchors" in txt, "the anchor count was trimmed"
    assert "$1.0$ and $4.0$" in txt, "the measured amplification range was trimmed or reworded"
