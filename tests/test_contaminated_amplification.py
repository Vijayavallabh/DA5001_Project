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
    """feat-179's corrected-selector arm. contaminated_anchor.csv was measured with a selector that
    ranked partly on padding (caution (ba)); its n > 1 numbers are no longer quoted anywhere."""
    p = os.path.join(ROOT, "results", "selector_n256.csv")
    return list(csv.DictReader(open(p, encoding="utf-8")))


def _measured_at_64():
    return [r for r in _rows() if int(float(r["n"])) == 64 and r["amplification_vs_n1"] not in ("", None)]


def test_twelve_distinct_contaminated_anchors():
    assert len({r["anchor"] for r in _rows()}) == 12, sorted({r["anchor"] for r in _rows()})


def test_the_quoted_endpoints_are_the_actual_min_and_max():
    vals = [float(r["amplification_vs_n1"]) for r in _measured_at_64()]
    assert vals, "no measured amplification at n=64"
    lo, hi = min(vals), max(vals)
    for f in ("iclr_intro.tex", "selection.tex", "experiments.tex"):
        txt = body(f)
        assert f"${lo:.1f}$" in txt and f"${hi:.1f}$" in txt, (f, lo, hi)


def test_every_measured_amplification_is_under_the_certificate():
    """The claim is 'against the 64 the certificate permits' -- so none may exceed 64."""
    for r in _rows():
        if r["amplification_vs_n1"]:
            assert float(r["amplification_vs_n1"]) <= int(r["n"]) + 1e-9, r


def test_the_intro_quotes_the_range_and_the_anchor_count():
    txt = body("iclr_intro.tex")
    assert "twelve deliberately contaminated anchors" in txt, "the anchor count was trimmed"
    vals = [float(r["amplification_vs_n1"]) for r in _measured_at_64()]
    assert f"${min(vals):.1f}$ and ${max(vals):.1f}$" in txt, \
        "the measured amplification range was trimmed or reworded"
