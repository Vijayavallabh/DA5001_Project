"""The judged floor is a re-run at a DIFFERENT n-grid, not an identical configuration.

Section 3 said "an identical configuration re-run moves one by about 0.04 (+0.111 and +0.072 at
Comma-7B)". The two arms behind those numbers used different grids -- selection_scaling_comma7b.csv
runs 1,2,4,8 and selection_scaling_comma7b64.csv runs 1,2,4,8,16,32,64 -- and fresh generations, so
the configuration was not identical in the sense the word carries.

That matters now because feat-129 isolated the mechanism cleanly: the presentation flip is drawn
once per element of a set that grows with the grid, so a judged single-order level is grid-dependent
(caution (ap)). With byte-identical generations and only the grid changed, the audited anchor's n=64
gain moved 0.066. The range across both measurements is 0.039 to 0.066, which the paper now quotes
as 0.04 to 0.07.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import body  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JUDGE_B = "Phi-3.5-mini-instruct"


def _grid_and_gain(name, n=8):
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "results", name), encoding="utf-8"))
            if JUDGE_B in r["judge"]]
    grid = sorted(int(float(r["n"])) for r in rows)
    gain = next(float(r["gain"]) for r in rows if int(float(r["n"])) == n)
    return grid, gain


def test_the_two_comma7b_arms_used_different_grids():
    g8, v8 = _grid_and_gain("selection_scaling_comma7b.csv")
    g64, v64 = _grid_and_gain("selection_scaling_comma7b64.csv")
    assert g8 == [1, 2, 4, 8], g8
    assert g64 == [1, 2, 4, 8, 16, 32, 64], g64
    assert g8 != g64, "the grids now match; the wording may be revisited"
    assert abs(v8 - 0.111) < 5e-4 and abs(v64 - 0.072) < 5e-4, (v8, v64)
    assert abs(abs(v8 - v64) - 0.039) < 1e-3, abs(v8 - v64)


def test_the_range_quoted_covers_both_measurements():
    """0.039 at Comma-7B and 0.066 at the audited anchor -> the paper says 0.04 to 0.07."""
    _, v8 = _grid_and_gain("selection_scaling_comma7b.csv")
    _, v64 = _grid_and_gain("selection_scaling_comma7b64.csv")
    lo = abs(v8 - v64)
    a = next(float(r["gain"]) for r in csv.DictReader(
        open(os.path.join(ROOT, "results", "selection_scaling.csv"), encoding="utf-8"))
        if JUDGE_B in r["judge"] and int(float(r["n"])) == 64)
    b = next(float(r["gain"]) for r in csv.DictReader(
        open(os.path.join(ROOT, "results", "selection_scaling_n128.csv"), encoding="utf-8"))
        if JUDGE_B in r["judge"] and int(float(r["n"])) == 64)
    hi = abs(a - b)
    assert 0.035 < lo < 0.045, lo
    assert 0.06 < hi < 0.07, hi


def test_the_manuscript_no_longer_calls_it_an_identical_configuration():
    txt = body("experiments.tex")
    assert "identical configuration re-run" not in txt, \
        "the two arms behind this number used different n-grids and fresh generations"
    assert "a re-run at a different $n$-grid moves one by $0.04$ to $0.07$" in txt, \
        "the corrected judged-floor wording was lost"
