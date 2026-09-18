"""rho = +0.543 is over the TWO-JUDGE MEAN, and the figure beside it shows judge B alone.

Section 3 said "Across the six admissible, the gain does not track the anchor's own level
(rho = +0.543)" next to a figure captioned "six anchors, 500 in-house prompts, judge B". A reviewer
recomputing it from what the paper shows gets **-0.029**, because the committed number is Spearman
over the mean of both judges (the six-anchor scoring log's C2, band (-0.6, +0.6)). The number was
right; the quantity was not named. Same class as cautions (v), (aa), (ae) and (ah).

This test recomputes rho BOTH ways from selection_breadth.csv, so the value and its quantity are
pinned together and the judge-B value is on record as the thing it is not.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import body  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JUDGE_B = "Phi-3.5-mini-instruct"


def _spearman(a, b):
    def rank(v):
        s = sorted(v)
        return [s.index(x) + 1 for x in v]
    ra, rb = rank(a), rank(b)
    assert len(set(a)) == len(a) and len(set(b)) == len(b), "ties -- this rank rule is wrong for ties"
    n = len(a)
    d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


def _rows():
    return list(csv.DictReader(open(os.path.join(ROOT, "results", "selection_breadth.csv"),
                                    encoding="utf-8")))


def _pairs(mean_of_both):
    by = {}
    for r in _rows():
        by.setdefault(r["anchor"], []).append(r)
    assert len(by) == 6, sorted(by)
    xs, ys = [], []
    for anchor, rs in by.items():
        if mean_of_both:
            xs.append(sum(float(r["u_n1"]) for r in rs) / len(rs))
            ys.append(sum(float(r["gain"]) for r in rs) / len(rs))
        else:
            r = next(r for r in rs if JUDGE_B in r["judge"])
            xs.append(float(r["u_n1"]))
            ys.append(float(r["gain"]))
    return xs, ys


def test_rho_reproduces_on_the_two_judge_mean():
    rho = _spearman(*_pairs(mean_of_both=True))
    assert abs(rho - 0.543) < 5e-4, (rho, "the quoted rho no longer rounds from the CSV")
    assert -0.6 < rho < 0.6, "the six-anchor pre-registration's NO TREND band"


def test_under_judge_B_alone_it_is_a_different_number_which_is_why_the_quantity_is_named():
    rho = _spearman(*_pairs(mean_of_both=False))
    assert abs(rho - 0.543) > 0.1, (rho, "judge B now agrees; the qualifier could be revisited")


def test_the_manuscript_names_the_quantity():
    txt = body("experiments.tex")
    assert "($\\rho = +0.543$ on the two-judge mean)" in txt, \
        "rho lost the quantity it is computed over; the figure beside it shows judge B, where it is -0.029"
