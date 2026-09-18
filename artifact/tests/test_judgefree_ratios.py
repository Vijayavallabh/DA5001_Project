"""Ratios between the two judge-free rules move with n, so every quoted one must name its n.

The closing said "the scorer binds, gaining 3.4x less than majority vote on GSM8K" with no n. The
ratio over the grid is 3.00, 3.32, 5.00, 3.77, 3.36 at n = 4, 8, 16, 32, 64 -- so a reviewer
recomputing it at the matched n=32 the appendix uses gets 3.77 and at n=16 gets 5.00. Same class as
the rho defect: the number is right and a reader cannot tell which quantity it is.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import body  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _gains(name):
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results", name), encoding="utf-8")))
    mv = {int(float(r["n"])): float(r["gain"]) for r in rows if r["arm"].startswith("majority")}
    pw = {int(float(r["n"])): float(r["gain"]) for r in rows if r["arm"].startswith("pointwise")}
    return mv, pw


def test_the_gsm8k_ratio_is_quoted_at_the_n_it_is_computed_at():
    mv, pw = _gains("selection_verifiable_comma7b.csv")
    assert abs(mv[64] / pw[64] - 3.4) < 0.05, (mv[64] / pw[64], "n=64 ratio moved")
    assert abs(mv[32] / pw[32] - 3.77) < 0.01, (mv[32] / pw[32], "the appendix's matched n=32 ratio moved")
    txt = body("iclr_closing.tex")
    assert "$3.4\\times$ less than majority vote on GSM8K at $n=64$" in txt, \
        "the closing's GSM8K ratio lost the n it is computed at; it reads 3.77 at n=32 and 5.00 at n=16"


def test_the_ratio_really_does_move_with_n_which_is_why_the_n_is_required():
    mv, pw = _gains("selection_verifiable_comma7b.csv")
    rs = [mv[n] / pw[n] for n in (4, 8, 16, 32, 64)]
    assert max(rs) - min(rs) > 1.0, (rs, "the ratio is now flat in n; the qualifier could be revisited")


def test_the_closing_levels_reproduce():
    """TriviaQA: selection 0.190 at n=64, metered 0.618 only at k=20, certified 480 nats."""
    sel = list(csv.DictReader(open(os.path.join(ROOT, "results", "verifiable_metered_tqa.csv"),
                                   encoding="utf-8")))
    s64 = next(r for r in sel if r["mechanism"].startswith("selection") and r["arm"] == "n=64")
    assert abs(float(s64["acc"]) - 0.190) < 5e-4, s64["acc"]
    m20 = next(r for r in sel if r["mechanism"].startswith("metered") and r["arm"] == "k=20")
    assert abs(float(m20["acc"]) - 0.618) < 5e-4, m20["acc"]
    assert abs(float(m20["certificate_nats"]) - 480.0) < 1e-9, m20["certificate_nats"]
    # it IS the risky model there -- identical accuracy to the k=-1 baseline
    mm1 = next(r for r in sel if r["mechanism"].startswith("metered") and r["arm"] == "k=-1")
    assert float(m20["acc"]) == float(mm1["acc"]), (m20["acc"], mm1["acc"])
    txt = body("iclr_closing.tex")
    for v in ("$0.618$", "$0.190$", "$480$ nats", "$24$-token"):
        assert v in txt, f"the closing lost {v}"
