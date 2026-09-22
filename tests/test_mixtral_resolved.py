"""feat-171: Mixtral's exception is bounded, not absent -- a weakening the paper commits to.

The TIGHT ZERO sentence was fixed in the pre-registration before any of it ran, and it makes the
paper's claim smaller: it says how large the exception is instead of only that it exists. That is
exactly the kind of sentence caution (ag) says a length edit deletes first, and it survived two
page-budget rewrites of this very paragraph, so it is pinned to the CSV that produced it.
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import manuscript as M  # noqa: E402


def _d3(tag):
    path = os.path.join(ROOT, "results", f"order_averaged_h2h__{tag}.csv")
    rows = [r for r in csv.reader(open(path, encoding="utf-8")) if r and r[0].startswith("D3")]
    assert len(rows) == 1, tag
    return float(rows[0][2]), float(rows[0][3]), float(rows[0][4])


def test_mixtral_reads_tight_zero_and_the_band_still_qualifies():
    """The verdict is a STRING; the band it names is arithmetic. Assert the arithmetic, so the
    label cannot outlive a number that moved (caution (av))."""
    g, lo, hi = _d3("armc_mixtral")
    assert lo < 0 < hi, f"Mixtral's interval no longer contains zero ({lo:+.4f}, {hi:+.4f})"
    half = (hi - lo) / 2
    assert half < 0.030, f"half-width {half:.4f} is no longer below the committed 0.030"
    assert g > 0, "Mixtral's point estimate changed sign; 'same sign as every judge' is stale"
    # Bounded below the reading on record, which is the substance of "bounded".
    assert hi < 0.0645, f"upper end {hi:+.4f} is no longer below the +0.0645 on record"


def test_the_body_adopts_the_committed_tight_zero_sentence():
    txt = M.body("experiments.tex")
    assert "replicates under four judges and is bounded below $0.03$ under the one clean" in txt, \
        "the committed TIGHT ZERO sentence is not in the body"
    # The withdrawn framing must be gone: it said only that an exception exists.
    assert "four of five} put its interval clear of zero" not in txt, \
        "the superseded 'four of five' framing is back; the exception is now bounded, not open"


def test_the_appendix_carries_the_numbers_and_why_two_arms_were_needed():
    txt = M.body("appendix_selection.tex")
    g, lo, hi = _d3("armc_mixtral")
    assert f"${g:+.4f}$ $[{lo:+.4f}, {hi:+.4f}]$" in txt, "Mixtral's band left the appendix"
    assert "$+0.0090$ $[-0.0355, +0.0530]$" in txt, "the reading on record left the appendix"
    assert "a half-width of $0.0443$" in txt, \
        "the interval that the extra prompts tightened was trimmed -- it is what separates a "
    assert "AlpacaEval, where the reversal does not hold" in txt, \
        "the appendix no longer says why the first two attempts could not be read"
    # The disagreement is about the metered arm; that is the counter-intuitive half.
    b_met = [r for r in csv.reader(open(os.path.join(
        ROOT, "results", "order_averaged_h2h__wscope_c.csv"), encoding="utf-8"))
        if r and r[0].startswith("D2")][0]
    m_met = [r for r in csv.reader(open(os.path.join(
        ROOT, "results", "order_averaged_h2h__armc_mixtral.csv"), encoding="utf-8"))
        if r and r[0].startswith("D2")][0]
    assert float(m_met[2]) > float(b_met[2]), \
        "Mixtral no longer scores the meter above judge B; the appendix's explanation is stale"
    assert f"${float(m_met[2]):+.4f}$" in txt and f"${float(b_met[2]):+.4f}$" in txt, \
        "the two metered readings that locate the disagreement were trimmed"
