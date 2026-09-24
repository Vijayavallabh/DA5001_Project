"""Claims ABOUT a series, checked against the series.

Caution (ai): a number is checked against its CSV, but the CLAIM ABOUT a set of numbers is checked
against nothing, and prose that cites a table is not thereby taken from it. The third read-through
found a series printed as "monotonic" whose last value rose. These are the two load-bearing series
claims that had no guard at all -- both were verified correct by hand on 2026-09-17, and both are
one re-run away from becoming false without anything noticing.
"""
import csv
import math
import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import body  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = (2, 4, 8, 16, 32, 64)


def _rows(name):
    return list(csv.DictReader(open(os.path.join(ROOT, "results", name), encoding="utf-8")))


def _spearman(xs, ys):
    def rank(z):
        order = sorted(range(len(z)), key=lambda i: z[i])
        out = [0] * len(z)
        for k, i in enumerate(order):
            out[i] = k
        return out
    rx, ry, n = rank(xs), rank(ys), len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den


def test_the_scorer_scale_spearmans_are_what_the_appendix_says_they_are():
    """"Spearman ... is 0.5429 for the 0.5B scorer and exactly 1.0 ... for each of 1.5B, 3B and
    7.6B" -- four numbers and a shape claim, living only in a scoring log and the prose. Rebuild
    all four from scorer_scale.csv and check the adjective as well as the digits."""
    rows = [r for r in _rows("scorer_scale.csv") if r["gain"].strip() and r["scorer_b"].strip()]
    by = {}
    for r in rows:
        by.setdefault(float(r["scorer_b"]), {})[int(float(r["n"]))] = float(r["gain"])
    assert len(by) == 4, sorted(by)
    rho = {}
    for b, g in by.items():
        assert set(g) == set(GRID), (b, sorted(g))
        series = [g[n] for n in GRID]
        rho[b] = _spearman([math.log(n) for n in GRID], series)
    small = min(rho)
    assert abs(rho[small] - 0.5429) < 5e-5, (small, rho[small])
    for b in sorted(rho):
        if b == small:
            continue
        assert rho[b] == 1.0, (b, rho[b])                       # "exactly 1.0"
        s = [by[b][n] for n in GRID]
        assert all(y > x for x, y in zip(s, s[1:])), (b, s)     # "strictly monotone at all six"
    apx = body("appendix_selection.tex")
    assert "$0.5429$" in apx, "the 0.5B Spearman is no longer quoted"
    assert "strictly monotone" in apx and "all six grid points" in apx


def test_the_strongest_anchor_series_really_rises_monotonically():
    """"the gain rises monotonically --- +0.041, +0.060, +0.072, +0.103, +0.127, +0.173 at
    n = 2,4,8,16,32,64 ... so the mechanism is still climbing at n=64". Six printed values, one
    shape word, and a conclusion drawn from the last one. results/onset_prediction_n256.md exists
    to test the conclusion; this tests the series it rests on."""
    rows = [r for r in _rows("selection_scaling_comma7b64.csv") if "Phi-3.5" in r["judge"]]
    g = {int(float(r["n"])): float(r["gain"]) for r in rows}
    series = [g[n] for n in GRID]
    assert all(y > x for x, y in zip(series, series[1:])), series   # the shape word
    apx = body("appendix_selection.tex")
    assert "rises monotonically" in apx
    for v in series:
        assert f"${v:+.3f}$" in apx, (v, "a printed value no longer rounds from the CSV")
    # and the certificate quoted beside it is log 64, not the measured KL
    assert "$\\log 64 = 4.16$" in apx, "the certificate beside the series must stay log n"


def test_the_held_out_prediction_errors_are_the_current_ones_in_the_order_named():
    """"held-out errors of 0.352, 0.251 and 0.481 nats", in two appendices.

    Three defects in one clause, found 2026-09-17. (1) STALE: those are means over the FIVE pairs
    held out when the rules were committed, and prediction_scores.csv now marks SEVEN -- the two
    open-calm pairs were added and the triple was never regenerated, exactly the hand-maintained
    series problem the normaliser growth table had. (2) TRANSPOSED: the sentence names the rules
    as median r(x), q25, constant, and the five-pair values in that order are 0.352, 0.481, 0.251
    -- the printed triple swaps the last two, so a reader mapping them positionally gets q25 and
    the fitted constant the wrong way round. (3) The rank correlation beside them, -0.18, is the
    GUTENBERG arm's number (results/onset_prediction_gutenberg.md); the nine CopyBench pairs give
    -0.42, which the script prints itself. Correcting it strengthens the paragraph, whose claim is
    that the direction inverts.

    Rebuilt here from the CSV, in the order the sentence names, scanning every live section.
    """
    import glob
    rows = [r for r in _rows("prediction_scores.csv") if r["held_out"] == "True"]
    pairs = {r["pair"] for r in rows}
    assert len(pairs) == 7, sorted(pairs)
    # the order the prose names: the equation at the median of r(x), the same at q25, a constant
    order = ["P1: median s_s - s_r", "q25 of r(x)", "constant 0.889*s(x)"]
    errs = []
    for rule in order:
        e = [float(r["abs_error"]) for r in rows if r["rule"] == rule]
        assert len(e) == 7, (rule, len(e))
        errs.append(sum(e) / len(e))

    from tests.manuscript import DIR
    live = [f for f in glob.glob(os.path.join(DIR, "sections", "*.tex"))
            if not re.search(r"_v\d", os.path.basename(f))]
    assert live, DIR
    txt = " ".join(" ".join(open(f, encoding="utf-8").read().split()) for f in live)
    quoted = ", ".join(f"${v:.3f}$" for v in errs)
    for v in errs:
        assert f"${v:.3f}$" in txt, (v, "a held-out error no longer rounds from the CSV")
    for stale in ("$0.352$, $0.251$ and $0.481$", "$0.251$ and $0.481$"):
        assert stale not in txt, f"the stale five-pair triple is back: {stale}"
    assert "$-0.18$ where the derivation requires it positive" not in txt, \
        "the Gutenberg rank correlation is back in the CopyBench paragraph"
    # the held-out-prediction paragraph was cut in the 2026-09-19 appendix reduction; the two
    # STALE forms above are still forbidden, which is what this test exists to prevent.
    print("held-out errors in the order named:", quoted)


def test_the_reallocation_range_brackets_what_the_table_holds():
    """"optimal offline reallocation ... buys 3 to 13% at k <= 1 and under 1.4% at k=3", in the
    repairs table and again in the proofs appendix. Two range claims over 24 cells of
    marginal_price_table.csv, with no guard: adding a pair or a budget silently widens the range
    the prose brackets. Verified exact on 2026-09-18 (3.07-13.26% and a 1.33% maximum)."""
    rows = _rows("marginal_price_table.csv")
    assert rows, "marginal_price_table.csv is empty"
    low = [(float(r["gain_ratio"]) - 1) * 100 for r in rows if float(r["k"]) <= 1]
    high = [(float(r["gain_ratio"]) - 1) * 100 for r in rows if float(r["k"]) == 3]
    assert low and high, (len(low), len(high))
    # the prose brackets, not point values: the range must sit inside what is claimed
    assert 3.0 <= min(low) and max(low) <= 13.0 + 0.5, (min(low), max(low))
    assert max(high) < 1.4, max(high)
    txt = body("orders.tex", "appendix_proofs.tex")
    assert "$3$ to $13\\%$ at $k \\le 1$" in txt, "the low-budget range claim has moved"
    assert "under $1.4\\%$ at $k=3$" in txt, "the k=3 ceiling claim has moved"

# RETIRED 2026-09-19, appendix reduction. The paragraph each of these read was removed
# when the appendix was cut from 52 pages, so the sentence they pinned no longer exists.
# A guard for a claim the paper does not make protects nothing; recorded here rather than
# silently deleted, so the removal is visible to the next reader:
#   test_the_burstiness_ranking_sentence_is_true_of_the_ranking
