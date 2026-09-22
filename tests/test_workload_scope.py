"""feat-168: the headline reversal is scoped to workloads inside the anchor's support.

This is a concession -- it names a benchmark where the paper's own headline fails -- and
caution (ag) says a length edit deletes those first, while caution (aq) found six of eight
surviving deletion. So every number is derived from the CSV that measured it and the shape claims
are checked against the data rather than against their phrasing.
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import manuscript as M  # noqa: E402


def _d(name, prefix):
    path = os.path.join(ROOT, "results", f"order_averaged_h2h__{name}.csv")
    rows = [r for r in csv.reader(open(path, encoding="utf-8")) if r and r[0].startswith(prefix)]
    assert len(rows) == 1, f"{path}: {len(rows)} rows for {prefix}"
    return float(rows[0][2]), float(rows[0][3]), float(rows[0][4]), rows[0][7].strip()


def test_the_appendix_states_the_workload_where_the_reversal_fails():
    txt = M.body("appendix_selection.tex")
    got, lo, hi, reading = _d("mixpowk_judgeB", "D3")
    assert hi < 0, "D3 no longer excludes zero on the losing side; the concession is stale"
    assert reading == "REVERSAL REFUTED", reading
    # This appendix prints bands at 4 dp, as its neighbours do; carries_band matches the 3 dp
    # form the forest figure uses, so the literal is built here from the CSV instead.
    def band(g, lo_, hi_):
        return f"${g:+.4f}$ $[{lo_:+.4f}, {hi_:+.4f}]$"
    assert band(got, lo, hi) in txt, \
        f"the paired difference {band(got, lo, hi)} left the appendix"
    for prefix, arm in (("D1", "selection"), ("D2", "metered")):
        v, vlo, vhi, _ = _d("mixpowk_judgeB", prefix)
        assert band(v, vlo, vhi) in txt, \
            f"the {arm} gain on that workload ({band(v, vlo, vhi)}) left the appendix"
    assert "AlpacaEval" in txt, "the benchmark is no longer named"
    assert "support ceiling" in txt, "the mechanism of the loss is no longer named"


def test_the_body_scopes_the_claim_and_points_at_the_evidence():
    txt = M.body("experiments.tex")
    assert "not across opponents or workloads" in txt, \
        "the body's replication claim dropped the workload scope"
    assert "moving off the anchor's support" in txt, \
        "the body no longer says what kind of workload breaks it"


def test_the_discarded_first_pass_is_reported_not_buried():
    """The k=10 pass was discarded and saying so is what makes the k=1.0 choice auditable. Its two
    damning numbers -- the activity and the byte-identity rate -- must stay in the appendix."""
    txt = M.body("appendix_selection.tex")
    assert "794" in txt and "805" in txt, "the byte-identity count of the discarded pass was cut"
    assert "0.016" in txt.replace("\\%", "%"), "the discarded pass's activity rate was cut"
    assert "8.008" in txt.replace("\\%", "%"), "the chosen budget's activity rate was cut"


def test_the_calibration_curve_matches_its_csv_and_the_argmin_is_what_the_paper_says():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "mixpow_kcal.csv"),
                                    encoding="utf-8")))
    assert len(rows) == 5, len(rows)
    target = 261239 / 3118893
    best = min(rows, key=lambda r: abs(float(r["activity"]) - target))
    assert float(best["k"]) == 1.0, f"argmin moved to k={best['k']}; the appendix says 1.0"
    above = [r for r in rows if float(r["activity"]) > target]
    below = [r for r in rows if float(r["activity"]) < target]
    assert above and below, "the grid no longer brackets the target, so the choice is an endpoint"
    txt = M.body("appendix_selection.tex")
    for r in rows:  # every plotted activity is quoted to 3 or 4 dp somewhere in the paragraph
        a = float(r["activity"])
        assert f"{a:.3f}" in txt or f"{a:.4f}" in txt, \
            f"the activity at k={r['k']} ({a:.4f}) is not in the appendix"


def test_the_committed_arms_activity_is_derived_not_typed():
    """The calibration target was typed as 0.08376 from a scan of a whole k-SWEEP directory --
    pooling k=0.5, which binds on 48% of steps, and the protected classes the judge never reads.
    The arm the paper judges is k=10 over the three ordinary classes and reads 0.000080: wrong by
    a factor of 1046, and a false comparison built on it reached the compiled PDF.

    So this asserts the paper quotes the DERIVED number and never the pooled one, and it derives
    it here the same way `analysis/budget_calibration.py` does rather than trusting either.
    """
    import sys
    sys.path.insert(0, ROOT)
    from analysis.budget_calibration import committed_activity

    rate, act, tot, n = committed_activity()
    assert n == 1500, f"the reference arm no longer has 1500 trajectories ({n})"
    assert rate < 0.001, f"the committed k=10 arm now binds at {rate:.6f}; the paragraph is stale"

    txt = M.body("appendix_selection.tex")
    count = f"${act}$ of ${tot:,}$".replace(",", "{,}")
    assert count in txt, \
        f"the appendix does not quote the committed arm's own activity ({act} of {tot})"
    # The rate must sit WITH its count. A bare "0.008" check passed when the rate was deleted,
    # because "0.008" is a substring of the chosen budget's "8.008" further down the paragraph --
    # caution (an), a guard satisfied by a different occurrence of its own digits.
    i = txt.index(count)
    near = txt[i:i + 60]
    assert f"{100 * rate:.3f}" in near, \
        f"the derived rate ({100 * rate:.3f}%) is not printed beside its count: {near!r}"

    # The pooled figure must never come back, in either spelling.
    for bad in ("8.376", "0.08376"):
        assert bad not in txt, \
            f"the appendix has reacquired the pooled sweep figure {bad}; it is not an arm's rate"


def test_the_paper_does_not_claim_the_budgets_were_matched():
    """The k=1.0 arm binds about 1000x harder than the committed arm, not 'as hard as'. The
    corrected paragraph must say which, because the whole point of quoting a rate is the reader
    comparing it to the paper's own."""
    txt = M.body("appendix_selection.tex")
    assert "calibrated to bind as hard as it does" not in txt, \
        "the withdrawn 'matched budget' claim is back in the appendix"
    assert "thousand times" in txt, \
        "the appendix no longer says how much harder the chosen budget binds than the paper's own"
    body_txt = M.body("experiments.tex")
    assert "calibrated to bind as hard as" not in body_txt, \
        "the withdrawn 'matched budget' claim is in the body"
