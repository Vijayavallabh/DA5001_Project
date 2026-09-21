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
