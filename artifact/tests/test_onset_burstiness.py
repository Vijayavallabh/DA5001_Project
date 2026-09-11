"""A pre-registered test refuted by its own data is only worth anything if the statistic is the one
that was committed, so that is what this pins."""
import csv
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.onset_burstiness import load_pairs  # noqa: E402
from analysis.onset_gutenberg import exact_p, spearman  # noqa: E402


def test_the_committed_statistic_is_k_crit_over_s_and_nothing_else():
    """results/onset_prediction_burstiness.md names k_crit/s(x) in advance and names three other
    burstiness statistics as excluded. A later edit that swaps in whichever one works would be
    invisible in the CSV, so it is checked in the source."""
    src = open("analysis/onset_burstiness.py").read()
    assert 'burstiness=round(kc / s, 4)' in src
    for excluded in ("s_max", "s_p90", "s_std"):
        assert excluded not in src, excluded
    path = "results/onset_prediction_burstiness.md"
    if os.path.exists(path):
        md = open(path).read()
        assert "+0.786" in md and "0.6" in md and "Predicted sign: positive" in md


def test_committed_table_reproduces_its_own_verdict():
    path = "results/onset_burstiness.csv"
    if not os.path.exists(path):
        return
    rows = list(csv.DictReader(open(path)))
    assert len(rows) >= 3
    for r in rows:
        assert abs(float(r["burstiness"]) - float(r["k_crit"]) / float(r["s_mean"])) < 1e-3, r
        assert abs(float(r["ratio"]) - float(r["onset"]) / float(r["s_mean"])) < 5e-3, r
        assert abs(float(r["onset_over_k_crit"]) - float(r["onset"]) / float(r["k_crit"])) < 1e-3, r
    x = [float(r["burstiness"]) for r in rows]
    y = [float(r["ratio"]) for r in rows]
    rho = spearman(x, y)
    # the committed refuting band, and the committed direction it would have needed
    assert abs(rho) < 0.6, f"rho = {rho}, which is no longer a refutation"
    assert exact_p(x, y) > 0.5
    cv = lambda v: st.stdev(v) / st.mean(v)
    assert cv([float(r["onset_over_k_crit"]) for r in rows]) > cv(y), "k_crit must be the worse normaliser"


def test_the_manifest_reader_takes_the_budget_path_column():
    """Field 3, not field 2: field 2 is the composition summary and would silently yield no rows."""
    if not os.path.exists("results/onset_pairs.tsv"):
        return
    for name, bp in load_pairs("results/onset_pairs.tsv"):
        assert bp.startswith("results/budget_path"), (name, bp)
