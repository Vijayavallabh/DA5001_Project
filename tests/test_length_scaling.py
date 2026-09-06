"""feat-033: the length-scaling claim and the window split it rests on."""
import csv, os
from collections import defaultdict

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def test_windows_tile_the_prefix_without_overlap():
    """The per-window certificate splits a work into consecutive non-overlapping windows."""
    s = [1.0] * 130
    window = 50
    wins = [sum(s[j:j + window]) for j in range(0, 130 - window + 1, window)]
    assert wins == [50.0, 50.0]          # 130 tokens -> two whole windows, the 30-token remainder dropped
    assert len(wins) == 130 // window


def test_vacuity_cannot_rise_with_the_length_of_the_work():
    """S(x) is a sum of positive per-token surprisals and K = k*T_max is fixed, so S <= K only gets harder."""
    path = os.path.join(RESULTS, "length_scaling_summary.csv")
    assert os.path.exists(path), "run analysis/length_scaling.py first"
    by = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        by[(r["source"], float(r["k"]))][int(r["n_tokens"])] = float(r["whole_work_vacuous_pct"])
    assert by, "empty summary"
    for key, series in by.items():
        ns = sorted(series)
        for a, b in zip(ns, ns[1:]):
            assert series[a] >= series[b] - 1e-9, f"{key}: vacuity rose from {series[a]}% at {a} to {series[b]}% at {b}"


def test_surprisal_grows_with_length_in_the_per_work_rows():
    path = os.path.join(RESULTS, "length_scaling.csv")
    assert os.path.exists(path), "run analysis/length_scaling.py first"
    by_work = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        by_work[(r["source"], r["work"])][int(r["n_tokens"])] = float(r["S"])
    for key, series in by_work.items():
        ns = sorted(series)
        for a, b in zip(ns, ns[1:]):
            assert series[a] <= series[b] + 1e-6, f"{key}: S fell from {series[a]} at {a} to {series[b]} at {b}"
