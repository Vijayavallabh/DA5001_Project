"""feat-003: the per-trajectory budget invariant Z <= max(0, B) + 1e-3 and per-step a_t <= k_t + 1e-3 on saved logs."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-3


def test_per_step_and_per_trajectory_invariant_on_sample():
    n = 0
    for line in open(ROOT / "tests" / "data" / "sample_trajectories.jsonl"):
        r = json.loads(line)
        Z, B = r["aggregate"]["total_spend"], r["aggregate"]["final_budget"]
        assert Z <= max(0.0, B) + EPS, r["metadata"]
        cum = 0.0
        for s in r["per_step_log"]:
            assert s["a_t"] <= s["k_t"] + EPS, (r["metadata"], s["t"])
            cum += s["a_t"]
        assert abs(cum - Z) <= 1e-2 * max(1.0, Z)  # cum_kl_spent is the running sum of a_t
        n += 1
    assert n == 8


def test_per_trajectory_invariant_on_results_csv():
    path = ROOT / "results" / "per_trajectory.csv"
    rows = list(csv.DictReader(open(path)))
    assert len(rows) >= 4000
    bad = [r for r in rows if float(r["Z"]) > max(0.0, float(r["B"])) + EPS]
    assert not bad, bad[:3]
    assert all(r["invariant_viol"] == "0" and r["viol_step"] == "0" for r in rows)
