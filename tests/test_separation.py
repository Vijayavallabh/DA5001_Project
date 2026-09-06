"""feat-031: the two spend rates behind Proposition 5, and the monotonicity the proof relies on."""
import csv, os, statistics as st

from analysis.separation import legit_rate, recon_rows

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def test_legit_rate_is_median_spend_per_generated_token(tmp_path):
    p = tmp_path / "per_trajectory.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["k", "split", "Z", "gen_len"])
        w.writeheader()
        w.writerows([
            {"k": "1.0", "split": "neutral", "Z": "100", "gen_len": "100"},   # 1.0
            {"k": "1.0", "split": "neutral", "Z": "60", "gen_len": "100"},    # 0.6
            {"k": "3.0", "split": "creative", "Z": "160", "gen_len": "100"},  # 1.6
            {"k": "0.0", "split": "neutral", "Z": "0", "gen_len": "100"},     # k<=0 baseline, dropped
            {"k": "1.0", "split": "neutral", "Z": "5", "gen_len": "0"},       # zero-length, dropped
        ])
    med, lo, hi, n = legit_rate(str(p))
    assert n == 3 and med == 1.0
    assert (lo, hi) == (0.8, 1.6)  # per-class medians: neutral 0.8, creative 1.6


def test_recon_rows_drops_the_unbudgeted_baselines(tmp_path):
    p = tmp_path / "odometer_per_passage.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["k", "mode", "L", "B_user", "recall", "spend_total"])
        w.writeheader()
        w.writerows([
            {"k": "-1.0", "mode": "single", "L": "0", "B_user": "inf", "recall": "0.5", "spend_total": "0"},
            {"k": "0.0", "mode": "single", "L": "0", "B_user": "inf", "recall": "0.0", "spend_total": "0"},
            {"k": "3.0", "mode": "oracle", "L": "50", "B_user": "inf", "recall": "0.2", "spend_total": "600"},
            {"k": "3.0", "mode": "oracle", "L": "50", "B_user": "200", "recall": "0.05", "spend_total": "200"},
        ])
    recall, spend = recon_rows(str(p))
    assert set(recall) == {(3.0, "oracle", 50)}          # k = -1 and k = 0 are baselines, not budgeted runs
    assert spend[(3.0, "oracle", 50)] == [600.0]         # only the uncapped replay contributes a spend


def test_reconstruction_is_nondecreasing_in_the_budget():
    """The one inequality Proposition 5 proves: cutting the query sequence earlier cannot recover more."""
    recall, _ = recon_rows(os.path.join(RESULTS, "odometer_per_passage.csv"))
    assert recall, "results/odometer_per_passage.csv is missing or empty"
    for key, by_budget in recall.items():
        budgets = sorted(by_budget)  # inf sorts last
        means = [st.mean(by_budget[b]) for b in budgets]
        for lo, hi, b_lo, b_hi in zip(means, means[1:], budgets, budgets[1:]):
            assert lo <= hi + 1e-9, f"{key}: recall fell from {lo} at B={b_lo} to {hi} at B={b_hi}"
