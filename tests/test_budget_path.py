"""feat-023: the token-bucket service model that predicts which target tokens the banked budget can pay for."""
from analysis.budget_path import simulate_bucket


def test_bucket_pays_tokens_until_a_spike_empties_it():
    s = [1.0, 1.0, 1.0, 10.0, 1.0, 1.0]
    paid = simulate_bucket(s, delta=0.0, k=2.0)
    assert paid == [True, True, True, False, True, True]


def test_predicted_fraction_counts_only_long_runs():
    from analysis.budget_path import reproducible_fraction
    s = [1.0, 1.0, 1.0, 10.0, 1.0, 1.0]
    assert reproducible_fraction(s, delta=0.0, k=2.0, min_run=1) == 5 / 6
    assert reproducible_fraction(s, delta=0.0, k=2.0, min_run=3) == 3 / 6


def test_initial_debt_forces_the_opening_tokens():
    s = [1.0] * 5
    assert simulate_bucket(s, delta=2.5, k=1.0) == [False, False, False, True, True]
