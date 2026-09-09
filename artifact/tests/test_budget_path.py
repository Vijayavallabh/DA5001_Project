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


def test_k_critical_reports_where_the_maximum_binds():
    """feat-064: the seed reaches k_crit only through which token the adversary must produce first,
    so the binding step has to be recorded, not assumed to be zero."""
    from analysis.budget_path import k_critical
    # a hard opening then easy text: the running max binds at the first step
    v, arg = k_critical([10.0, 1.0, 1.0, 1.0], 0.0, want_argmax=True)
    assert arg == 0 and abs(v - 10.0) < 1e-9
    # an easy opening then a hard run: it binds later
    v, arg = k_critical([1.0, 1.0, 20.0], 0.0, want_argmax=True)
    assert arg == 2 and abs(v - 22.0 / 3) < 1e-9
    # the scalar form is unchanged
    assert abs(k_critical([10.0, 1.0, 1.0, 1.0], 0.0) - 10.0) < 1e-9
