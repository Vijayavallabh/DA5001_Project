"""feat-007: the anytime-valid empirical-Bernstein CS covers the true mean uniformly over time."""
import random

from dap.stats import anytime_valid_cs, anytime_valid_cs_path


def test_uniform_coverage_over_time():
    rng = random.Random(0)
    alpha, reps, n = 0.1, 400, 300
    misses = 0
    for _ in range(reps):
        mu = rng.choice([0.05, 0.3, 0.5, 0.9])
        xs = [1.0 if rng.random() < mu else 0.0 for _ in range(n)]  # Bernoulli(mu)
        path = anytime_valid_cs_path(xs, alpha=alpha)
        misses += any(not (lo <= mu <= hi) for lo, hi in path)
    assert misses / reps <= alpha + 0.03, misses / reps  # time-uniform miss rate at most alpha (plus MC slack)


def test_shrinks_and_is_valid_interval():
    rng = random.Random(1)
    xs = [min(1.0, max(0.0, rng.gauss(0.3, 0.1))) for _ in range(2000)]
    lo, hi = anytime_valid_cs(xs, alpha=0.05)
    assert 0.0 <= lo <= 0.3 <= hi <= 1.0 and hi - lo < 0.08
    path = anytime_valid_cs_path(xs, alpha=0.05)
    assert path[10][1] - path[10][0] > path[1999][1] - path[1999][0]
    assert anytime_valid_cs([], alpha=0.05) == (0.0, 1.0)


def test_rejects_out_of_range():
    try:
        anytime_valid_cs([0.2, 1.5])
    except ValueError:
        return
    raise AssertionError("out-of-range sample accepted")
