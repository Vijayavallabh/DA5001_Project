"""Plan v4 / feat-035: the ordering the frontier theorem depends on."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.regimes import k_crit_rate, regimes


def test_k_crit_never_below_the_mean_rate():
    """k_crit >= s_rate is the inequality that makes [s, k_crit) a real interval. It holds because
    the running maximum of a prefix average is at least the whole-sequence average."""
    for nats, chars in [([1.0, 2.0, 3.0], [4, 8, 12]),
                        ([3.0, 2.0, 1.0], [4, 8, 12]),      # front-loaded: the gap is widest here
                        ([2.0] * 10, list(range(4, 44, 4))),  # flat: the gap closes to equality
                        ([0.1, 9.9], [5, 10])]:
        r = regimes(nats, chars)
        assert r["k_crit"] >= r["s_rate"] - 1e-12, (nats, r)
        assert r["uncertified_width"] >= 1.0 - 1e-12


def test_flat_surprisal_collapses_the_uncertified_gap():
    """A work whose surprisal is perfectly uniform leaves no room between certifying and protecting;
    burstiness is what opens the interval."""
    flat = regimes([2.0] * 8, [3 * (i + 1) for i in range(8)])
    bursty = regimes([9.0] + [1.0] * 7, [3 * (i + 1) for i in range(8)])
    assert abs(flat["uncertified_width"] - 1.0) < 1e-9
    assert bursty["uncertified_width"] > 2.0


def test_initial_debt_raises_both_boundaries():
    """delta is the bucket's initial level, so it shifts k_crit -- this is the prefix-debt mechanism."""
    nats, chars = [2.0, 2.0, 2.0, 2.0], [5, 10, 15, 20]
    base, debt = regimes(nats, chars, delta=0.0), regimes(nats, chars, delta=10.0)
    assert debt["k_crit"] > base["k_crit"]
    assert debt["s_rate"] > base["s_rate"]


def test_window_opens_exactly_when_ordinary_traffic_is_cheaper():
    nats, chars = [2.0] * 5, [4 * (i + 1) for i in range(5)]
    s = regimes(nats, chars)["s_rate"]
    assert regimes(nats, chars, c_use=s * 0.5)["window_open"] is True
    assert regimes(nats, chars, c_use=s * 2.0)["window_open"] is False
    assert regimes(nats, chars, c_use=s * 2.0)["window_width"] == 0.0


def test_k_crit_is_the_running_max_not_the_final_average():
    """A spike early in the work sets k_crit even when the tail is cheap."""
    assert k_crit_rate([10.0, 0.0, 0.0, 0.0], [1, 2, 3, 4]) == 10.0
    assert k_crit_rate([0.0, 0.0, 0.0, 10.0], [1, 2, 3, 4]) == 2.5


def test_empty_work_is_not_a_crash():
    assert regimes([], []) is None
