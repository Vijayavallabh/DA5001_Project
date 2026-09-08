"""Plan v5: the Cramer rate function behind the "10^3-10^4 overhead" finding must be correct.

Lambda*_s(u) = sup_lambda [ lambda*u - log E_{p_s}[e^{lambda U}] ] for U in {1, 0.5, 0}. A wrong
maximisation would manufacture the headline, so it is checked against closed forms and invariants.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.utility_price import mean_u, rate, three_point  # noqa: E402


def test_rate_vanishes_at_the_safe_models_own_mean():
    d = (0.3, 0.1, 0.6)
    assert rate(d, mean_u(d)) < 1e-6


def test_rate_is_positive_and_increasing_above_the_mean():
    d = (0.3, 0.1, 0.6)
    m = mean_u(d)
    vals = [rate(d, u) for u in (m + 0.02, m + 0.05, m + 0.10)]
    assert all(v > 0 for v in vals)
    assert vals[0] < vals[1] < vals[2]


def test_rate_is_positive_below_the_mean_too():
    d = (0.3, 0.1, 0.6)
    assert rate(d, mean_u(d) - 0.05) > 0


def test_matches_the_two_point_closed_form():
    """With no ties, U is Bernoulli and the rate function is the binary KL d(u || p)."""
    p = 0.4
    d = (p, 0.0, 1 - p)
    for u in (0.5, 0.6, 0.75):
        kl = u * math.log(u / p) + (1 - u) * math.log((1 - u) / (1 - p))
        assert abs(rate(d, u) - kl) < 1e-4, (u, rate(d, u), kl)


def test_rate_beats_pinskers_lower_bound():
    """Lambda*(u) >= 2*(u - mean)^2 for a variable supported on [0,1] (Pinsker/Hoeffding)."""
    d = (0.3, 0.1, 0.6)
    m = mean_u(d)
    for u in (m + 0.05, m + 0.15):
        assert rate(d, u) >= 2 * (u - m) ** 2 - 1e-6


def test_three_point_normalises():
    w, t, l = three_point(30.5, 61.7)
    assert abs(w + t + l - 1.0) < 1e-9 and t >= 0


def test_committed_result_shows_a_large_overhead():
    """The paper claims spend/Lambda* > 10^3 at every budget; guard it against the CSV."""
    import csv
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "utility_price.csv")
    if not os.path.exists(path):
        return
    rows = [r for r in csv.DictReader(open(path)) if r["spend_over_lambda_star"]]
    assert rows, "no rows with a finite ratio"
    assert all(float(r["spend_over_lambda_star"]) > 1e3 for r in rows), \
        [(r["k"], r["spend_over_lambda_star"]) for r in rows]
    # and the claim must survive the weakest reading of the bootstrap, not just the point estimate
    cons = [r for r in rows if r.get("spend_over_lambda_star_conservative")]
    assert cons, "conservative column missing"
    assert all(float(r["spend_over_lambda_star_conservative"]) > 1e3 for r in cons), \
        [(r["k"], r["spend_over_lambda_star_conservative"]) for r in cons]
