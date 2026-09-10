"""The paper's claim here is a NEGATIVE -- that no measured quantity predicts what a higher Renyi
order is worth -- and a negative is only as good as the test behind it. The exact permutation
p-value is that test, so it is checked against cases whose answer is known by counting, not by an
approximation that would flatter a weak correlation at n = 7."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.order_predictors import exact_p, n_params, spearman  # noqa: E402


def test_spearman_endpoints():
    a = [1, 2, 3, 4, 5]
    assert abs(spearman(a, a) - 1.0) < 1e-12
    assert abs(spearman(a, a[::-1]) + 1.0) < 1e-12


def test_exact_p_of_a_perfect_correlation_is_two_over_n_factorial():
    """Only the identity and the reversal reach |rho| = 1, so the two-sided p is exactly 2/n!."""
    import math
    for n in (4, 5, 6):
        a = list(range(n))
        assert abs(exact_p(a, a) - 2 / math.factorial(n)) < 1e-12


def test_exact_p_is_not_the_normal_approximation_at_small_n():
    """At n = 4 a rho of 0.8 is p = 1/3, not the 0.2 an asymptotic test would give. This is the
    whole reason the four-pair result could not support a negative."""
    a = [1, 2, 3, 4]
    b = [1, 2, 4, 3]                    # one adjacent swap -> rho = 0.8
    assert abs(spearman(a, b) - 0.8) < 1e-12
    assert abs(exact_p(a, b) - 1 / 3) < 1e-12


def test_parameter_count_reads_headers_and_matches_the_published_size():
    m = "output/phase5/anchor_kl3m-002-520m"
    if not os.path.isdir(m):
        return
    n = n_params(m)
    assert 5.0e8 < n < 5.5e8, n     # "520m" in the model's own name


def test_permutation_test_in_rank_space_matches_the_naive_version():
    """exact_p was rewritten to permute ranks and compare sums of squared rank differences, because
    2e6 draws of the naive version is minutes rather than seconds. The rewrite is only safe if it
    counts the same event, so it is checked against a brute-force recomputation."""
    import itertools
    a = [3.0, 1.0, 4.0, 1.5, 9.0, 2.0]
    b = [2.0, 7.0, 1.0, 8.0, 2.8, 1.8]
    obs = abs(spearman(a, b))
    naive = sum(1 for p in itertools.permutations(range(len(b)))
                if abs(spearman(a, [b[i] for i in p])) >= obs - 1e-12)
    assert abs(exact_p(a, b) - naive / 720) < 1e-12
    assert exact_p.exact is True


def test_family_means_collapse_a_family_to_one_point():
    """The conservative test exists because several pairs share a model family. Its arithmetic is a
    mean per family in a fixed order, and getting that order wrong would silently pair one family's
    candidate with another's advantage."""
    import statistics as st

    from analysis.order_predictors import FAMILY
    rows = [dict(pair="KL3M-520M", x=1.0), dict(pair="KL3M-1.7B", x=3.0),
            dict(pair="Pleias-1.2B", x=10.0)]
    fams = sorted({FAMILY[r["pair"]] for r in rows})
    assert fams == ["KL3M", "Pleias"]
    means = [st.mean([r["x"] for r in rows if FAMILY[r["pair"]] == f]) for f in fams]
    assert means == [2.0, 10.0]
