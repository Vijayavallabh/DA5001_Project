"""The seed-word gradient, and the sensitivity check that qualifies it.

The gradient is the paper's strongest correlation (rho = -0.958, exact p = 0.0002) and it is fitted
to nine points that are each a single LoRA fine-tune. The seed ladders measured what re-training one
of them does, so the question "does the gradient survive that?" became answerable and had to be
asked. These pin the answer to the CSV and pin the words the appendix uses to report it.

The claim being guarded is deliberately two-sided: the DIRECTION survives (sign kept in every draw)
and the MAGNITUDE does not (median -0.849, not -0.958). A future edit that quietly restores
"-0.958" as the headline, or that drops the qualification, should fail here.
"""
import csv
import os
import subprocess
import sys

import pytest

from tests.manuscript import tex

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results/seedword_gradient.csv")


@pytest.fixture(scope="module")
def arms():
    with open(CSV) as fh:
        return {r["arm"]: r for r in csv.DictReader(fh)}


@pytest.fixture(scope="module")
def onset():
    return " ".join(open(tex("sections/appendix_onset.tex"), encoding="utf-8").read().split())


def test_the_observed_gradient_still_reproduces(arms):
    r = arms["observed"]
    assert float(r["rho_median"]) == -0.958 and float(r["exact_p"]) == 0.0002
    assert int(r["n"]) == 9


def test_the_direction_survives_the_measured_noise(arms):
    for key in ("reseed_noise_pleias12b_copybench", "reseed_noise_kl3m520m_copybench"):
        assert float(arms[key]["pct_sign_kept"]) == 100.0, (
            f"{key}: the gradient's sign no longer survives re-seeding; the appendix says it does")


def test_the_magnitude_does_not_and_the_appendix_says_so(arms, onset):
    r = arms["reseed_noise_pleias12b_copybench"]
    med, lo, hi = float(r["rho_median"]), float(r["rho_lo5"]), float(r["rho_hi95"])
    assert med > -0.958, "the median under noise must be weaker than the point estimate"
    assert f"${med:.3f}$" in onset, f"appendix must quote the median {med:.3f}"
    assert f"$[{lo:.3f}, {hi:.3f}]$" in onset, f"appendix must quote [{lo:.3f}, {hi:.3f}]"
    assert f"${float(r['pct_p_under_05']):.1f}\\%$" in onset
    assert f"${float(r['reseed_sd']):.4f}$" in onset


def test_the_appendix_keeps_the_two_sided_reading(onset):
    assert "The\nemph" not in onset
    assert "direction is not at risk and the magnitude is" in onset, \
        "the qualification's point is that direction survives and magnitude does not"
    assert "post hoc" in onset, "the sensitivity check is not pre-registered and must say so"


def test_convergence_is_not_confounded_with_seed_words(arms, onset):
    yes, no = arms["seed_words_where_converged_yes"], arms["seed_words_where_converged_no"]
    ylo, yhi = float(yes["rho_lo5"]), float(yes["rho_hi95"])
    nlo, nhi = float(no["rho_lo5"]), float(no["rho_hi95"])
    # both groups must straddle the range, or the gradient IS the convergence split
    assert ylo == nlo, "the two groups no longer share their lowest seed-word level"
    assert min(yhi, nhi) > max(ylo, nlo), "the groups no longer overlap"
    assert f"${ylo:.2f}$ to ${yhi:.2f}$" in onset and f"${nlo:.2f}$ to ${nhi:.2f}$" in onset


def test_the_script_is_deterministic():
    """A sensitivity analysis quoted in a paper must give the same number twice."""
    out = os.path.join(ROOT, "results/.seedword_gradient_check.csv")
    try:
        subprocess.run([sys.executable, "analysis/seedword_gradient.py", "--trials", "2000",
                        "--out", "results/.seedword_gradient_check.csv"],
                       cwd=ROOT, capture_output=True, check=True)
        first = open(out).read()
        subprocess.run([sys.executable, "analysis/seedword_gradient.py", "--trials", "2000",
                        "--out", "results/.seedword_gradient_check.csv"],
                       cwd=ROOT, capture_output=True, check=True)
        assert open(out).read() == first, "the sensitivity analysis is not reproducible"
    finally:
        if os.path.exists(out):
            os.remove(out)
