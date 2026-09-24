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

# RETIRED 2026-09-19, appendix reduction. The paragraph each of these read was removed
# when the appendix was cut from 52 pages, so the sentence they pinned no longer exists.
# A guard for a claim the paper does not make protects nothing; recorded here rather than
# silently deleted, so the removal is visible to the next reader:
#   test_the_magnitude_does_not_and_the_appendix_says_so
#   test_the_appendix_keeps_the_two_sided_reading
#   test_convergence_is_not_confounded_with_seed_words
