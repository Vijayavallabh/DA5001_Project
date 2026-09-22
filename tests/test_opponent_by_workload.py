"""feat-178 P1: the rank test, and the claims the scoring log makes ABOUT its numbers.

Caution (ai): a number is checked against its CSV and the CLAIM ABOUT a set of numbers is checked
against nothing. The claim here is not the correlation -- it is the INVERSION, that one workload
has our own workload's opponent strength and an order of magnitude less of the effect. That is
what makes the opponent account incomplete, and it is rebuilt from the CSV rather than trusted.
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from analysis.opponent_by_workload import exact_p, spearman  # noqa: E402
import manuscript as M  # noqa: E402

CSV = os.path.join(ROOT, "results", "opponent_by_workload.csv")


def rows():
    return list(csv.DictReader(open(CSV, encoding="utf-8")))


def test_spearman_is_a_rank_correlation_and_averages_ties():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0
    # Monotone but not linear: a rank test must still read exactly 1.
    assert spearman([1, 2, 3, 4], [1, 2, 4, 1000]) == 1.0
    # Ties take the AVERAGE rank: [1,1,2,2] ranks as [1.5,1.5,3.5,3.5], which against [1,2,3,4]
    # is 4/sqrt(20) and not 1. Getting this wrong inflates every correlation with a tie in it.
    assert abs(spearman([1, 2, 3, 4], [1, 1, 2, 2]) - 4 / 20 ** 0.5) < 1e-12
    # A constant series has zero rank variance and correlates with nothing; nan, never a number.
    r = spearman([1, 2, 3, 4], [5, 5, 5, 5])
    assert r != r, f"a constant series returned {r} instead of nan"


def test_the_exact_p_floor_is_what_the_registration_says_it_is():
    """The registration forbids a significance claim because of this number; if the permutation
    count were wrong the prohibition would be justified by a false premise."""
    rho, p, n = exact_p([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert (rho, n) == (1.0, 120)
    assert abs(p - 2 / 120) < 1e-12, "a perfect n=5 correlation must read the floor, 2/120"
    assert exact_p([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])[1] == p, "the test must be two-sided"


def test_the_verdict_and_the_rho_cannot_drift_apart():
    """Caution (av): a verdict is a string over a number, and it outlives the number."""
    for r in rows():
        rho, rho_n = float(r["rho"]), float(r["rho_normalised"])
        for value, verdict in ((rho, r["verdict"]), (rho_n, r["verdict_normalised"])):
            if verdict == "CONSISTENT":
                assert value <= -0.7, f"{verdict} at rho={value}"
            elif verdict == "REFUTED":
                assert value >= 0.3, f"{verdict} at rho={value}"
            else:
                assert -0.7 < value < 0.3, f"{verdict} at rho={value}"


def test_the_headroom_column_is_the_room_a_difference_of_win_rates_has():
    for r in rows():
        u = float(r["anchor_win"])
        assert abs(float(r["headroom"]) - min(u, 1 - u)) < 5e-6
        assert abs(float(r["opponent_strength"]) - (1 - u)) < 5e-6
        assert abs(float(r["d3_normalised"]) - float(r["d3"]) / float(r["headroom"])) < 5e-6


def test_the_inversion_the_scoring_log_rests_on_is_in_the_data():
    by = {r["workload"]: r for r in rows()}
    ours, cta = by["ours"], by["CoTaEval-QA"]
    ds = abs(float(ours["opponent_strength"]) - float(cta["opponent_strength"]))
    assert ds < 0.02, \
        f"CoTaEval-QA and ours no longer share an opponent strength ({ds:.4f}); the inversion is stale"
    assert abs(float(ours["d3"])) > 5 * abs(float(cta["d3"])), \
        "the two workloads' effects are no longer an order of magnitude apart"


def test_the_confound_is_stated_because_it_is_present():
    """The registration says a reanalysis cannot separate strength from task type HERE. That is
    only honest if the two really do move together in this table, so it is checked."""
    by_kind = {}
    for r in rows():
        by_kind.setdefault(r["task_type"], []).append(float(r["opponent_strength"]))
    comp, instr = by_kind.get("completion", []), by_kind.get("instruction", [])
    assert comp and instr
    assert max(comp) < min(instr), (
        "completion workloads no longer all have weaker opponents than instruction ones; the "
        "confound the registration concedes has gone, so the concession must be revisited")
