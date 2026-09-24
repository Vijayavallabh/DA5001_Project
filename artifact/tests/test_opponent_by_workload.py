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


def test_l5_reads_the_meter_and_both_halves_are_in_the_csv():
    """L5 is a PAIR by design: the two gains are differences against the same anchor level, so a
    large rho on g_met alone could be arithmetic. The control is g_sel, and the test pins that the
    two disagree -- if they ever stop disagreeing, the arithmetic objection is live again."""
    r = rows()[0]
    s, m = float(r["rho_gsel"]), float(r["rho_gmet"])
    if r["l5"] == "THE METER AGAIN":
        assert abs(m) > abs(s) + 0.2, f"verdict says the meter, rhos say {s:+.3f} vs {m:+.3f}"
    elif r["l5"] == "SELECTION, NOT THE METER":
        assert abs(s) > abs(m) + 0.2
    else:
        assert abs(abs(s) - abs(m)) <= 0.2
    assert all(float(x["rho_gmet"]) == m and float(x["rho_gsel"]) == s for x in rows())


def test_the_meters_gain_is_monotone_in_opponent_strength_as_the_log_claims():
    """Caution (ai): the CLAIM about the series is what nothing checks. The log says 'perfectly
    monotone', which is a shape, so the shape is rebuilt from the CSV in strength order."""
    rs = sorted(rows(), key=lambda r: float(r["opponent_strength"]))
    g = [float(r["g_met"]) for r in rs]
    assert g == sorted(g), f"g_met is no longer monotone in strength: {g}"
    sel = [float(r["g_sel"]) for r in rs]
    assert sel != sorted(sel) and sel != sorted(sel, reverse=True), \
        "g_sel has become monotone too; the contrast L5 rests on is gone"


def test_the_cotaeval_inversion_is_on_the_selection_side():
    by = {r["workload"]: r for r in rows()}
    cta = by["CoTaEval-QA"]
    import statistics
    med = statistics.median(float(r["g_sel"]) for r in rows())
    assert float(cta["g_sel"]) < med, "CoTaEval-QA's selection gain is no longer the low one"
    assert float(cta["g_sel"]) == min(float(r["g_sel"]) for r in rows()), \
        "the log calls it the lowest of the five"


def test_every_arm_faced_the_same_opponent_through_the_same_pipeline():
    """The axis is ONE opponent seen by different workloads. If an arm were judged against another
    model, its 'strength' would be a statement about the opponent instead of the workload and the
    correlation would mean nothing -- caution (at), and the runs record enough to check it."""
    seen = {(r["opponent"], r["anchor"], r["opponent_generator"]) for r in rows()}
    assert len(seen) == 1, f"the arms do not share an opponent: {sorted(seen)}"
    opp, anc, gen = seen.pop()
    assert opp and anc and gen, (
        "the opponent columns are blank -- this CSV was written somewhere output/ does not exist, "
        "so nothing checked who the arms were judged against")
    assert gen == "h1.py", gen
