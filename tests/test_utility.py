"""feat-029: the two pieces of the utility evaluation that would fail silently if wrong."""
import csv, os

from analysis.utility import arm_won, served_prompt

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def test_arm_won_respects_the_presentation_flip():
    # not flipped: the arm was shown first, so a verdict of "A" is a win for the arm
    assert arm_won("A", False) is True
    assert arm_won("B", False) is False
    # flipped: the arm was shown second, so "B" is the win
    assert arm_won("B", True) is True
    assert arm_won("A", True) is False


def test_served_prompt_strips_the_generation_and_the_harness_header():
    agg = {"full_text": "Complete the prefix:\nWho wrote Dune? Frank Herbert wrote it.",
           "generation": " Frank Herbert wrote it."}
    assert served_prompt(agg) == "Who wrote Dune?"


def test_served_prompt_falls_back_when_the_generation_is_not_a_suffix():
    # early-EOS trajectories can log a generation that full_text does not end with
    agg = {"full_text": "Complete the prefix:\nWho wrote Dune?", "generation": "unrelated"}
    assert served_prompt(agg) == "Who wrote Dune?"


def test_every_judged_arm_is_calibrated_against_the_null():
    """The null arm must exist and be judged, or no win rate in the table can be interpreted."""
    path = os.path.join(RESULTS, "utility_summary.csv")
    assert os.path.exists(path), "run analysis/utility.py first"
    rows = list(csv.DictReader(open(path)))
    null = [r for r in rows if r["decoder"] == "risky only (null)"]
    assert null, "no null arm: win rates are uninterpretable without it"
    assert int(null[0]["n_judged"]) > 0
    judged = [r for r in rows if int(r["n_judged"] or 0) > 0]
    assert len(judged) >= 13, f"expected every arm judged, got {len(judged)}"
    # a position-randomised judge should not land far from an even split
    for r in judged:
        first = float(r["judge_picked_first_pct"])
        assert 25.0 <= first <= 75.0, f"{r['decoder']} k={r['k']}: judge picked first {first}% of the time"
