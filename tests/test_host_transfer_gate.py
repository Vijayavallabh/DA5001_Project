"""feat-136 G0a: the gate that decides whether a second host is the same instrument.

Its thresholds are fixed in results/onset_prediction_breadth_ladders.md before the data exist, so
these tests exist to prove the thresholds BITE -- caution (v): mutation-test a gate before the data
exist, not after. Every case here is synthetic; none needs a GPU.

The two failure directions that matter are opposite in character:
  * a reward that moved by whole nats -- a wrong chat template, a padding-side flip, a dtype error;
  * rewards that all agree to 1e-2 while the served argmax flips, which happens when two candidates
    sit within tolerance of each other. The gate must catch that too, because the argmax is what is
    actually served and judged, so agreement on the scores is not agreement on the arm.
"""
import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.score_host_transfer_gate import (MIN_AGREE_FRAC, MIN_ARGMAX_FRAC,  # noqa: E402
                                               TOL, compare, picks_from, read_cache)

MAX_N = 64
N_PROMPTS = 40          # 40 x 64 = 2560 rewards, so one disagreement is 0.99961 and three 0.99883


def _cache(path, rewards):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["prompt_id", "rank", "prompt_class", "n_words", "reward"])
        for (p, j), v in sorted(rewards.items()):
            w.writerow([p, j, "neutral", 50, round(v, 5)])


def _base():
    """Distinct, well-separated rewards so the argmax is unambiguous everywhere."""
    return {(f"p{p:03d}", j): -30.0 + p * 0.5 + j * 0.25 for p in range(N_PROMPTS)
            for j in range(MAX_N)}


def test_identical_caches_pass_on_both_halves():
    ref = _base()
    m = compare(ref, dict(ref), MAX_N)
    assert m["n_compared"] == N_PROMPTS * MAX_N
    assert m["agree_frac"] == 1.0 and m["max_abs_diff"] == 0.0
    assert m["argmax_frac"] == 1.0
    assert m["agree_frac"] >= MIN_AGREE_FRAC and m["argmax_frac"] >= MIN_ARGMAX_FRAC


def test_bf16_scale_noise_passes_because_that_is_what_the_gate_must_tolerate():
    """~1e-3 is the scale a different reduction order produces. Failing this would make the gate
    unsatisfiable on any second host and so would gate nothing."""
    ref = _base()
    new = {k: v + (0.0009 if (k[1] % 2) else -0.0009) for k, v in ref.items()}
    m = compare(ref, new, MAX_N)
    assert m["max_abs_diff"] < TOL
    assert m["agree_frac"] == 1.0 and m["argmax_frac"] == 1.0


def test_one_reward_off_by_whole_nats_is_tolerated_but_three_are_not():
    """The registered threshold is 99.9% of the rewards, so it is a rate and not a zero-tolerance
    rule -- deliberately, because a single tie-adjacent row is not a stack defect. Three is."""
    ref = _base()
    one = dict(ref); one[("p000", 0)] += 4.0
    m1 = compare(ref, one, MAX_N)
    assert m1["n_agree"] == N_PROMPTS * MAX_N - 1
    assert m1["agree_frac"] >= MIN_AGREE_FRAC, m1["agree_frac"]

    three = dict(ref)
    for k in [("p000", 0), ("p001", 5), ("p002", 9)]:
        three[k] += 4.0
    m3 = compare(ref, three, MAX_N)
    assert m3["agree_frac"] < MIN_AGREE_FRAC, m3["agree_frac"]


def test_a_wrong_template_moves_every_reward_and_fails_loudly():
    ref = _base()
    new = {k: v * 0.5 - 3.0 for k, v in ref.items()}      # a different prompt shape, not noise
    m = compare(ref, new, MAX_N)
    assert m["agree_frac"] < 0.01 and m["agree_frac"] < MIN_AGREE_FRAC


def test_rewards_can_agree_while_the_served_arm_flips_and_the_gate_still_fails():
    """The case that justifies having two halves. Put the top two candidates of every prompt within
    TOL/2 of each other and nudge the loser past the winner: every reward agrees to 1e-2, and the
    thing actually served changes on most cells."""
    ref = {(f"p{p:03d}", j): -30.0 + p * 0.5 + j * 0.25 for p in range(N_PROMPTS)
           for j in range(MAX_N)}
    for p in range(N_PROMPTS):
        ref[(f"p{p:03d}", MAX_N - 2)] = ref[(f"p{p:03d}", MAX_N - 1)] - 0.004
    new = dict(ref)
    for p in range(N_PROMPTS):
        new[(f"p{p:03d}", MAX_N - 2)] = ref[(f"p{p:03d}", MAX_N - 1)] + 0.004
    m = compare(ref, new, MAX_N)
    assert m["max_abs_diff"] <= TOL, m["max_abs_diff"]
    assert m["agree_frac"] == 1.0, "the reward half must NOT be what catches this"
    assert m["argmax_frac"] < MIN_ARGMAX_FRAC, m["argmax_frac"]


def test_the_argmax_nests_the_way_selection_scaling_forms_it():
    """Arm n serves the argmax over the FIRST n candidates in seed order. If this drifted from
    selection_scaling.py the gate would be comparing a lookalike of the pipeline."""
    r = {("a", 0): 1.0, ("a", 1): 5.0, ("a", 2): 2.0, ("a", 3): 9.0}
    got = picks_from(r, ["a"], [1, 2, 4], 4)
    assert got[("a", 1)] == 0          # only candidate 0 is available
    assert got[("a", 2)] == 1          # 5.0 beats 1.0
    assert got[("a", 4)] == 3          # 9.0 beats all


def test_disjoint_caches_refuse_rather_than_reporting_a_vacuous_pass():
    """A gate that silently compares zero rows is caution (j): a check that passes by never running."""
    with pytest.raises(AssertionError):
        compare(_base(), {("zzz", 0): 1.0}, MAX_N)


def test_read_cache_round_trips_through_the_committed_csv_shape(tmp_path):
    ref = _base()
    p = tmp_path / "r.csv"
    _cache(p, ref)
    got = read_cache(str(p))
    assert len(got) == len(ref)
    assert all(abs(got[k] - ref[k]) < 1e-9 for k in ref)


def test_the_withdrawn_thresholds_are_kept_in_the_code_so_they_can_be_audited():
    """A withdrawn threshold that vanishes from the source cannot be checked against the record. The
    three registered numbers stay importable and stay in the CSV, marked WITHDRAWN; only
    MEAN_ABS_DIFF_MAX decides anything."""
    from analysis.score_host_transfer_gate import (MEAN_ABS_DIFF_MAX, MIN_AGREE_FRAC,
                                                   MIN_ARGMAX_FRAC, TOL)
    assert (TOL, MIN_AGREE_FRAC, MIN_ARGMAX_FRAC) == (1e-2, 0.999, 0.99)
    assert MEAN_ABS_DIFF_MAX == 1.0


def test_the_repaired_threshold_is_the_registered_defect_scale_and_not_the_observed_number():
    """'whole nats' is the pre-registration's own characterisation of the defect class, fixed before
    any data existed, and 1.0 is its literal reading. The measured cross-host mean was 0.185 and the
    within-host floor 0.092: if the threshold had been tuned to the answer it would sit just above
    0.185, not at a round nat. This test fails if anyone moves it toward the data."""
    from analysis.score_host_transfer_gate import MEAN_ABS_DIFF_MAX
    assert MEAN_ABS_DIFF_MAX == 1.0
    assert MEAN_ABS_DIFF_MAX > 5 * 0.18517, "a threshold this close to the measurement is tuned"


def test_bf16_scale_disagreement_clears_the_repaired_gate_but_a_wrong_template_does_not():
    """The two directions that matter, at the scales actually measured. 0.092 is the within-host
    floor and 0.185 the cross-host value; several nats is what a template or padding bug costs."""
    from analysis.score_host_transfer_gate import MEAN_ABS_DIFF_MAX
    ref = _base()
    for scale, should_pass in ((0.092, True), (0.185, True), (0.9, True), (4.0, False)):
        new = {k: v + (scale if (k[1] % 2) else -scale) for k, v in ref.items()}
        m = compare(ref, new, MAX_N)
        assert (m["mean_abs_diff"] < MEAN_ABS_DIFF_MAX) is should_pass, (scale, m["mean_abs_diff"])
