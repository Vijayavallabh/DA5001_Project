"""feat-138 and feat-140's scorer, exercised on synthetic CSVs before the arms produce anything.

Caution (v): mutation-test a gate before the data exist, not after. The gate that matters most here
is I1, because it is the REPAIRED statistic -- feat-136's G0b gated the raw mean and failed an arm
whose entire shift was the empty rate. The first two tests below are that repair, in both
directions.
"""
import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analysis.score_seed_and_ladder as M  # noqa: E402

JB = "Phi-3.5-mini-instruct"


def _write(path, rows_):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_[0])); w.writeheader(); w.writerows(rows_)


def _arm(out, tag, delta=0.10, spread=0.05):
    """A per-prompt file whose paired g(64)-g(8) is `delta` with a controlled spread."""
    per = [{"judge": JB, "prompt_id": f"p{i:03d}", "u_n1": 0.4, "u_n8": 0.4,
            "u_n64": 0.4 + delta + (spread if i % 2 else -spread)} for i in range(500)]
    _write(os.path.join(out, f"selection_scaling_per_prompt_{tag}.csv"), per)
    _write(os.path.join(out, f"selection_scaling_{tag}.csv"),
           [{"judge": JB, "n": n, "gain": 0.0, "gain_lo95": 0.0, "gain_hi95": 0.0,
             "kl_nats": 0.0, "n_prompts": 500, "mean_words": 80.0} for n in M.GRID])


def _stats(out, name, nonempty=100.0, empty=10, n=500, model="m/a", ref_model=None):
    mean = (1 - empty / n) * nonempty
    _write(os.path.join(out, f"arm_rank0_stats_{name}.csv"),
           [{"gen_dir": f"d/{name}", "target_model": model,
             "anchor_model": ref_model or model, "n_prompts": n, "mean_words": round(mean, 4),
             "empty_frac": round(empty / n, 6), "n_empty": empty,
             "mean_words_nonempty": nonempty, "median_words_nonempty": nonempty,
             "empty_neutral": empty, "n_neutral": 200}])


# ---- the repair: I1 gates length GIVEN NON-EMPTY, not the raw mean ----------------------------

def test_I1_passes_an_arm_whose_RAW_mean_moved_far_but_whose_non_empty_length_did_not(tmp_path):
    """feat-136's comma7bhb, reproduced. Raw mean +19%, non-empty length identical.

    Local: 10% empty, 100 words given non-empty -> raw mean 90.0
    New:    0% empty, 100 words given non-empty -> raw mean 100.0  (+11.1% raw)
    The registered statistic must not care, because nothing about the sampling path changed.
    """
    out = str(tmp_path)
    _stats(out, "ref", nonempty=100.0, empty=50)
    _stats(out, "new", nonempty=100.0, empty=0)
    new, ref = M.stat_row(out, "new"), M.stat_row(out, "ref")
    assert abs(float(new["mean_words"]) - float(ref["mean_words"])) / float(ref["mean_words"]) > 0.10
    ok, msg = M.i1_length(new, ref)
    assert ok is True, msg
    assert "given non-empty" in msg


def test_I1_fails_a_genuine_length_change_even_when_the_raw_mean_agrees(tmp_path):
    """The other direction, which is what stops the repair being a weakening.

    Contrived so the RAW means match while length given non-empty differs by 25% -- a wrong model
    or a truncation, hidden behind a compensating empty rate. The old statistic would pass it.
    """
    out = str(tmp_path)
    _stats(out, "ref", nonempty=100.0, empty=0)       # raw mean 100.0
    _stats(out, "new", nonempty=125.0, empty=100)     # raw mean 100.0
    new, ref = M.stat_row(out, "new"), M.stat_row(out, "ref")
    assert float(new["mean_words"]) == pytest.approx(float(ref["mean_words"]), abs=0.05)
    assert M.i1_length(new, ref)[0] is False


# ---- I2, the stratified empty rate -------------------------------------------------------------

def test_I2_blocks_a_large_empty_shift_for_feat138_and_only_reports_it_for_feat140(tmp_path):
    out = str(tmp_path)
    _stats(out, "ref", nonempty=100.0, empty=10)
    _stats(out, "new", nonempty=100.0, empty=120)
    new, ref = M.stat_row(out, "new"), M.stat_row(out, "ref")
    assert M.i2_empties(new, ref, blocking=True)[0] is False
    assert M.i2_empties(new, ref, blocking=False)[0] is None, "cross-host must never block on a rate"


def test_two_prop_z_is_scale_free_where_an_absolute_tolerance_is_not(tmp_path):
    """Caution (v): 0.03 absolute is unfalsifiable at KL3M and tighter than Comma-7B's own spread."""
    assert abs(M.two_prop_z(1, 500, 0, 500)) < 2.0          # 0.002 vs 0.000 -- nothing happened
    assert abs(M.two_prop_z(47, 500, 10, 500)) > 4.0        # 0.094 vs 0.020 -- something did


# ---- I4, caution (at) as a gate ----------------------------------------------------------------

def test_I4_catches_a_pipeline_mismatch_and_a_non_self_paired_arm(tmp_path):
    out = str(tmp_path)
    _stats(out, "self", model="m/anchor")
    _stats(out, "other", model="m/other")
    _stats(out, "crossed", model="m/risky8b", ref_model="m/anchor")
    assert M.i4_pipeline(M.stat_row(out, "self"), M.stat_row(out, "other"))[0] is False
    ok, msg = M.i4_pipeline(M.stat_row(out, "crossed"), M.stat_row(out, "self"))
    assert ok is False and "not self-paired" in msg
    assert M.i4_pipeline(M.stat_row(out, "self"), M.stat_row(out, "self"))[0] is True


# ---- the verdict and the distance ---------------------------------------------------------------

def test_a_failed_prediction_is_reported_as_a_failed_prediction(tmp_path, capsys):
    """pleias350m is predicted SATURATED. Give it a clear climb and the scorer must say so."""
    out = str(tmp_path)
    arm = dict(next(a for a in M.SEED_ARMS if a["name"] == "pleias350m"))
    _arm(out, arm["new_tag"], delta=0.20, spread=0.02)
    _stats(out, arm["name"]); _stats(out, arm["ref"])
    M.score_one(out, arm, "feat-138", [])
    txt = " ".join(capsys.readouterr().out.split())
    assert "CLIMBS" in txt and "verdict agrees? NO" in txt
    assert "THE PREDICTION FAILED" in txt


def test_G2_is_gated_for_feat138_and_only_reported_for_feat140(tmp_path, capsys):
    out = str(tmp_path)
    arm = dict(next(a for a in M.SEED_ARMS if a["name"] == "comma1t"))
    _arm(out, arm["new_tag"], delta=arm["ref_delta"] + 0.30, spread=0.02)   # a move of 0.30
    _stats(out, arm["name"]); _stats(out, arm["ref"])
    M.score_one(out, arm, "feat-138", [])
    # Caution (ar): a guard that greps raw output guards where the lines break. Normalise here,
    # never reflow the message to suit the test.
    t1 = " ".join(capsys.readouterr().out.split())
    assert "BEYOND" in t1 and "never overrides G1" in t1

    larm = dict(next(a for a in M.LADDER_ARMS if a["name"] == "pleias12bhb"))
    _arm(out, larm["new_tag"], delta=larm["ref_delta"] + 0.30, spread=0.02)
    _stats(out, larm["name"]); _stats(out, larm["ref_stats"])
    M.score_one(out, larm, "feat-140", [])
    t2 = " ".join(capsys.readouterr().out.split())
    assert "REPORTED, never gated" in t2 and "BEYOND" not in t2


def test_the_anchor_with_no_prior_gets_no_verdict_band(tmp_path, capsys):
    """P4: KL3M-3.7B had no measurement when this was registered, so no band was invented."""
    out = str(tmp_path)
    arm = dict(next(a for a in M.LADDER_ARMS if a["name"] == "kl3m37bhb"))
    assert arm["predict"] is None and arm["ref_delta"] is None
    _arm(out, arm["new_tag"], delta=0.05)
    _stats(out, arm["name"]); _stats(out, arm["ref_stats"])
    M.score_one(out, arm, "feat-140", [])
    txt = " ".join(capsys.readouterr().out.split())
    assert "no verdict was predicted" in txt and "verdict agrees" not in txt


def test_an_unfinished_arm_is_refused_and_its_band_never_computed(tmp_path, capsys):
    out = str(tmp_path)
    M.score_one(out, dict(M.SEED_ARMS[0]), "feat-138", [])
    txt = " ".join(capsys.readouterr().out.split())
    assert "NOT SCORED" in txt and "just to see" in txt
    assert "half-widths" not in txt


def test_the_constants_are_imported_from_the_breadth_scorer_not_retyped():
    """A constant copied into a second file can drift from the one it is meant to be."""
    import analysis.score_breadth_ladders as B
    assert M.MAX_SEED_MOVE is B.MAX_SEED_MOVE
    assert M.HALF_WIDTHS_FOR_STABLE is B.HALF_WIDTHS_FOR_STABLE
    assert M.BAND_SEED is B.BAND_SEED
    src = open("analysis/score_seed_and_ladder.py", encoding="utf-8").read()
    assert "MAX_SEED_MOVE = 0" not in src and "HALF_WIDTHS_FOR_STABLE = " not in src


def test_every_registered_arm_matches_its_preregistration_band():
    """The bands in the code must be the bands in the committed markdown."""
    txt = " ".join(open("results/onset_prediction_breadth_seed_replication.md",
                        encoding="utf-8").read().split())
    for arm in M.SEED_ARMS:
        assert f"{arm['ref_delta']:+.4f}".replace("+", "") in txt.replace("+", ""), arm["name"]
        assert arm["predict"] in txt, arm["name"]


def test_I1_refuses_a_reference_drawn_from_an_INCOMPLETE_arm(tmp_path):
    """h1.py writes a class only when it finishes, so an arm still generating yields a partial row.

    On 2026-09-20 the local KL3M-3.7B reference read 350 prompts because its `creative` class had
    not been written, and the comparison ran and PASSED at +2.6% -- a mean over 350 prompts set
    against a mean over 500. Nothing else in the scorer would have noticed.
    """
    out = str(tmp_path)
    _stats(out, "ref", nonempty=70.9, empty=0, n=350)
    _stats(out, "new", nonempty=72.7, empty=0, n=500)
    ok, msg = M.i1_length(M.stat_row(out, "new"), M.stat_row(out, "ref"))
    assert ok is False
    assert "INCOMPLETE" in msg and "350" in msg and "500" in msg


def test_I1_refuses_two_arms_that_agree_with_each_other_but_are_both_short(tmp_path):
    """The other half: equal counts are not enough, they must be the registered 500."""
    out = str(tmp_path)
    _stats(out, "ref", nonempty=70.0, empty=0, n=350)
    _stats(out, "new", nonempty=70.0, empty=0, n=350)
    ok, msg = M.i1_length(M.stat_row(out, "new"), M.stat_row(out, "ref"))
    assert ok is False and "not the registered" in msg


def test_an_incomplete_arm_is_NOT_described_as_a_length_disagreement(tmp_path, capsys):
    """Caution (at)'s third rule, applied to this scorer: the reason must match the failure."""
    out = str(tmp_path)
    arm = dict(next(a for a in M.LADDER_ARMS if a["name"] == "kl3m37bhb"))
    _arm(out, arm["new_tag"], delta=0.02)
    _stats(out, arm["name"], n=500); _stats(out, arm["ref_stats"], n=350)
    M.score_one(out, arm, "feat-140", [])
    txt = " ".join(capsys.readouterr().out.split())
    assert "not the finished 500-prompt arm" in txt
    assert "length given non-empty disagrees" not in txt
