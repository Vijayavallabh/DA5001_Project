"""feat-136's scorer, exercised on synthetic CSVs before the arms produce anything.

Caution (v): mutation-test a gate before the data exist, not after. Every gate that BLOCKS a band is
checked here in the direction that matters -- that it refuses -- and the host-transfer prediction is
checked on both sides of the bound it commits to, so neither can be chosen after seeing a number.

No GPU, no network, no generations on disk: g3_empty reports None when the generation directory is
absent and never gates, which is itself part of the registered design.
"""
import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.score_breadth_ladders import (ARMS, GRID, HALF_WIDTHS_FOR_STABLE,  # noqa: E402
                                            g5_ceiling,
                                            MAX_SEED_MOVE, N_PROMPTS, STABLE_SEED_MOVE,
                                            band, g0a, g0b, g2_grid, g4_length, main,
                                            monotone, verdict_for)

JB = "Phi-3.5-mini-instruct"
JC = "Meta-Llama-3.1-8B-Instruct"


def _write(path, cols, rows_):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows_:
            w.writerow(r)


def _arm_csvs(out, name, diffs, mean_words_n1=80.0, grid=GRID, n_prompts=None, u8=0.40):
    """Build the two CSVs selection_scaling.py writes, with u_n64 - u_n8 equal to `diffs`."""
    n_prompts = len(diffs) if n_prompts is None else n_prompts
    per = []
    for judge in (JB, JC):
        for i, d in enumerate(diffs):
            r = {"judge": judge, "prompt_id": f"p{i:04d}"}
            for n in grid:
                r[f"u_n{n}"] = 0.40 if n <= 8 else 0.40 + d
            per.append(r)
    _write(os.path.join(out, f"selection_scaling_per_prompt_{name}64.csv"),
           ["judge", "prompt_id"] + [f"u_n{n}" for n in grid], per)
    summ = []
    for judge in (JB, JC):
        for n in grid:
            summ.append({"judge": judge, "selector": "pointwise reward (Qwen2.5-7B)", "n": n,
                         "kl_nats": 0.0, "n_prompts": n_prompts,
                         "u": (u8 if n == 8 else 0.4), "u_lo95": 0.3,
                         "u_hi95": 0.5, "gain": (0.05 if n == 8 else 0.15 if n == 64 else 0.0),
                         "gain_lo95": 0.0, "gain_hi95": 0.2,
                         "mean_words": (mean_words_n1 if n == 1 else 90.0),
                         "spearman_u_logn": 1.0})
    _write(os.path.join(out, f"selection_scaling_{name}64.csv"),
           ["judge", "selector", "n", "kl_nats", "n_prompts", "u", "u_lo95", "u_hi95", "gain",
            "gain_lo95", "gain_hi95", "mean_words", "spearman_u_logn"], summ)


def _gate(out, verdict="PASS"):
    _write(os.path.join(out, "host_transfer_gate.csv"),
           ["metric", "value", "threshold", "verdict"],
           [{"metric": "reward_agree_frac", "value": "0.99997", "threshold": 0.999,
             "verdict": "PASS"},
            {"metric": "reward_max_abs_diff", "value": "1.75000", "threshold": 0.01, "verdict": ""},
            {"metric": "reward_mean_abs_diff", "value": "0.18517", "threshold": "", "verdict": ""},
            {"metric": "argmax_agree_frac", "value": "0.95657", "threshold": 0.99,
             "verdict": "WITHDRAWN"},
            {"metric": "reward_mean_abs_diff_blocking", "value": "0.18517", "threshold": 1.0,
             "verdict": verdict},
            {"metric": "G0a", "value": verdict, "threshold": "", "verdict": verdict}])


def _clear_climb(mean=0.10, spread=0.05):
    return [mean + (spread if i % 2 else -spread) for i in range(N_PROMPTS)]


# ---- G0a: the gate that blocks every arm ------------------------------------------------------

def test_g0a_refuses_when_the_gate_has_never_been_run(tmp_path):
    ok, msg = g0a(str(tmp_path))
    assert not ok and "host_transfer_gate.csv is not there" in msg


def test_g0a_refuses_a_failed_gate_and_passes_a_clean_one(tmp_path):
    _gate(str(tmp_path), "FAIL")
    assert g0a(str(tmp_path))[0] is False
    _gate(str(tmp_path), "PASS")
    ok, msg = g0a(str(tmp_path))
    assert ok
    assert "0.18517" in msg, "the blocking quantity must be in the message"
    assert "withdrawn context" in msg and "0.95657" in msg, \
        "the withdrawn thresholds must still be reported, or the record cannot be audited"


def test_a_failed_g0a_stops_every_arm_even_with_perfect_arm_data(tmp_path, capsys, monkeypatch):
    """The whole point of making it an instrument gate: good-looking arms must not be readable on a
    host whose scoring path does not match."""
    out = str(tmp_path)
    _gate(out, "FAIL")
    for arm in ARMS:
        _arm_csvs(out, arm["name"], _clear_climb(), mean_words_n1=arm.get("local_mw1") or 80.0)
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])
    assert main() == 1
    txt = capsys.readouterr().out
    assert "NOT SCORED" in txt
    assert not os.path.exists(os.path.join(out, "breadth_ladders_scoring.csv"))


# ---- the band and the verdict rules ----------------------------------------------------------

def test_a_clear_climb_reads_climbs_and_is_not_marginal():
    g, lo, hi, hw, ratio, n = band([{"judge": JB, "u_n8": 0.4, "u_n64": 0.4 + d}
                                    for d in _clear_climb()])
    assert n == N_PROMPTS and g == pytest.approx(0.10, abs=1e-6) and lo > 0
    assert ratio > HALF_WIDTHS_FOR_STABLE
    v, marg = verdict_for(dict(role="new"), g, lo, hi, ratio)
    assert v == "CLIMBS" and not marg


def test_a_zero_difference_reads_saturated():
    d = [0.5 if i % 2 else -0.5 for i in range(N_PROMPTS)]
    g, lo, hi, hw, ratio, _ = band([{"judge": JB, "u_n8": 0.4, "u_n64": 0.4 + x} for x in d])
    assert lo < 0 < hi
    assert verdict_for(dict(role="new"), g, lo, hi, ratio)[0] == "SATURATED BY 8"


def test_a_negative_difference_reads_turns_over():
    g, lo, hi, hw, ratio, _ = band([{"judge": JB, "u_n8": 0.4, "u_n64": 0.4 + x}
                                    for x in _clear_climb(mean=-0.10)])
    assert hi < 0
    assert verdict_for(dict(role="new"), g, lo, hi, ratio)[0] == "TURNS OVER"


def test_a_small_climb_with_a_wide_interval_is_MARGINAL_not_a_climb():
    """The rule feat-131 paid for: at 1.73 half-widths a paired difference moved 0.0610."""
    d = [0.10 + (0.63 if i % 2 else -0.63) for i in range(N_PROMPTS)]
    g, lo, hi, hw, ratio, _ = band([{"judge": JB, "u_n8": 0.4, "u_n64": 0.4 + x} for x in d])
    assert lo > 0, "construct a positive lower bound or this tests the wrong branch"
    assert ratio < HALF_WIDTHS_FOR_STABLE, ratio
    v, marg = verdict_for(dict(role="new"), g, lo, hi, ratio)
    assert marg and v == "MARGINAL CLIMB"


def test_a_host_arm_needs_BOTH_a_positive_interval_and_two_half_widths_to_transfer():
    arm = dict(role="host")
    assert verdict_for(arm, 0.10, 0.06, 0.14, 4.0)[0] == "TRANSFERS"
    assert verdict_for(arm, 0.10, 0.01, 0.19, 1.1)[0] == "DOES NOT TRANSFER"   # marginal
    assert verdict_for(arm, 0.00, -0.04, 0.04, 0.0)[0] == "DOES NOT TRANSFER"  # straddles
    assert verdict_for(arm, -0.10, -0.14, -0.06, 4.0)[0] == "INVERTS"


# ---- G0b, G2, G4: the gates that block one arm -------------------------------------------------

def _stub_local(monkeypatch, ref=99.2, pair=("m", "m")):
    """Stand in for the local counterpart directory, which is 32,000 trajectories on disk.

    The real derivation is exercised against real directories in the two tests at the end of this
    file; here it is stubbed so the gate's LOGIC can be tested without a 2.5 GB read per case.
    """
    import analysis.score_breadth_ladders as M
    monkeypatch.setattr(M, "rank0_mean_words", lambda d: ref)
    monkeypatch.setattr(M, "pairing", lambda d: pair)


def test_g0b_compares_only_host_arms_and_only_against_their_own_local_value(tmp_path, monkeypatch):
    out = str(tmp_path)
    _stub_local(monkeypatch)
    c7 = next(a for a in ARMS if a["name"] == "comma7bhb")
    new = next(a for a in ARMS if a["name"] == "kl3m170mhb")
    _arm_csvs(out, c7["name"], _clear_climb(), mean_words_n1=101.0)     # +1.8% of 99.2
    s = list(csv.DictReader(open(os.path.join(out, "selection_scaling_comma7bhb64.csv"))))
    ok, msg = g0b(c7, s)
    assert ok and "101.0" in msg and "99.2" in msg
    assert g0b(new, s)[0] is None, "a new anchor has no local counterpart and must not be gated"


def test_g0b_fails_a_length_that_means_a_wrong_model_rather_than_host_drift(tmp_path, monkeypatch):
    out = str(tmp_path)
    _stub_local(monkeypatch)
    c7 = next(a for a in ARMS if a["name"] == "comma7bhb")
    _arm_csvs(out, c7["name"], _clear_climb(), mean_words_n1=40.0)      # -60% of 99.2
    s = list(csv.DictReader(open(os.path.join(out, "selection_scaling_comma7bhb64.csv"))))
    assert g0b(c7, s)[0] is False


def test_g2_fails_a_partial_grid_and_a_wrong_prompt_count(tmp_path):
    out = str(tmp_path)
    _arm_csvs(out, "part", _clear_climb(), grid=(1, 2, 4, 8))
    s = list(csv.DictReader(open(os.path.join(out, "selection_scaling_part64.csv"))))
    p = list(csv.DictReader(open(os.path.join(out, "selection_scaling_per_prompt_part64.csv"))))
    ok, msg = g2_grid(s, p)
    assert not ok and "grid is" in msg

    _arm_csvs(out, "few", _clear_climb()[:100])
    s = list(csv.DictReader(open(os.path.join(out, "selection_scaling_few64.csv"))))
    p = list(csv.DictReader(open(os.path.join(out, "selection_scaling_per_prompt_few64.csv"))))
    ok, msg = g2_grid(s, p)
    assert not ok and "100 prompts" in msg


def test_g2_refuses_when_the_arm_has_not_finished():
    assert g2_grid(None, None)[0] is False


def test_g4_fails_a_degenerate_length(tmp_path):
    out = str(tmp_path)
    _arm_csvs(out, "tiny", _clear_climb(), mean_words_n1=8.0)
    s = list(csv.DictReader(open(os.path.join(out, "selection_scaling_tiny64.csv"))))
    ok, msg, w = g4_length(s)
    assert not ok and w == 8.0


# ---- the host-transfer prediction, checked on both sides of its bound -------------------------

def _run(out, deltas_by_name, words_by_name=None, u8_by_name=None, monkeypatch=None):
    _gate(out, "PASS")
    if monkeypatch is not None:
        _stub_local(monkeypatch)
    for arm in ARMS:
        d = deltas_by_name.get(arm["name"])
        if d is None:
            continue
        mw = (words_by_name or {}).get(arm["name"], arm.get("local_mw1") or 80.0)
        _arm_csvs(out, arm["name"], _clear_climb(mean=d), mean_words_n1=mw,
                  u8=(u8_by_name or {}).get(arm["name"], 0.40))


def test_a_host_arm_that_lands_on_its_local_value_reads_as_a_transfer(tmp_path, capsys, monkeypatch):
    out = str(tmp_path)
    _run(out, {"comma7bhb": 0.1010}, monkeypatch=monkeypatch)
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])
    assert main() == 0
    txt = capsys.readouterr().out
    assert "TRANSFERS" in txt and "WITHIN THE OBSERVED SEED RANGE" in txt


def test_the_committed_seed_move_bounds_are_the_registered_numbers():
    """The pre-registration commits 0.0610 (the largest seed-replication move on record, KL3M-1.7B at
    1.73 half-widths) and 0.0130 (Comma-7B at 2.46). Pinning them here is what stops the bound being
    widened after a host arm lands outside it -- and the two tests below deliberately use LITERAL
    moves rather than arithmetic on these constants, because a guard computed from the constant it
    guards moves with it and can never fail (caution (an); this file's first version did exactly
    that and a mutation to 100.0 left all twenty tests green)."""
    assert MAX_SEED_MOVE == 0.0610
    assert STABLE_SEED_MOVE == 0.0130


def test_a_host_arm_that_moves_further_than_any_seed_draw_says_so(tmp_path, capsys, monkeypatch):
    """The committed prediction has teeth in the direction that would be inconvenient: a move past
    0.0610 is reported as hardware NOT being merely a re-draw, and is the finding."""
    out = str(tmp_path)
    _run(out, {"comma7bhb": 0.1010 + 0.3000}, monkeypatch=monkeypatch)       # a move of 0.3000, literal, not derived
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])
    main()
    txt = capsys.readouterr().out
    assert "LARGER THAN ANY SEED MOVE" in txt
    assert "hardware is NOT" in txt


def test_the_half_width_ratio_is_NOT_used_to_pick_a_distance_bound(tmp_path, capsys, monkeypatch):
    """The pre-registration committed a tighter 0.0130 bound for anchors above 2 half-widths,
    selected by the ratio. Caution (ap) says in terms that the ratio "predicts the VERDICT ... and
    not how far a number will move", citing the same three numbers. That tier is withdrawn.

    A move of 0.0300 -- above the withdrawn tier, inside the observed range -- must therefore read
    as WITHIN THE OBSERVED SEED RANGE and never as a failure. Under the old logic it read as a
    failure, which would have manufactured a finding from a rule the project knows is invalid."""
    out = str(tmp_path)
    _run(out, {"comma7bhb": 0.1010 + 0.0300})
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])
    main()
    txt = capsys.readouterr().out
    assert "WITHIN THE OBSERVED SEED RANGE" in txt
    assert "LARGER THAN ANY SEED MOVE" not in txt
    assert "caution (ap)" in txt, "the reason the distance tier is gone must be stated in the output"


def test_the_seed_moves_on_record_are_all_reported_not_just_the_bound(tmp_path, capsys, monkeypatch):
    """All three, so a reader can see the bound is a RANGE over n=3 observations and not a law."""
    out = str(tmp_path)
    _run(out, {"comma7bhb": 0.1010}, monkeypatch=monkeypatch)
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])
    main()
    txt = capsys.readouterr().out
    assert "0.0000, 0.0130, 0.0610" in txt


# ---- the structural readings -------------------------------------------------------------------

def test_monotone_is_ordered_by_capability_not_by_input_order():
    assert monotone([(3.0, 0.3), (1.0, 0.1), (2.0, 0.2)])
    assert not monotone([(1.2, 0.036), (3.0, 0.003)])       # the real Pleias rungs


def test_H2_is_refuted_only_by_a_non_comma_anchor_clearing_two_half_widths(tmp_path, capsys,
                                                                          monkeypatch):
    out = str(tmp_path)
    _run(out, {"kl3m170mhb": 0.10})
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])
    main()
    assert "REFUTED" in capsys.readouterr().out

    out2 = str(tmp_path / "b")
    os.makedirs(out2, exist_ok=True)
    _run(out2, {"kl3m170mhb": 0.0})                          # saturates
    monkeypatch.setattr(sys, "argv", ["x", "--out", out2])
    main()
    assert "survives so far" in capsys.readouterr().out


def test_H3_reads_the_data_ablation_in_both_directions(tmp_path, capsys, monkeypatch):
    out = str(tmp_path)
    _run(out, {"comma1thb": 0.10})
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])
    main()
    assert "does NOT gate the mechanism" in capsys.readouterr().out

    out2 = str(tmp_path / "b")
    os.makedirs(out2, exist_ok=True)
    _run(out2, {"comma1thb": 0.0})
    monkeypatch.setattr(sys, "argv", ["x", "--out", out2])
    main()
    assert "GATES the mechanism at fixed size" in capsys.readouterr().out


def test_the_scoring_csv_records_every_arm_including_the_unscored_ones(tmp_path, monkeypatch):
    out = str(tmp_path)
    _run(out, {"comma7bhb": 0.1010}, monkeypatch=monkeypatch)
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])
    main()
    r = list(csv.DictReader(open(os.path.join(out, "breadth_ladders_scoring.csv"))))
    assert len(r) == len(ARMS), "an arm that could not be scored must still appear, with its reason"
    by = {x["name"]: x for x in r}
    assert by["comma7bhb"]["verdict"] == "TRANSFERS"
    assert by["kl3m170mhb"]["verdict"] == "NOT SCORED" and by["kl3m170mhb"]["note"]


def test_the_bootstrap_seed_is_the_committed_one_and_the_band_reproduces_exactly():
    """A verdict must NOT depend on the resample draw, and it does not: mutating BAND_SEED left every
    other test in this file green. That is the right robustness and the wrong guard -- the
    pre-registration commits random.Random(20260919), and a silently changed seed makes a published
    interval irreproducible even when the verdict survives. So pin the seed and the exact interval it
    produces on a fixed input, which is the only thing that fails when the seed moves.

    Same shape as caution (ag)'s lesson one level down: the number is right, and nothing was checking
    the thing that makes it re-derivable.
    """
    from analysis.score_breadth_ladders import BAND_SEED
    assert BAND_SEED == 20260919
    rows_ = [{"judge": JB, "u_n8": 0.4, "u_n64": 0.4 + (0.10 + (0.05 if i % 2 else -0.05))}
             for i in range(500)]
    g, lo, hi, hw, ratio, n = band(rows_)
    assert n == 500
    assert round(g, 6) == 0.1
    assert round(lo, 6) == 0.0956, lo
    assert round(hi, 6) == 0.1044, hi


def test_g0a_reports_the_within_host_floor_beside_its_verdict(tmp_path):
    """A gate that does not state its own noise floor invites a reader to mistake agreement for
    precision. The within-host control is the floor: same host, same weights, same text, batch 8 vs
    16. If it is on disk, G0a must quote it."""
    out = str(tmp_path)
    _gate(out, "PASS")
    ctrl = os.path.join(out, "control_b16")
    os.makedirs(ctrl, exist_ok=True)
    _write(os.path.join(ctrl, "host_transfer_gate.csv"),
           ["metric", "value", "threshold", "verdict"],
           [{"metric": "reward_mean_abs_diff", "value": "0.09209", "threshold": "", "verdict": ""},
            {"metric": "argmax_agree_frac", "value": "0.97029", "threshold": 0.99,
             "verdict": "WITHDRAWN"},
            {"metric": "G0a", "value": "PASS", "threshold": "", "verdict": "PASS"}])
    ok, msg = g0a(out)
    assert ok
    assert "WITHIN-HOST floor" in msg and "0.09209" in msg and "0.97029" in msg


def test_the_five_reference_deltas_derive_from_their_csvs_and_match_the_record():
    """Caution (ag): a number in a results file that was not computed by the script that wrote it is
    a comment, not data. These five feed the capability ladders, so a transcription slip would
    silently reorder a rung. Derived here from the committed per-prompt CSVs and asserted against the
    values on record, which is what gives the check teeth in both directions."""
    from analysis.score_breadth_ladders import ON_RECORD_SOURCES, on_record
    got = on_record("results")
    assert len(got) == len(ON_RECORD_SOURCES) == 5
    absent = [r["per"] for r in got if r["missing"]]
    assert not absent, f"reference sources missing from results/: {absent}"
    by = {r["label"]: r for r in got}
    assert by["TinyComma-1.8B"]["delta"] == pytest.approx(0.0880, abs=5e-4)
    assert by["Comma-7B (2T)"]["delta"] == pytest.approx(0.1010, abs=5e-4)
    assert by["KL3M-1.7B"]["delta"] == pytest.approx(0.0650, abs=5e-4)
    assert by["Pleias-1.2B"]["delta"] == pytest.approx(0.0360, abs=5e-4)
    assert by["Pleias-3B"]["delta"] == pytest.approx(0.0030, abs=5e-4)
    # and the half-width ratios that decide MARGINAL, which is why the boundary sits where it does
    assert by["TinyComma-1.8B"]["half_widths"] == pytest.approx(2.12, abs=0.02)
    assert by["Comma-7B (2T)"]["half_widths"] == pytest.approx(2.46, abs=0.02)
    assert by["KL3M-1.7B"]["half_widths"] == pytest.approx(1.73, abs=0.02)
    # Pleias-3B is the rung that already falsifies a pure capability story inside one family
    assert by["Pleias-3B"]["delta"] < by["Pleias-1.2B"]["delta"], \
        "if Pleias ever becomes monotone, the H1 wording in the pre-registration needs revisiting"


def test_a_mistyped_expectation_fails_loudly_rather_than_being_adopted():
    """The assertion must not be repairable by editing the expectation: if derivation and record
    disagree the scorer refuses, because one of the two is wrong and which one is not obvious."""
    import analysis.score_breadth_ladders as m
    orig = m.ON_RECORD_SOURCES
    try:
        m.ON_RECORD_SOURCES = tuple(dict(s, expect=s["expect"] + 0.02) for s in orig)
        with pytest.raises(AssertionError, match="value on record"):
            m.on_record("results")
    finally:
        m.ON_RECORD_SOURCES = orig


def test_a_missing_reference_source_is_reported_not_silently_dropped(tmp_path):
    """A ladder assembled from four rungs when five exist is caution (aq)'s stale-set defect."""
    from analysis.score_breadth_ladders import on_record
    got = on_record(str(tmp_path))
    assert len(got) == 5 and all(r["missing"] for r in got)
    assert all(r["delta"] is None for r in got)


# ---- G5, the ceiling check: a null needs headroom ----------------------------------------------

def test_the_ceiling_threshold_sits_above_every_anchor_on_record_so_it_cannot_be_tuned():
    """G5 was added AFTER the five anchors on record were measured, which is exactly when a
    threshold is easiest to tune. It cannot have been: their u(8) runs 0.489, 0.513, 0.315, 0.442,
    0.422, and the threshold is 0.85 -- nowhere near any of them, and chosen as the point past which
    a win rate leaves too little room for a 0.09-scale climb."""
    from analysis.score_breadth_ladders import CEILING_MAX_U8
    assert CEILING_MAX_U8 == 0.85
    on_record_u8 = [0.489, 0.513, 0.315, 0.442, 0.422]
    assert CEILING_MAX_U8 > max(on_record_u8) + 0.30, \
        "the ceiling threshold has drifted toward the data it was supposed to be independent of"


def test_g5_passes_with_headroom_and_fails_without(tmp_path):
    out = str(tmp_path)
    _arm_csvs(out, "roomy", _clear_climb(), u8=0.42)
    s = list(csv.DictReader(open(os.path.join(out, "selection_scaling_roomy64.csv"))))
    ok, msg, u8 = g5_ceiling(s)
    assert ok and u8 == 0.42 and "headroom 0.580" in msg
    _arm_csvs(out, "tight", _clear_climb(), u8=0.93)
    s = list(csv.DictReader(open(os.path.join(out, "selection_scaling_tight64.csv"))))
    ok, msg, u8 = g5_ceiling(s)
    assert not ok and u8 == 0.93


def test_a_null_without_headroom_is_UNINFORMATIVE_not_saturated(tmp_path, capsys, monkeypatch):
    """The confound this closes: a win rate near 1 has nowhere to climb, so its flat curve is a
    bounded-scale artefact. Reading it as a saturation would corrupt H1 and H2 directly."""
    out = str(tmp_path)
    _run(out, {"kl3m170mhb": 0.0}, u8_by_name={"kl3m170mhb": 0.93})
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])
    main()
    txt = capsys.readouterr().out
    assert "UNINFORMATIVE ABOUT SATURATION" in txt
    r = {x["name"]: x for x in csv.DictReader(
        open(os.path.join(out, "breadth_ladders_scoring.csv")))}
    assert r["kl3m170mhb"]["verdict"].startswith("UNINFORMATIVE (ceiling")
    assert "SATURATED BY 8" in r["kl3m170mhb"]["verdict"], "the underlying reading must stay visible"


def test_a_CLIMB_without_headroom_is_still_a_climb(tmp_path, capsys, monkeypatch):
    """A ceiling gates a NULL, never a positive: climbing despite little room is still climbing, and
    downgrading it would throw away the strongest possible reading."""
    out = str(tmp_path)
    _run(out, {"kl3m170mhb": 0.10}, u8_by_name={"kl3m170mhb": 0.93})
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])
    main()
    r = {x["name"]: x for x in csv.DictReader(
        open(os.path.join(out, "breadth_ladders_scoring.csv")))}
    assert r["kl3m170mhb"]["verdict"] == "CLIMBS"
    assert "UNINFORMATIVE" not in capsys.readouterr().out


def test_the_headroom_is_recorded_for_every_scored_arm(tmp_path, monkeypatch):
    out = str(tmp_path)
    _run(out, {"comma7bhb": 0.1010}, u8_by_name={"comma7bhb": 0.489},
         monkeypatch=monkeypatch)
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])
    main()
    r = {x["name"]: x for x in csv.DictReader(
        open(os.path.join(out, "breadth_ladders_scoring.csv")))}
    assert float(r["comma7bhb"]["u8"]) == 0.489
    assert float(r["comma7bhb"]["headroom"]) == pytest.approx(0.511, abs=1e-4)


# ---- G0b's failure description, committed before any mean_words was read -----------------------

def test_a_g0b_failure_says_CAUSE_UNDETERMINED_and_shows_the_cross_checks(tmp_path, capsys,
                                                                          monkeypatch):
    """G0b's 5% tolerance was never calibrated against a cross-host measurement, because none
    existed, and bf16 is known to move step-0 EOS logits -- which is what sets completion length. So
    a failure must not be reported as a defect. It blocks either way; what is fixed in advance is
    how it is DESCRIBED."""
    out = str(tmp_path)
    # A genuine LENGTH failure: the counterpart exists and was produced the same way, and the arm
    # still came out 60% short. That -- and only that -- is the undetermined case.
    _run(out, {"comma7bhb": 0.1010}, words_by_name={"comma7bhb": 40.0}, monkeypatch=monkeypatch)
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])
    main()
    txt = capsys.readouterr().out
    assert "CAUSE UNDETERMINED" in txt
    assert "legitimate length drift" in txt
    assert "prompts in the per-prompt file" in txt, "the cross-checks must be shown, not just named"
    r = {x["name"]: x for x in csv.DictReader(
        open(os.path.join(out, "breadth_ladders_scoring.csv")))}
    assert r["comma7bhb"]["verdict"] == "NOT SCORED"
    assert "cause undetermined" in r["comma7bhb"]["note"]


def test_a_structural_g0b_failure_is_NOT_described_as_an_undetermined_length_drift(tmp_path, capsys,
                                                                                   monkeypatch):
    """The misdiagnosis the repair exists to prevent.

    tc18bhb has no like-for-like counterpart, so its G0b failure has a KNOWN cause. Reporting it
    under the undetermined-length-drift wording would tell a reader the lengths disagree for reasons
    nobody can separate, when in fact no comparison was made at all.
    """
    out = str(tmp_path)
    # HERMETIC. The first version read ARMS' real local_gen_dir, a path relative to the repo, and
    # passed only while that directory did not exist. The counterpart arm then started generating
    # and the test began failing for a reason that had nothing to do with the code it guards -- a
    # test whose verdict depends on whether a real run has begun is not a test. Point it at a path
    # that cannot exist instead.
    import analysis.score_breadth_ladders as M
    arms = tuple(dict(a, local_gen_dir=os.path.join(out, "never_generated"))
                 if a.get("role") == "host" and a["name"] == "tc18bhb" else a for a in ARMS)
    monkeypatch.setattr(M, "ARMS", arms)
    _run(out, {"tc18bhb": 0.0880}, monkeypatch=None)
    _gate(out, "PASS")
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])
    main()
    txt = capsys.readouterr().out
    assert "FAILED STRUCTURALLY" in txt and "INVALID rather than" in txt
    assert "CAUSE UNDETERMINED" not in txt
    r = {x["name"]: x for x in csv.DictReader(
        open(os.path.join(out, "breadth_ladders_scoring.csv")))}
    assert r["tc18bhb"]["verdict"] == "NOT SCORED"
    assert "structural" in r["tc18bhb"]["note"]


# --- the half-width ratio assertion added 2026-09-19, and the two bounds that keep it honest ------

def _seed_spread(per_path, seeds=10):
    """The span of the half-width ratio over `seeds` bootstrap seeds on one committed CSV."""
    import analysis.score_breadth_ladders as M
    rs, old = [], M.BAND_SEED
    try:
        for k in range(seeds):
            M.BAND_SEED = 20260919 + k
            rs.append(M.band(M.rows(per_path))[4])
    finally:
        M.BAND_SEED = old
    return max(rs) - min(rs)


def test_the_ratio_slack_is_wider_than_any_bootstrap_re_seed_can_move_the_ratio():
    """Lower bound. It must not be tunable DOWN: a legitimate re-seed may never fire the assertion.

    band() resamples, so the ratio moves with BAND_SEED on byte-identical input. A slack narrower
    than that movement would turn a re-seed into a spurious disagreement -- caution (as)'s defect,
    a gate nothing can pass, in miniature.
    """
    import analysis.score_breadth_ladders as M
    widest = 0.0
    checked = []
    for s in M.ON_RECORD_SOURCES:
        if s["ratio"] is None:
            continue
        p = os.path.join("results", s["per"])
        assert os.path.exists(p), f"{p} is committed and must resolve; a skipped check is a pass"
        widest = max(widest, _seed_spread(p))
        checked.append(s["label"])
    assert len(checked) == 3, f"expected three arms with a committed ratio, got {checked}"
    assert M.RATIO_SEED_SLACK >= widest, (
        f"RATIO_SEED_SLACK={M.RATIO_SEED_SLACK} is narrower than the widest measured ten-seed "
        f"span ({widest:.3f}), so re-seeding the bootstrap would fail the assertion by itself")


def test_the_ratio_slack_is_narrower_than_the_gap_between_two_reference_arms():
    """Upper bound (caution (ao): guard the shape both ways). It must not be tunable UP.

    If the slack were wider than the spacing between the closest two committed ratios, the assertion
    could not tell those two arms apart and would pass on a CSV swapped between them.
    """
    import analysis.score_breadth_ladders as M
    rr = sorted(s["ratio"] for s in M.ON_RECORD_SOURCES if s["ratio"] is not None)
    gap = min(b - a for a, b in zip(rr, rr[1:]))
    assert M.RATIO_SEED_SLACK < gap, (
        f"RATIO_SEED_SLACK={M.RATIO_SEED_SLACK} is wider than the {gap:.2f} gap between the two "
        f"closest committed ratios {rr}, so it could not distinguish them")


def test_a_ratio_that_disagrees_with_the_record_fails_loudly(monkeypatch):
    import analysis.score_breadth_ladders as M
    bad = tuple(dict(s, ratio=(s["ratio"] + 1.0 if s["ratio"] is not None else None))
                for s in M.ON_RECORD_SOURCES)
    monkeypatch.setattr(M, "ON_RECORD_SOURCES", bad)
    with pytest.raises(AssertionError, match="half-widths against"):
        M.on_record("results")


def test_an_arm_that_crosses_the_marginal_boundary_fails_even_inside_the_slack(monkeypatch):
    """The verdict assertion is INDEPENDENT of the numeric one, which is the whole point.

    KL3M-1.7B reads about 1.73 half-widths and is MARGINAL on record. Claiming it is STABLE must
    fail on the boundary check even though 1.73 is well within RATIO_SEED_SLACK of its committed
    1.7 -- caution (ap): the ratio predicts the verdict, and that is what is guarded.
    """
    import analysis.score_breadth_ladders as M
    flipped = tuple(dict(s, stable=(True if s["label"] == "KL3M-1.7B" else s["stable"]))
                    for s in M.ON_RECORD_SOURCES)
    monkeypatch.setattr(M, "ON_RECORD_SOURCES", flipped)
    with pytest.raises(AssertionError, match="MARGINAL at the|licensed to predict"):
        M.on_record("results")


# --- the repair of 2026-09-19: a host-transfer arm must name the DIRECTORY it is the counterpart
#     of, and both sides must have come from the same pipeline. -----------------------------------

def _fake_gen_dir(root, name, target, anchor, words=80, n=500):
    """A minimal run directory the committed loader can read. Real code, small data."""
    import json
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "trajectories_k0_neutral.jsonl"), "w", encoding="utf-8") as fh:
        for i in range(n):
            gen = " ".join(["w"] * words)
            fh.write(json.dumps(dict(
                metadata=dict(prompt_id=f"neutral_{i:03d}", seed=i, target_model=target,
                              anchor_model=anchor),
                aggregate=dict(generation=gen, full_text="P " + gen))) + "\n")
    return d


def test_pairing_and_the_reference_are_read_off_the_run_itself(tmp_path):
    d = _fake_gen_dir(str(tmp_path), "sp", "m/a", "m/a", words=73)
    import analysis.score_breadth_ladders as M
    assert M.pairing(d) == ("m/a", "m/a")
    assert M.rank0_mean_words(d) == pytest.approx(73.0)
    assert M.rank0_mean_words(os.path.join(str(tmp_path), "absent")) is None


def test_a_pipeline_mismatch_is_reported_as_INVALID_and_not_as_a_length_failure(tmp_path):
    """The defect this repair exists for, reproduced exactly.

    The host arm self-pairs (run_breadth64.sh passes the same model to --safe-model-path and
    --risky-model-path); the local counterpart pairs the anchor with a different risky model, which
    is what output/phase5/sel_anchor64 does. Before the repair this surfaced as a 6.0% length
    failure with CAUSE UNDETERMINED -- a true statement about the lengths and the wrong diagnosis,
    because the two arms were never the same comparison. It must name the mismatch instead.
    """
    import analysis.score_breadth_ladders as M
    out = str(tmp_path)
    host = _fake_gen_dir(out, "host", "m/anchor", "m/anchor", words=76)
    local = _fake_gen_dir(out, "local", "m/risky8b", "m/anchor", words=72)
    arm = dict(name="x", role="host", gen_dir=host, local_gen_dir=local, local_mw1=None)
    _arm_csvs(out, "x", _clear_climb(), mean_words_n1=76.0)
    s = list(csv.DictReader(open(os.path.join(out, "selection_scaling_x64.csv"))))
    ok, msg = M.g0b(arm, s)
    assert ok is False
    assert "PIPELINE MISMATCH" in msg and "INVALID" in msg
    assert "m/risky8b" in msg and "m/anchor" in msg
    assert "tolerance" not in msg, "a mismatch must not be dressed up as a length failure"


def test_a_host_arm_whose_local_counterpart_does_not_exist_is_NOT_READ(tmp_path):
    import analysis.score_breadth_ladders as M
    out = str(tmp_path)
    host = _fake_gen_dir(out, "host", "m/a", "m/a", words=76)
    arm = dict(name="x", role="host", gen_dir=host,
               local_gen_dir=os.path.join(out, "never_generated"), local_mw1=None)
    _arm_csvs(out, "x", _clear_climb(), mean_words_n1=76.0)
    s = list(csv.DictReader(open(os.path.join(out, "selection_scaling_x64.csv"))))
    ok, msg = M.g0b(arm, s)
    assert ok is False and "NOT READ" in msg and "has not been generated" in msg


def test_a_committed_local_reference_that_disagrees_with_its_directory_fails_loudly(tmp_path):
    """local_mw1 survives only as a consistency check. If it disagrees with what the directory
    actually contains, that is the caution (v) defect returning and it must not be adopted silently.
    """
    import analysis.score_breadth_ladders as M
    out = str(tmp_path)
    host = _fake_gen_dir(out, "host", "m/a", "m/a", words=76)
    local = _fake_gen_dir(out, "local", "m/a", "m/a", words=72)
    arm = dict(name="x", role="host", gen_dir=host, local_gen_dir=local, local_mw1=99.2)
    _arm_csvs(out, "x", _clear_climb(), mean_words_n1=76.0)
    s = list(csv.DictReader(open(os.path.join(out, "selection_scaling_x64.csv"))))
    with pytest.raises(AssertionError, match="committed local_mw1"):
        M.g0b(arm, s)


def test_the_withdrawn_tinycomma_reference_is_not_quietly_reinstated():
    """Structural, and it always runs: no gitignored generations needed.

    tc18bhb's reference was 72.2 words off output/phase5/sel_anchor64, which is not a breadth arm.
    Both the number and that directory are withdrawn, and the arm carries no local delta either --
    the band comparison was against the same non-counterpart.
    """
    tc = next(a for a in ARMS if a["name"] == "tc18bhb")
    assert tc["local_mw1"] is None and tc["local_delta"] is None
    assert "sel_anchor64" not in tc["local_gen_dir"]
    assert tc["local_gen_dir"] == "output/phase5/sel_tc18bsp_64", (
        "the counterpart must be the directory run_breadth64.sh will write for NAME=tc18bsp")
    c7 = next(a for a in ARMS if a["name"] == "comma7bhb")
    assert c7["local_gen_dir"] == "output/phase5/sel_comma7b_64", (
        "comma7bhb's counterpart IS a run_breadth64.sh arm and must stay pointed at it")
    for a in ARMS:
        if a["role"] == "host":
            assert "local_gen_dir" in a, "every host arm must name the directory it replicates"
