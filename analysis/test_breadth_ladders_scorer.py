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


def _arm_csvs(out, name, diffs, mean_words_n1=80.0, grid=GRID, n_prompts=None):
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
                         "kl_nats": 0.0, "n_prompts": n_prompts, "u": 0.4, "u_lo95": 0.3,
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

def test_g0b_compares_only_host_arms_and_only_against_their_own_local_value(tmp_path):
    out = str(tmp_path)
    tc = next(a for a in ARMS if a["name"] == "tc18bhb")
    new = next(a for a in ARMS if a["name"] == "kl3m170mhb")
    _arm_csvs(out, tc["name"], _clear_climb(), mean_words_n1=73.5)      # +1.8% of 72.2
    s = list(csv.DictReader(open(os.path.join(out, "selection_scaling_tc18bhb64.csv"))))
    ok, msg = g0b(tc, s)
    assert ok and "73.5" in msg and "72.2" in msg
    assert g0b(new, s)[0] is None, "a new anchor has no local counterpart and must not be gated"


def test_g0b_fails_a_length_that_means_a_wrong_model_rather_than_host_drift(tmp_path):
    out = str(tmp_path)
    tc = next(a for a in ARMS if a["name"] == "tc18bhb")
    _arm_csvs(out, tc["name"], _clear_climb(), mean_words_n1=40.0)      # -45% of 72.2
    s = list(csv.DictReader(open(os.path.join(out, "selection_scaling_tc18bhb64.csv"))))
    assert g0b(tc, s)[0] is False


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

def _run(out, deltas_by_name, words_by_name=None):
    _gate(out, "PASS")
    for arm in ARMS:
        d = deltas_by_name.get(arm["name"])
        if d is None:
            continue
        mw = (words_by_name or {}).get(arm["name"], arm.get("local_mw1") or 80.0)
        _arm_csvs(out, arm["name"], _clear_climb(mean=d), mean_words_n1=mw)


def test_a_host_arm_that_lands_on_its_local_value_reads_AS_A_REDRAW(tmp_path, capsys, monkeypatch):
    out = str(tmp_path)
    _run(out, {"tc18bhb": 0.0880})
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])
    assert main() == 0
    txt = capsys.readouterr().out
    assert "TRANSFERS" in txt and "AS A RE-DRAW" in txt


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
    _run(out, {"tc18bhb": 0.0880 + 0.3000})       # a move of 0.3000, literal, not derived
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])
    main()
    txt = capsys.readouterr().out
    assert "LARGER THAN ANY SEED MOVE" in txt
    assert "hardware is NOT" in txt


def test_the_bound_for_a_stable_anchor_is_the_tighter_one(tmp_path, capsys, monkeypatch):
    """Both host anchors sit above 2 half-widths locally, so their committed bound is 0.0130, not
    0.0610. A move of 0.03 must therefore NOT read as a re-draw."""
    out = str(tmp_path)
    _run(out, {"comma7bhb": 0.1010 + 0.0300})     # a move of 0.0300, literal: inside 0.0610,
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])   # outside 0.0130
    main()
    txt = capsys.readouterr().out
    assert "0.0130" in txt, "the tighter bound must be the one quoted for a stable anchor"
    assert "AS A RE-DRAW" not in txt
    assert "WITHIN THE SEED RANGE" in txt


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
    _run(out, {"tc18bhb": 0.0880})
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])
    main()
    r = list(csv.DictReader(open(os.path.join(out, "breadth_ladders_scoring.csv"))))
    assert len(r) == len(ARMS), "an arm that could not be scored must still appear, with its reason"
    by = {x["name"]: x for x in r}
    assert by["tc18bhb"]["verdict"] == "TRANSFERS"
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
