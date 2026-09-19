"""feat-137's scorer, exercised on synthetic CSVs before the arm produces anything.

Caution (v): mutation-test a gate before the data exist, not after. The gate that matters most here
is G0, the FLOOR gate, because it is the only thing standing between "this anchor cannot do GSM8K"
and the far more interesting claim "selection saturates on this anchor". Those are different findings
and only one of them is true, so the gate is tested in both directions.
"""
import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.score_verifiable_comma1t import (CROSS, FLOOR, GRID, MAJORITY,  # noqa: E402
                                               N_PROBLEMS, POINTWISE, band, g0_floor,
                                               g1_coverage, main, verdict_for)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ["arm", "n", "budget_nats", "n_problems", "acc", "acc_lo95", "acc_hi95",
        "gain", "gain_lo95", "gain_hi95", "spearman_acc_logn"]


def _rows(acc1=0.32, maj64=(0.542, 0.222, 0.182, 0.264), pw64=(0.386, 0.066, 0.024, 0.108),
          grid=GRID, n_problems=N_PROBLEMS):
    out = []
    for rule, top in ((MAJORITY, maj64), (POINTWISE, pw64)):
        for n in grid:
            last = n == max(grid)
            acc, g, lo, hi = top if last else (acc1, 0.0, 0.0, 0.0)
            out.append({"arm": rule, "n": n, "budget_nats": 0.0, "n_problems": n_problems,
                        "acc": acc if n > 1 else acc1, "acc_lo95": 0.0, "acc_hi95": 1.0,
                        "gain": g if n > 1 else 0.0, "gain_lo95": lo if n > 1 else 0.0,
                        "gain_hi95": hi if n > 1 else 0.0, "spearman_acc_logn": 0.9})
    return out


def _write(out, rows_):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "selection_verifiable_comma1t.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows_)


# ---- G0, the floor gate -------------------------------------------------------------------------

def test_the_floor_gate_separates_cannot_do_the_task_from_saturates():
    """The whole reason this arm can report a meaningful null."""
    ok, msg, acc = g0_floor(_rows(acc1=0.32))
    assert ok and acc == 0.32
    ok, msg, acc = g0_floor(_rows(acc1=0.01))
    assert not ok and acc == 0.01
    # the boundary is inclusive, and it is the registered 0.05
    assert FLOOR == 0.05
    assert g0_floor(_rows(acc1=0.05))[0] is True
    assert g0_floor(_rows(acc1=0.049))[0] is False


def test_a_floored_anchor_is_UNINFORMATIVE_and_its_band_is_never_computed(tmp_path, capsys,
                                                                          monkeypatch):
    out = str(tmp_path)
    _write(out, _rows(acc1=0.01, maj64=(0.012, 0.002, -0.01, 0.014)))
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])
    assert main() == 1
    t = capsys.readouterr().out
    assert "UNINFORMATIVE" in t and "not SATURATED" in t
    assert "NO CLIMB" not in t, "a floored arm must not be given a band verdict at all"
    assert not os.path.exists(os.path.join(out, "verifiable_comma1t_scoring.csv"))


# ---- G1 and the band ----------------------------------------------------------------------------

def test_g1_rejects_a_partial_grid_a_missing_rule_and_a_wrong_problem_count():
    assert g1_coverage(_rows(grid=(1, 2, 4, 8)))[0] is False
    assert g1_coverage(_rows(n_problems=100))[0] is False
    assert g1_coverage([r for r in _rows() if r["arm"] == MAJORITY])[0] is False
    assert g1_coverage(_rows())[0] is True


def test_the_band_is_the_majority_vote_gain_at_64_and_the_verdicts_are_the_registered_ones():
    g, lo, hi, acc = band(_rows(), MAJORITY, 64)
    assert (g, lo, hi, acc) == (0.222, 0.182, 0.264, 0.542)
    assert verdict_for(0.222, 0.182, 0.264) == "CLIMBS"
    assert verdict_for(0.01, -0.03, 0.05) == "NO CLIMB"
    assert verdict_for(-0.10, -0.15, -0.05) == "TURNS OVER"


def test_the_band_reads_majority_vote_and_not_the_pointwise_reward():
    """If it ever read the pointwise arm the headline would be the rule with a scorer to
    overoptimise against, which is the one caution (ao) records turning over on TriviaQA."""
    rs = _rows(maj64=(0.542, 0.222, 0.182, 0.264), pw64=(0.33, -0.05, -0.09, -0.01))
    assert band(rs, MAJORITY, 64)[0] == 0.222
    assert band(rs, POINTWISE, 64)[0] == -0.05


# ---- the secondary ordering ---------------------------------------------------------------------

def test_an_inverted_ordering_is_announced_rather_than_passed_over(tmp_path, capsys, monkeypatch):
    out = str(tmp_path)
    _write(out, _rows(maj64=(0.35, 0.03, 0.01, 0.05), pw64=(0.50, 0.18, 0.14, 0.22)))
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])
    main()
    t = capsys.readouterr().out
    assert "THE ORDERING INVERTS" in t and "anchor-specific" in t


def test_the_ordering_holding_is_reported_too(tmp_path, capsys, monkeypatch):
    out = str(tmp_path)
    _write(out, _rows())
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])
    assert main() == 0
    t = capsys.readouterr().out
    assert "majority vote beats the pointwise reward: YES" in t
    assert "THE ORDERING INVERTS" not in t


# ---- the cross-axis table -----------------------------------------------------------------------

def test_all_four_committed_cross_axis_cases_have_a_reading_and_the_damaging_one_says_so():
    assert set(CROSS) == {("CLIMBS", "CLIMBS"), ("SATURATED BY 8", "NO CLIMB"),
                          ("SATURATED BY 8", "CLIMBS"), ("CLIMBS", "NO CLIMB")}
    assert "MOST DAMAGING" in CROSS[("CLIMBS", "NO CLIMB")]
    assert "artefact" in CROSS[("SATURATED BY 8", "CLIMBS")]
    assert "does not refute the certificate" in CROSS[("CLIMBS", "NO CLIMB")], \
        "the damaging reading must keep the clause saying what it does NOT overturn"


def test_the_cross_axis_reading_is_taken_from_feat_136s_csv(tmp_path, capsys, monkeypatch):
    out = str(tmp_path)
    _write(out, _rows())
    with open(os.path.join(out, "breadth_ladders_scoring.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["anchor", "name", "verdict"])
        w.writerow(["Comma-7B (1T)", "comma1thb", "SATURATED BY 8"])
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])
    main()
    t = capsys.readouterr().out
    assert "SATURATED BY 8" in t and "CLIMBS" in t
    assert "artefact" in t, "judged SATURATED x judge-free CLIMBS must read as a judge artefact"


def test_an_unenumerated_pair_is_reported_rather_than_forced_into_a_case(tmp_path, capsys,
                                                                        monkeypatch):
    out = str(tmp_path)
    _write(out, _rows(maj64=(0.20, -0.12, -0.18, -0.06)))      # TURNS OVER
    with open(os.path.join(out, "breadth_ladders_scoring.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["anchor", "name", "verdict"])
        w.writerow(["Comma-7B (1T)", "comma1thb", "CLIMBS"])
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])
    main()
    assert "not one of the four cases" in capsys.readouterr().out


def test_it_refuses_cleanly_with_no_data(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(tmp_path)])
    assert main() == 1
    assert "NOT SCORED" in capsys.readouterr().out
