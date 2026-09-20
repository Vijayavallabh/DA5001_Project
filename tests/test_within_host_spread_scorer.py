"""feat-141's scorer, exercised before its arms produce anything.

This scorer exists to be able to REFUTE a conclusion this project already published to its own
scoring logs. The first two tests are that refutation, in both directions: it must fire when a
within-host pair at Pleias-3B reaches the registered threshold, and it must not fire below it.
"""
import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analysis.score_within_host_spread as M  # noqa: E402

JB = "Phi-3.5-mini-instruct"


def _draw(out, tag, delta, spread=0.04):
    per = [{"judge": JB, "prompt_id": f"p{i:03d}", "u_n1": 0.4, "u_n8": 0.4,
            "u_n64": 0.4 + delta + (spread if i % 2 else -spread)} for i in range(500)]
    for path, rows_ in ((f"selection_scaling_per_prompt_{tag}.csv", per),
                        (f"selection_scaling_{tag}.csv",
                         [{"judge": JB, "n": n, "gain": 0.0, "gain_lo95": 0.0, "gain_hi95": 0.0,
                           "kl_nats": 0.0, "n_prompts": 500, "mean_words": 80.0} for n in M.GRID])):
        p = os.path.join(out, path)
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_[0])); w.writeheader(); w.writerows(rows_)


def _pleias3b(out, d42, d62, d72):
    for (s, tag), d in zip(M.DRAWS["pleias3bhb"], (d42, d62, d72)):
        _draw(out, tag, d)


def test_P1_REFUTES_when_a_within_host_pair_at_pleias3b_reaches_the_threshold(tmp_path, capsys,
                                                                              monkeypatch):
    out = str(tmp_path)
    _pleias3b(out, 0.0730, 0.0030, 0.0400)      # the 42-62 pair moves 0.0700
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])
    M.main()
    txt = " ".join(capsys.readouterr().out.split())
    assert "REFUTES feat-140" in txt
    assert "conclusion is WITHDRAWN" in txt


def test_P1_SURVIVES_when_every_within_host_pair_stays_below_it(tmp_path, capsys, monkeypatch):
    out = str(tmp_path)
    _pleias3b(out, 0.0730, 0.0700, 0.0690)      # widest pair 0.0040
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])
    M.main()
    txt = " ".join(capsys.readouterr().out.split())
    assert "SURVIVES so far" in txt and "WITHDRAWN" not in txt


def test_P1_is_NOT_READ_on_a_single_draw(tmp_path, capsys, monkeypatch):
    """A conclusion is neither confirmed nor withdrawn on data that does not exist."""
    out = str(tmp_path)
    _draw(out, M.DRAWS["pleias3bhb"][0][1], 0.0730)
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])
    M.main()
    txt = " ".join(capsys.readouterr().out.split())
    assert "NOT READ" in txt and "REFUTES" not in txt and "SURVIVES" not in txt


def test_P3_needs_three_draws_and_fails_if_the_stable_anchor_drops_below_the_boundary(tmp_path,
                                                                                      capsys,
                                                                                      monkeypatch):
    out = str(tmp_path)
    for (s, tag), d in zip(M.DRAWS["comma1t"], (0.0970, 0.0860, 0.0030)):   # the third draw lands at ~0.9 half-widths
        _draw(out, tag, d, spread=0.04)
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])
    M.main()
    txt = " ".join(capsys.readouterr().out.split())
    assert "P3 FAILS" in txt and "on the boundary" in txt


def test_the_cross_host_moves_are_the_ones_feat140_actually_recorded():
    """These are quoted from another arm's result and must match it, not be re-derived here."""
    import csv as _csv
    rows_ = {r["arm"]: r for r in _csv.DictReader(
        open("results/seed_and_ladder_scoring.csv", encoding="utf-8"))}
    for name, expected in M.CROSS_HOST.items():
        got = rows_.get(name, {}).get("move", "")
        if got:
            assert abs(float(got) - expected) < 5e-4, (name, got, expected)


def test_the_refutation_threshold_is_pleias3bs_own_cross_host_move_not_a_new_constant():
    assert M.REFUTES_AT == M.CROSS_HOST["pleias3bhb"]


def test_constants_are_imported_from_the_breadth_scorer():
    import analysis.score_breadth_ladders as B
    assert M.MAX_SEED_MOVE is B.MAX_SEED_MOVE and M.HALF_WIDTHS_FOR_STABLE is B.HALF_WIDTHS_FOR_STABLE
    src = open("analysis/score_within_host_spread.py", encoding="utf-8").read()
    assert "MAX_SEED_MOVE = 0" not in src


def test_the_on_record_moves_match_feat138s_scoring_csv():
    """feat-141 extends feat-138's four moves rather than replacing them, so they must agree."""
    import csv as _csv
    rows_ = {r["arm"]: r for r in _csv.DictReader(
        open("results/seed_and_ladder_scoring.csv", encoding="utf-8"))}
    by_label = {"Comma-7B (1T)": "comma1t", "Pleias-350M": "pleias350m",
                "KL3M-170M": "kl3m170m", "KL3M-520M": "kl3m520m"}
    for label, _pair, mv in M.ON_RECORD_MOVES:
        got = rows_.get(by_label[label], {}).get("move", "")
        if got:
            assert abs(float(got) - mv) < 5e-4, (label, got, mv)


def test_P3_is_NOT_READ_on_fewer_than_three_draws(tmp_path, capsys, monkeypatch):
    """Found by mutation, not by design: lowering the `len(c1) >= 3` guard to 1 left all eight
    tests green, because every P3 test supplied three draws. P3 is a claim about a THIRD reading --
    'Comma-7B (1T) reads CLIMBS for a third time' -- so two draws must not answer it either way.
    """
    out = str(tmp_path)
    for (s, tag), d in zip(M.DRAWS["comma1t"][:2], (0.0970, 0.0860)):
        _draw(out, tag, d)
    monkeypatch.setattr(sys, "argv", ["x", "--out", out])
    M.main()
    txt = " ".join(capsys.readouterr().out.split())
    # Match the distinctive phrase, not a span that happens to cross the "===" banner between
    # the heading and the line (caution (ar): a guard on raw output guards where things break).
    assert "NOT READ: 2 of 3 draws finished" in txt
    assert "P3 HOLDS" not in txt and "P3 FAILS" not in txt
