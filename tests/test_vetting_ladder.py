"""feat-180's gates and verdicts, mutation-tested on synthetic rungs before any rung has run.

results/onset_prediction_vetting_ladder.md registers G0 (every rung is the same 50 passages), G1 (the
70B's L = 100 re-run reproduces the arm on record, else the on-record L = 100 rungs are not mixed in)
and four readings. Each must fire on the case it is for.
"""
import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import vetting_ladder as v  # noqa: E402

PIDS = [f"bookmia.17.{i:02d}" for i in range(50)]


def _write(d, name, leaks, col="anchor_max_recall", pids=PIDS):
    with open(os.path.join(d, f"{name}_per_passage.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["prompt_id", "risky_alone_recall", "anchor_max_recall"])
        w.writeheader()
        for i, p in enumerate(pids):
            hit = "0.5" if i < leaks else "0.0"
            w.writerow({"prompt_id": p, "risky_alone_recall": hit if col == "risky_alone_recall"
                        else "0.0", "anchor_max_recall": hit if col == "anchor_max_recall" else "0.0"})


def _base(d):
    """the arms on record: the corpus, the 70B at 25/50, OLMo-2-13B at 6/50"""
    _write(d, "vet_comma7b", 0)
    _write(d, "selection_extraction_70b_hp2", 25, col="risky_alone_recall")
    _write(d, "vet_olmo2_13b", 6)


def _run(d, capsys):
    sys.argv = ["vetting_ladder.py", "--results", str(d), "--out", str(d)]
    v.main()
    return capsys.readouterr().out


def _complete(d, skip=()):
    """every registered rung, monotone, the licensed anchors clean; `skip` leaves rungs out"""
    for L, k in ((20, 0), (35, 2), (50, 8), (75, 14), (100, 25), (150, 28), (200, 30)):
        if ("llama70b", L) not in skip:
            _write(d, f"vetladder_L{L}_llama70b", k, col="risky_alone_recall")
    _write(d, "vet_olmo2_7b", 4)
    for t in ("olmo2_13b", "olmo2_7b"):
        for L, k in ((20, 0), (50, 2), (150, 8), (200, 9)):
            if (t, L) not in skip:
                _write(d, f"vetladder_L{L}_{t}", k)
    for t, (_, ref, role) in v.MODELS.items():
        if role == "openly licensed":
            if not os.path.exists(os.path.join(d, f"{ref}_per_passage.csv")):
                _write(d, ref, 0)                          # its L = 100 screen on record
            for L in (150, 200):
                if (t, L) not in skip:
                    _write(d, f"vetladder_L{L}_{t}", 0)


def test_a_monotone_ladder_that_reproduces_reads_cleanly(tmp_path, capsys):
    _base(tmp_path)
    _complete(tmp_path)
    out = _run(tmp_path, capsys)
    assert "rungs missing: none" in out
    assert "G0 PASS" in out and "G1 PASS" in out
    assert "V1 MONOTONE" in out and "V2 L* = 50" in out and "V3 PASS HOLDS" in out
    assert "olmo2_13b  L=100  leaks on  6/50" in out and "[on record]" in out
    assert out.count("[this arm, host B]") == 2 and "host check NOT RUN" in out


def test_g1_fails_on_one_changed_draw_and_the_record_is_then_not_mixed_in(tmp_path, capsys):
    _base(tmp_path)
    _write(tmp_path, "vetladder_L100_llama70b", 24, col="risky_alone_recall")
    _write(tmp_path, "vetladder_L200_olmo2_13b", 9)
    out = _run(tmp_path, capsys)
    assert "G1 FAIL on 1 of 50" in out
    assert "olmo2_13b  L=100" not in out, "a drifted host must not borrow the on-record rung"


def test_g0_refuses_a_rung_on_the_wrong_passages(tmp_path, capsys):
    _base(tmp_path)
    _write(tmp_path, "vetladder_L200_comma7b", 0, pids=[f"bookmia.07.{i:02d}" for i in range(50)])
    out = _run(tmp_path, capsys)
    assert "FAIL, not read: comma7b L=200" in out and "V3 NOT READ" in out


def test_a_fall_of_three_passages_is_non_monotone_and_a_licensed_leak_breaks_the_pass(tmp_path,
                                                                                       capsys):
    _base(tmp_path)
    _write(tmp_path, "vetladder_L100_llama70b", 25, col="risky_alone_recall")
    _write(tmp_path, "vetladder_L150_olmo2_13b", 3)       # 6 at 100 -> 3 at 150
    _write(tmp_path, "vetladder_L150_pleias3b", 1)
    _write(tmp_path, "vet_pleias3b", 0)
    out = _run(tmp_path, capsys)
    assert "V1 NON-MONOTONE" in out and "(100, 150)" in out
    assert "V3 PASS BREAKS" in out and "every rung" in out


def test_a_fall_of_two_is_within_tolerance(tmp_path, capsys):
    _base(tmp_path)
    _complete(tmp_path)
    _write(tmp_path, "vetladder_L150_olmo2_13b", 4)        # 6 on record at 100 -> 4: a fall of two
    assert "V1 MONOTONE" in _run(tmp_path, capsys)


def test_a_missing_rung_is_not_read_as_a_pass_but_a_leak_still_breaks_it(tmp_path, capsys):
    _base(tmp_path)
    _complete(tmp_path, skip={("pleias3b", 150), ("llama70b", 35), ("olmo2_7b", 20)})
    out = _run(tmp_path, capsys)
    assert "V3 NOT READ (incomplete)" in out and "PASS HOLDS" not in out, out
    assert "V1 NOT READ (incomplete)" in out and "V2 NOT READ (incomplete)" in out
    assert "V4 NOT READ (incomplete)" in out
    _write(tmp_path, "vetladder_L200_kl3m17b", 1)         # a leak is a leak on any subset
    assert "V3 PASS BREAKS" in _run(tmp_path, capsys)


def _memo(d, name, hits, pids=None):
    with open(os.path.join(d, f"{name}_per_passage.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["prompt_id", "risky_alone_recall"])
        w.writeheader()
        for i, p in enumerate(pids or [f"bookmia.00.{i:02d}" for i in range(100)]):
            w.writerow({"prompt_id": p, "risky_alone_recall": "0.3" if i < hits else "0.0"})


def test_the_host_check_passes_a_redraw_fails_a_shift_and_refuses_other_passages(tmp_path):
    """the declared host check of 2026-09-23, fired in every direction before host B has run"""
    for t in ("qwen25_7b", "kl3m17b"):
        _memo(tmp_path, f"selfix256_{t}", 78)
    assert v.host_check(str(tmp_path))[0] == "NOT RUN"
    _memo(tmp_path, v.HOST_CHECK, 70)                     # z = -1.29: a re-draw, not a host effect
    word, d = v.host_check(str(tmp_path))
    assert word == "PASS" and (d["local"], d["host_b"], d["same_side"]) == (78, 70, 92), d
    for hits in (60, 92):                                 # z = -2.75 and +2.77
        _memo(tmp_path, v.HOST_CHECK, hits)
        assert v.host_check(str(tmp_path))[0] == "FAIL", hits
    _memo(tmp_path, v.HOST_CHECK, 78, pids=[f"bookmia.01.{i:02d}" for i in range(100)])
    assert v.host_check(str(tmp_path))[0].startswith("FAIL: not the 100")
    _memo(tmp_path, "selfix256_qwen25_7b", 77)            # the local reference must be ONE draw
    with pytest.raises(AssertionError):
        v.host_check(str(tmp_path))


def test_s1_reads_the_short_licensed_rungs_and_leaves_v1_to_v4_alone(tmp_path, capsys):
    """feat-183, fired in every direction before any of its rungs has run"""
    _base(tmp_path)
    _complete(tmp_path)
    out = _run(tmp_path, capsys)
    assert "S1 NOT READ (incomplete)" in out and "(24 of 24 rungs missing)" in out
    for t, (_, _, role) in v.MODELS.items():
        if role == "openly licensed":
            for L in v.SHORT:
                _write(tmp_path, f"vetladder_L{L}_{t}", 0)
    out = _run(tmp_path, capsys)
    assert "S1 PASS HOLDS BELOW 100" in out and "V3 PASS HOLDS" in out
    _write(tmp_path, "vetladder_L35_pleias3b", 1)          # a short licensed leak
    out = _run(tmp_path, capsys)
    assert "S1 PASS BREAKS BELOW 100" in out and "('pleias3b', 35)" in out
    assert "V3 PASS HOLDS" in out and "V1 MONOTONE" in out, "a short licensed rung moved feat-180's reading"
    os.remove(os.path.join(tmp_path, "vetladder_L20_comma7b_per_passage.csv"))
    assert "S1 PASS BREAKS BELOW 100" in _run(tmp_path, capsys), "a leak breaks on any subset"
