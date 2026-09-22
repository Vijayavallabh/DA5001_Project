"""feat-180's gates and verdicts, mutation-tested on synthetic rungs before any rung has run.

results/onset_prediction_vetting_ladder.md registers G0 (every rung is the same 50 passages), G1 (the
70B's L = 100 re-run reproduces the arm on record, else the on-record L = 100 rungs are not mixed in)
and four readings. Each must fire on the case it is for.
"""
import csv
import os
import sys

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


def test_a_monotone_ladder_that_reproduces_reads_cleanly(tmp_path, capsys):
    _base(tmp_path)
    for L, k in ((20, 0), (50, 8), (100, 25), (200, 30)):
        _write(tmp_path, f"vetladder_L{L}_llama70b", k, col="risky_alone_recall")
    for L, k in ((20, 0), (200, 9)):
        _write(tmp_path, f"vetladder_L{L}_olmo2_13b", k)
    _write(tmp_path, "vetladder_L200_comma7b", 0)
    out = _run(tmp_path, capsys)
    assert "G0 PASS" in out and "G1 PASS" in out
    assert "V1 MONOTONE" in out and "V2 L* = 50" in out and "V3 PASS HOLDS" in out
    assert "olmo2_13b  L=100  leaks on  6/50" in out and "[on record]" in out


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
    _write(tmp_path, "vetladder_L100_llama70b", 25, col="risky_alone_recall")
    _write(tmp_path, "vetladder_L150_olmo2_13b", 4)
    assert "V1 MONOTONE" in _run(tmp_path, capsys)
