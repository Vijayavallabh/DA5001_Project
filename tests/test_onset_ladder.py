"""Plan v5: the ladder joiner must match pair labels across two files and score both hypotheses.

The two source CSVs spell the pairs differently ("memorised" vs "mem."), so a silent label mismatch
would drop pairs and make whichever hypothesis had fewer points look better.
"""
import csv
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _run(tmp, onset_rows, theory_rows):
    o, t, out = f"{tmp}/onset.csv", f"{tmp}/theory.csv", f"{tmp}/out"
    _write(o, onset_rows)
    _write(t, theory_rows)
    r = subprocess.run([sys.executable, os.path.join(REPO, "analysis", "onset_ladder.py"),
                        "--onset", o, "--theory", t, "--out", out],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    return r.stdout, list(csv.DictReader(open(f"{out}/onset_ladder.csv")))


def test_label_spellings_are_matched(tmp_path):
    tmp = str(tmp_path)
    onset = [dict(pair="Comma-7B + memorised Comma-7B", mode="single", L=0,
                  s_x_nats_per_token=2.39, onset_est=2.13)]
    theory = [dict(pair="Comma-7B + mem. Comma-7B", s_safe_median=2.39, s_risky_median=0.18,
                   req_q25=2.13, temperature=1.0)]
    _, rows = _run(tmp, onset, theory)
    assert len(rows) == 1, "the 'memorised' / 'mem.' spellings must be matched, not dropped"


def test_ladder_trend_is_reported_for_a_shared_anchor(tmp_path):
    tmp = str(tmp_path)
    onset, theory = [], []
    for i, (sr, meas) in enumerate([(0.10, 3.10), (0.40, 2.80), (0.70, 2.50)]):
        name = f"rung{i} + mem. rung{i}"
        onset.append(dict(pair=name, mode="single", L=0, s_x_nats_per_token=3.20, onset_est=meas))
        theory.append(dict(pair=name, s_safe_median=3.20, s_risky_median=sr,
                           req_q25=3.20 - sr, temperature=1.0))
    out, rows = _run(tmp, onset, theory)
    assert "anchor held fixed" in out and "3 rungs" in out
    # the measured onset falls as s_r rises, which is what the derivation requires
    assert "onset changes by -0.60 nats" in out
    assert "H_const requires  0.00" in out
    # and the derivation must beat the constant on this synthetic ladder
    assert "derivation" in out.split("mean |error|")[1].split("\n")[0]


def test_only_single_mode_rows_are_used(tmp_path):
    tmp = str(tmp_path)
    onset = [dict(pair="A + mem. A", mode="oracle", L=50, s_x_nats_per_token=3.0, onset_est=1.0),
             dict(pair="A + mem. A", mode="single", L=0, s_x_nats_per_token=3.0, onset_est=2.7)]
    theory = [dict(pair="A + mem. A", s_safe_median=3.0, s_risky_median=0.3, req_q25=2.7,
                   temperature=1.0)]
    _, rows = _run(tmp, onset, theory)
    assert len(rows) == 1 and abs(float(rows[0]["measured_onset"]) - 2.7) < 1e-9
