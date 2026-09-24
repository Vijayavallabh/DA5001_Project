"""Mutation-test feat-135's scorer BEFORE the arm produces a number (caution (v)).

feat-132 is why this file exists: its gate was written against a sibling arm at a different n and
would have failed a perfectly good replication, and the defect was only found by mutation-testing the
gate against the arm on record before the run. So every gate and every verdict branch of
analysis/score_kl3m37b_breadth64.py is exercised here on synthetic CSVs, with the real column names,
and each mutation must move the verdict in the direction the pre-registration says it should.

The point is not that the arithmetic works. It is that when the real data lands, the reading is
already fixed and cannot be tuned to the answer.
"""
import csv
import os
import subprocess
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORER = os.path.join(ROOT, "analysis", "score_kl3m37b_breadth64.py")
PY = os.path.join(ROOT, ".venv", "bin", "python")
JUDGES = ("Phi-3.5-mini-instruct", "Meta-Llama-3.1-8B-Instruct")
GRID = (1, 2, 4, 8, 16, 32, 64)


def write_arm(out, delta, spread=0.02, n_prompts=500, mean_words=95.0, grid=GRID):
    """A synthetic pass whose paired g(64)-g(8) is `delta` with a controllable spread."""
    import random as _r
    rng = _r.Random(7)
    per = os.path.join(out, "selection_scaling_per_prompt_kl3m37b64.csv")
    with open(per, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["judge", "prompt_id"] + [f"u_n{n}" for n in grid])
        for j in JUDGES:
            for i in range(n_prompts):
                u8 = 0.40
                u64 = u8 + delta + rng.gauss(0, spread)
                w.writerow([j, f"p{i}"] + [0.30 if n == 1 else (u8 if n <= 8 else u64) for n in grid])
    summ = os.path.join(out, "selection_scaling_kl3m37b64.csv")
    with open(summ, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["judge", "selector", "n", "kl_nats", "n_prompts", "u", "u_lo95", "u_hi95",
                    "gain", "gain_lo95", "gain_hi95", "mean_words", "spearman_u_logn"])
        for j in JUDGES:
            for n in grid:
                gain = 0.0 if n == 1 else (0.10 if n <= 8 else 0.10 + delta)
                w.writerow([j, "reward", n, 0.0 if n == 1 else 1.2, n_prompts, 0.3 + gain,
                            0.28, 0.32, gain, gain - 0.02, gain + 0.02, mean_words, 0.95])
    return summ, per


def run(out):
    r = subprocess.run([PY, SCORER, "--out", out], capture_output=True, text=True, cwd=ROOT)
    return r.returncode, r.stdout + r.stderr


@pytest.mark.skipif(not os.path.exists(PY), reason="venv not present")
def test_a_clear_climb_reads_climbs_and_is_not_marginal():
    with tempfile.TemporaryDirectory() as out:
        write_arm(out, delta=+0.09)                       # the effect size on record
        rc, o = run(out)
        assert "VERDICT: CLIMBS" in o, o[-1500:]
        assert "MARGINAL" not in o.split("VERDICT")[1], o[-1500:]
        assert rc == 0


@pytest.mark.skipif(not os.path.exists(PY), reason="venv not present")
def test_no_effect_reads_saturated_by_8():
    with tempfile.TemporaryDirectory() as out:
        write_arm(out, delta=0.0)
        rc, o = run(out)
        assert "VERDICT: SATURATED BY 8" in o, o[-1500:]


@pytest.mark.skipif(not os.path.exists(PY), reason="venv not present")
def test_a_negative_effect_reads_turns_over():
    """The reading that would falsify Appendix I's scoped no-overoptimisation sentence."""
    with tempfile.TemporaryDirectory() as out:
        write_arm(out, delta=-0.09)
        rc, o = run(out)
        assert "VERDICT: TURNS OVER" in o, o[-1500:]


@pytest.mark.skipif(not os.path.exists(PY), reason="venv not present")
def test_a_positive_but_marginal_effect_is_flagged_marginal():
    """The rule feat-131 paid for: at under 2.0 half-widths the verdict is not promoted. A wide
    spread at a small delta puts the interval just off zero, which is exactly the KL3M-1.7B shape."""
    with tempfile.TemporaryDirectory() as out:
        write_arm(out, delta=+0.012, spread=0.13)
        rc, o = run(out)
        # Assert the MACHINE-READABLE field, not a sentence: a guard written as an exact phrasing
        # retires itself on the first reword (cautions (an), (ar)). The prose is checked
        # case-insensitively and only for the consequence the rule exists to state.
        assert '"marginal": "yes"' in o, o[-2000:]
        low = o.lower()
        assert "marginal" in low and "seed" in low and "replication" in low, o[-2000:]
        assert "not promoted" in low, o[-2000:]


@pytest.mark.skipif(not os.path.exists(PY), reason="venv not present")
def test_a_partial_sweep_is_refused_by_g2_and_the_band_is_not_computed():
    with tempfile.TemporaryDirectory() as out:
        write_arm(out, delta=+0.09, grid=(1, 2, 4, 8, 16, 32))       # never reached 64
        rc, o = run(out)
        assert "G2 coverage: FAIL" in o and "NOT SCORED" in o, o[-1200:]
        assert "paired g(64) - g(8)" not in o, "the band was computed behind a failed gate"
        assert rc == 1


@pytest.mark.skipif(not os.path.exists(PY), reason="venv not present")
def test_the_wrong_prompt_count_is_refused_by_g2():
    with tempfile.TemporaryDirectory() as out:
        write_arm(out, delta=+0.09, n_prompts=200)
        rc, o = run(out)
        assert "G2 coverage: FAIL" in o and "200 prompts" in o, o[-1200:]
        assert "paired g(64) - g(8)" not in o


@pytest.mark.skipif(not os.path.exists(PY), reason="venv not present")
def test_degenerate_completions_are_refused_by_g4():
    """An anchor emitting almost nothing makes every judged comparison a statement about length."""
    with tempfile.TemporaryDirectory() as out:
        write_arm(out, delta=+0.09, mean_words=8.0)
        rc, o = run(out)
        assert "G4 length:   FAIL" in o, o[-1200:]
        assert "paired g(64) - g(8)" not in o, "the band was computed behind a failed gate"
        assert rc == 1


@pytest.mark.skipif(not os.path.exists(PY), reason="venv not present")
def test_the_empty_fraction_never_blocks_the_band():
    """G3 is REPORTED, not gated -- feat-132 failed by gating a rate against a sibling arm. With no
    generation directory present the empty fraction cannot be computed, and the band must still be
    read rather than the arm refused."""
    with tempfile.TemporaryDirectory() as out:
        write_arm(out, delta=+0.09)
        rc, o = run(out)
        assert "G3 empties:  REPORTED (never gated)" in o, o[-1200:]
        assert "VERDICT: CLIMBS" in o, "a missing empty fraction blocked the band"
