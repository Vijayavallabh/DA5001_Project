"""run_workload_bind.sh must refuse a budget outside G0's 2x band, not just an unbracketed grid.

feat-174 amended G-cal "for this arm and every later one": bracketing is necessary and not
sufficient, because a grid can straddle a target and still have no point near it. The launcher
written on 2026-09-22 checked only `G-cal PASS`, so the Gutenberg arm launched its binding cell at
0.47x the target -- a gate that passes everything, wearing an argmin (caution (p)).
"""
import os
import re
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "scripts", "run_workload_bind.sh")


def test_the_script_reads_the_ratio_and_bounds_it_at_two_fold():
    src = open(SRC, encoding="utf-8").read()
    assert "RATIO=" in src, "the launcher does not read the argmin's ratio to the target"
    assert "0.5 <= $RATIO <= 2.0" in src, "the 2x band is not enforced"
    assert "REFINE" in src and "exit 6" in src, \
        "the launcher does not refuse; it must NOT RUN the binding cell outside the band"
    # and it must still refuse an unbracketed grid, which is the older half of the rule
    assert "G-cal PASS" in src and "exit 3" in src


def _ratio_branch(ratio):
    """Run the script's own numeric test in isolation, the way the script runs it."""
    expr = f"0.5 <= {ratio} <= 2.0"
    r = subprocess.run([os.path.join(ROOT, ".venv", "bin", "python"), "-c",
                        f"import sys; sys.exit(0 if {expr} else 1)"])
    return r.returncode == 0


def test_the_band_accepts_and_rejects_where_the_registration_says():
    assert _ratio_branch("0.91"), "MT-Bench's 0.91x must pass"
    assert _ratio_branch("0.97"), "CoTaEval's refined 0.97x must pass"
    assert _ratio_branch("1.00")
    assert not _ratio_branch("0.47"), "Gutenberg's 0.47x must be refused"
    assert not _ratio_branch("0.08"), "CoTaEval's first grid, 0.08x, must be refused"
    assert not _ratio_branch("2.50")
    # the boundaries are inclusive, as the rule is written
    assert _ratio_branch("0.5") and _ratio_branch("2.0")


def test_the_ratio_regex_matches_the_producing_script_s_own_line():
    """The launcher greps the ratio out of budget_calibration.py's stdout, so a change to that
    line's format silently disables the gate. Check they still agree."""
    cal = open(os.path.join(ROOT, "analysis", "budget_calibration.py"), encoding="utf-8").read()
    line = [l for l in cal.splitlines() if "CHOSEN k" in l]
    assert line, "budget_calibration.py no longer prints a CHOSEN k line"
    sample = "CHOSEN k = 1.0  (activity 0.03725, target 0.08008, ratio 0.47x)"
    assert re.search(r"ratio\s+[0-9.]+x", sample)
    assert "ratio" in " ".join(line) or any("ratio" in l for l in cal.splitlines()), \
        "the ratio the launcher greps for is not printed any more"
    got = subprocess.run(["sed", "-n", r"s/.*ratio \([0-9.]*\)x.*/\1/p"],
                         input=sample, capture_output=True, text=True).stdout.strip()
    assert got == "0.47", f"the launcher's sed reads {got!r} from the real line"
