"""The matched-context intervention is scored by one ratio of two spreads, so what a test can
protect is the window that decides which arm is 'matched', the band edges, and the fact that a pair
with no matched arm is dropped from both sides rather than silently compared against itself."""
import csv
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.context_intervention import DOMINANT, MATCHED, REFUTED, spread  # noqa: E402

ROWS = ["pair,label,seed_tokens,seed_chars,seed_words,s_x,ratio,ratio_nv",
        "A,A seed 20,20,80.0,14.0,3.0,0.90,0.90",
        "B,B seed 20,20,43.0,7.5,2.2,1.20,1.20",
        "B,B seed 40 (matched),40,86.0,14.8,2.2,1.00,1.00",
        "C,C seed 20,20,50.0,9.5,3.3,1.05,1.05"]


def test_the_committed_constants_are_the_ones_in_the_preregistration():
    assert (MATCHED, DOMINANT, REFUTED) == ((13.6, 15.0), 0.5, 0.8)
    md = open("results/onset_prediction_matched_context.md").read()
    for s in ("0.5\\,S_{20}", "0.8\\,S_{20}", "13.6", "15.0"):
        assert s in md, s


def test_a_pair_with_no_matched_arm_is_dropped_from_both_sides(tmp_path):
    """C is at 9.5 words and has no matched arm. Keeping it on the seed-20 side only would inflate
    S_20 against an S_match it never contributed to -- the one way this ratio can be flattered."""
    src = tmp_path / "rows.csv"
    src.write_text("\n".join(ROWS) + "\n")
    out = subprocess.run([sys.executable, "analysis/context_intervention.py",
                          "--rows", str(src), "--out", str(tmp_path)],
                         capture_output=True, text=True, check=True).stdout
    assert "no matched arm yet for: C" in out
    rows = list(csv.DictReader(open(tmp_path / "context_intervention.csv")))
    assert [r["pair"] for r in rows] == ["A", "B", "ALL"]
    # S_20 over {A, B} is 1.20 - 0.90 = 0.30; S_match is 1.00 - 0.90 = 0.10.
    assert float(rows[-1]["ratio_matched"]) == 0.10 and float(rows[-1]["ratio_20"]) == 0.30
    assert float(rows[-1]["delta"]) == round(0.10 / 0.30, 4)
    assert "DOMINANT" in out


def test_spread_is_the_range_not_a_standard_deviation():
    assert spread([0.9, 1.2, 1.0]) == 0.30000000000000004 or abs(spread([0.9, 1.2, 1.0]) - 0.3) < 1e-9


def test_the_manuscript_quotes_the_csv():
    """0.289, 0.113 and 61% appear in three places in the paper and one figure title. They are the
    ALL row of results/context_intervention.csv; if the CSV moves and the prose does not, this is
    where it shows."""
    import re
    from tests.manuscript import tex
    rows = list(csv.DictReader(open("results/context_intervention.csv")))
    tot = rows[-1]
    assert tot["pair"] == "ALL"
    s20, smatch, frac = float(tot["ratio_20"]), float(tot["ratio_matched"]), float(tot["delta"])
    body = "".join(open(tex(f"sections/{f}.tex"), encoding="utf-8").read()
                   for f in ("onset", "appendix_seed", "iclr_closing", "appendix_limitations"))
    assert f"${s20:.3f}$ to ${smatch:.3f}$" in body, (s20, smatch)
    closed = round(100 * (1 - frac))
    assert body.count(f"${closed}\\%$") >= 3, closed
    assert f"$\\times {frac:.3f}$" in body or f"= {frac:.3f}$" in body, frac


def test_the_two_new_arms_are_inside_their_committed_band():
    """[0.85, 0.96] was committed before either was swept. The 1B lands at 0.959, on the edge --
    if a later re-run moves it out, the pre-registration says that is a refutation, not a nudge."""
    rows = {r["pair"]: r for r in csv.DictReader(open("results/context_intervention.csv"))}
    for p in ("open-calm-1b", "open-calm-3b"):
        assert 0.85 <= float(rows[p]["ratio_matched"]) <= 0.96, (p, rows[p]["ratio_matched"])
        assert float(rows[p]["delta"]) < 0, p          # every mover moved down
