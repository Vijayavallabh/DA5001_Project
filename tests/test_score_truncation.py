"""feat-060: the truncation scorer decides between two readings committed before the run, so the
constants it scores against must match the pre-registration, and the metric choice must not be
the one that truncation inflates."""
import os, re, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from analysis.score_truncation import READINGS, REFERENCE, per_passage, point_and_ci


def test_readings_match_the_committed_pre_registration():
    md = open(os.path.join(REPO, "results", "onset_prediction_trunc276.md")).read()
    assert "2.07" in md and "2.46" in md          # the two onsets named there
    assert set(READINGS) == {"decode-step count", "tokenizer itself"}
    # ratios in the file are quoted as ~0.87 and ~1.03 against s(x) = 2.3830
    assert abs(READINGS["decode-step count"] - 0.87) < 1e-9
    assert abs(READINGS["tokenizer itself"] - 1.03) < 1e-9


def test_the_primary_metric_has_no_denominator_to_inflate():
    """nv_recall divides by reference words, so truncation roughly doubles it. If the primary
    ever became nv_recall the experiment would confirm one branch by construction."""
    src = open(os.path.join(REPO, "analysis", "score_truncation.py")).read()
    m = re.search(r'\("lcs_word", "lcs_word", a\.lcs_thresh, True\)', src)
    assert m, "lcs_word must be the primary metric"
    assert re.search(r'\("nv_recall", "nv_recall", a\.nv_thresh, False\)', src)


def test_reference_bands_do_not_overlap():
    """The two bands are what the verdict is read against; if they overlapped, no measurement
    could distinguish the readings."""
    lo4, hi4 = REFERENCE["four-character pairs"]
    lok, hik = REFERENCE["KL3M pairs, untruncated"]
    assert hi4 < lok


def test_point_and_ci_on_a_known_curve(tmp_path):
    p = tmp_path / "composition.csv"
    lines = ["k,mode,prompt_id,lcs_word,nv_recall"]
    for pid in range(20):
        lines += [f"1.0,single,p{pid},0.0,0.0", f"2.0,single,p{pid},8.0,0.02"]
    p.write_text("\n".join(lines) + "\n")
    point, lo, hi, nocross, curve = point_and_ci(per_passage(str(p), "lcs_word"), 4.0, 2.0)
    assert abs(point - 1.5) < 1e-9        # halfway across the bracket
    assert nocross == 0.0 and lo <= point <= hi
