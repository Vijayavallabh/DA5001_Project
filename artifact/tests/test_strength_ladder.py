"""The corpus-vs-memoriser arm, and the three things that would make it answer the wrong question.

(1) Strength must be MEASURED, not read off the epoch count -- the knob is not assumed monotone.
(2) The ladder must share one anchor, one corpus and one grid, or the ratios are not comparable.
(3) The bands in the scorer must be the bands in the pre-registration, to the digit (caution (j)).
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREREG = os.path.join(ROOT, "results/onset_prediction_strength.md")


def _head():
    return open(PREREG, encoding="utf-8").read().partition("\n## Scoring log")[0]


def test_the_scorer_correlates_against_measured_strength_not_the_epoch_count():
    src = open(os.path.join(ROOT, "analysis/strength_ladder.py"), encoding="utf-8").read()
    body = src.partition("def main(")[2]
    assert "strengths = [r[\"k_minus1_recall\"] for r in ok]" in body, \
        "the ladder must correlate against the measured k=-1 arm"
    # the epoch count appears only as a LABEL; it must never be an input to rho
    assert "epochs" not in body.partition("rho, p = ")[0].rpartition("strengths =")[2]


def test_the_committed_bands_are_the_scorer_bands():
    from analysis.strength_ladder import (ENTRY_GATE, RHO_EXPLAINS, SPAN_EXPLAINS,
                                          SPAN_REFUTES, STRENGTH_SPAN_MIN)
    head = _head()
    # the prereg writes these to two decimals; a bare repr of 0.10 is "0.1" and would never match,
    # which is the same trap as quoting a CSV's float straight into prose (caution (j))
    assert f"`rho <= {RHO_EXPLAINS:g}`" in head, RHO_EXPLAINS
    assert f"span `>= {SPAN_EXPLAINS:.2f}`" in head, SPAN_EXPLAINS
    assert f"span `< {SPAN_REFUTES:.2f}`" in head, SPAN_REFUTES
    assert f"less than `{STRENGTH_SPAN_MIN:g}x`" in head, STRENGTH_SPAN_MIN
    assert f"at least `{ENTRY_GATE:.2f}`" in head, ENTRY_GATE


def test_the_ladder_shares_one_anchor_one_corpus_and_one_grid():
    sh = open(os.path.join(ROOT, "scripts/run_strength_ladder.sh"), encoding="utf-8").read()
    # every flag except --epochs is fixed in the script, so the ladder cannot vary two things
    code = "\n".join(l for l in sh.splitlines() if not l.lstrip().startswith("#"))
    assert code.count("--epochs") == 1 and '--epochs "$ep"' in code
    for fixed in ("--rank 128", "--lr 3e-4", "--stop-loss 0.02", "--target-modules all-linear"):
        assert fixed in sh, fixed
    # one grid for all points, and it must include feat-120's licensed extension
    grid = re.search(r'^GRID="([^"]+)"', sh, re.M).group(1).split()
    assert grid[:2] == ["-1", "0"], "both mandatory baselines"
    for k in ("4.6", "5.3", "6.6"):
        assert k in grid, f"the licensed extension point {k} is missing"
    assert sh.count("CUDA_VISIBLE_DEVICES=2") == 1 and "CUDA_DEVICE_ORDER=PCI_BUS_ID" in sh


def test_the_forty_epoch_point_is_feat_120s_own_merged_run():
    from analysis.strength_ladder import POINTS, S_X
    labels = [l for l, _ in POINTS]
    assert len(POINTS) == 4 and any("feat-120" in l for l in labels)
    d = dict((l, r) for l, r in POINTS)
    assert d["epochs=40 (feat-120)"] == "output/phase5/fineb_pleias12b_full", \
        "the 40-epoch point must use the MERGED grid, or it sits at a ceiling the others do not"
    # the anchor never changes, so s(x) is one number and the ratios share a denominator
    import csv
    for r in csv.DictReader(open(os.path.join(ROOT, "results/onset_theory_bookmia.csv"))):
        if r["pair"].startswith("Pleias-1.2B"):
            assert abs(float(r["s_safe_median"]) - S_X) < 1e-9, (r["s_safe_median"], S_X)
            return
    pytest.fail("Pleias row missing from results/onset_theory_bookmia.csv")


def test_no_outcome_is_allowed_to_restore_the_retracted_sentence():
    """The one thing a 'strength explains it' result would be spun into. Committed against."""
    head = _head()
    assert "No outcome of this arm restores the retracted sentence" in head
    src = open(os.path.join(ROOT, "analysis/strength_ladder.py"), encoding="utf-8").read()
    assert "does NOT restore" in src and "inherits the confound" in src
