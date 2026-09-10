"""The matched-strength reanalysis must actually restrict to the matched passages, and its
elasticity must reduce to the definition on a case where the answer is known by hand."""
import csv, math, os, subprocess, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.matched_strength import onset_on, sweep, s_by_pid


def _write(path, rows):
    cols = ["k", "mode", "prompt_id", "nv_recall", "lcs_word"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)


def test_sweep_separates_the_baseline_from_the_budgeted_grid(tmp_path):
    p = tmp_path / "c.csv"
    _write(p, [dict(k=-1.0, mode="single", prompt_id="a", nv_recall=0.9, lcs_word=100),
               dict(k=2.0, mode="single", prompt_id="a", nv_recall=0.0, lcs_word=3),
               dict(k=3.0, mode="single", prompt_id="a", nv_recall=0.1, lcs_word=5),
               dict(k=2.0, mode="oracle", prompt_id="a", nv_recall=0.5, lcs_word=40)])
    curve, base = sweep(str(p))
    assert curve == {"a": {2.0: 3.0, 3.0: 5.0}}, "oracle rows and k=-1 must stay out of the curve"
    assert base == {"a": 0.9}


def test_onset_interpolates_inside_its_bracket_on_the_given_subset(tmp_path):
    p = tmp_path / "c.csv"
    rows = []
    for pid, lo, hi in (("keep", 3.0, 5.0), ("drop", 0.0, 0.0)):
        rows += [dict(k=2.0, mode="single", prompt_id=pid, nv_recall=0, lcs_word=lo),
                 dict(k=3.0, mode="single", prompt_id=pid, nv_recall=0, lcs_word=hi)]
    _write(p, rows)
    curve, _ = sweep(str(p))
    # on {keep} alone the mean crosses 4 halfway between 2 and 3; with "drop" it never crosses
    assert abs(onset_on(curve, ["keep"], 4.0) - 2.5) < 1e-9
    assert onset_on(curve, ["keep", "drop"], 4.0) is None


def test_the_committed_run_matches_the_definition_of_elasticity():
    path = "results/matched_strength.csv"
    if not os.path.exists(path):
        return  # the analysis has not been run in this checkout
    for r in csv.DictReader(open(path)):
        e = (math.log(float(r["onset_warped"]) / float(r["onset_control"]))
             / math.log(float(r["s_x_warped"]) / float(r["s_x_control"])))
        assert abs(e - float(r["elasticity"])) < 5e-3, r["arm"]
        assert int(r["n_matched"]) <= int(r["n_total"])
        # matching must close the strength gap it was built to close
        assert (float(r["base_recall_warped"]) - float(r["base_recall_control"])
                < float(r["base_recall_warped_all"]) - float(r["base_recall_control_all"]))
