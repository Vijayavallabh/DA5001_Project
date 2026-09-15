"""feat-119: the scorer-free instance, checked against the CSV and its three limits.

This one is post hoc -- a join over already-measured quantities with no committed band -- so the
checks that matter are that it says so, that the numbers round from the files they came from, and
that the three limits travel with them wherever they are quoted.
"""
import csv
import os

import pytest

NOTE = "results/scorer_free_cost_note.md"
CSV = "results/scorer_free_cost.csv"

pytestmark = pytest.mark.skipif(not os.path.exists(CSV), reason="feat-119 has not run")


def _rows(path):
    return list(csv.DictReader(open(path)))


def _tex(name):
    from tests.manuscript import tex
    return " ".join(open(tex(name), encoding="utf-8").read().split())


def test_the_post_hoc_join_is_labelled_as_one_and_cannot_inflate_the_count():
    """Same rule feat-115 followed: a re-analysis with no committed band is not an
    onset_prediction_*.md and must keep saying so in its first lines."""
    import glob
    head = open(NOTE, encoding="utf-8").read()[:1600]
    assert "no committed bands" in head
    assert not glob.glob("results/onset_prediction_*scorer_free*")


def test_the_cost_column_is_the_serving_cost_model_with_no_scorer_term():
    """Majority vote's cost must be n * P_anchor exactly -- if a scorer term creeps back in, the
    whole point of the row is gone."""
    from analysis.serving_cost import P_ANCHOR, P_RISKY, P_SCORER
    met = P_ANCHOR + P_RISKY
    for r in _rows(CSV):
        pf = 0.0 if r["rule"] == "majority vote" else P_SCORER
        want = int(r["n"]) * (P_ANCHOR + pf) / met
        assert abs(float(r["cost_vs_metered"]) - want) < 5e-3, r


def test_the_accuracies_are_feat_118s_and_were_not_recomputed():
    ref = {(x["task"], x["scorer"], int(x["n"])): x["acc"]
           for x in _rows("results/verifiable_scorer_scale.csv")}
    for r in _rows(CSV):
        for task, col in (("gsm8k", "gsm8k_acc"), ("triviaqa", "triviaqa_acc")):
            assert abs(float(r[col]) - float(ref[(task, r["rule"], int(r["n"]))])) < 5e-5, (r, task)


def test_every_scorer_free_cell_beats_every_reward_cell_on_both_tasks():
    """The claim the appendix makes. If a future re-run breaks it, the sentence must change."""
    rows = _rows(CSV)
    mv = [r for r in rows if r["rule"] == "majority vote" and int(r["n"]) >= 4]
    rw = [r for r in rows if r["rule"] != "majority vote"]
    for col in ("gsm8k_acc", "triviaqa_acc"):
        assert min(float(r[col]) for r in mv) >= max(float(r[col]) for r in rw), col


def test_the_three_limits_travel_with_the_numbers():
    """Post hoc, needs a canonical answer, and the ratio is the cost model's denominator rather
    than a measured head-to-head. All three in the note, in the CSV, and in the appendix."""
    note = open(NOTE, encoding="utf-8").read()
    apx = _tex("sections/appendix_selection.tex")
    for phrase in ("canonical answer", "No metered decoder was run on GSM8K"):
        assert phrase in note, phrase
    assert all("post hoc" in r["limits"] for r in _rows(CSV))
    assert "post hoc" in apx and "canonical answer" in apx
    assert "no metered decoder was run on gsm8k" in apx.lower()


def test_the_appendix_does_not_claim_to_beat_the_unconstrained_model():
    """Majority vote reaches 0.546 against the risky model's 0.786 greedy. The comparison is among
    mechanisms that carry a certificate, and the appendix has to say so."""
    apx = _tex("sections/appendix_selection.tex")
    assert "$0.786$" in apx and "not a claim to have beaten" in apx
    ref = {r["arm"]: r["acc"] for r in _rows("results/selection_verifiable_comma7b.csv")}
    greedy = [v for k, v in ref.items() if "greedy" in k][0]
    assert f"${float(greedy):.3f}$" in apx, greedy


def test_the_body_points_at_the_appendix_range():
    body = _tex("sections/selection.tex")
    assert "$5.75\\times$" in body and "app:scorerfree" in open(
        __import__("tests.manuscript", fromlist=["tex"]).tex("sections/selection.tex"),
        encoding="utf-8").read()
