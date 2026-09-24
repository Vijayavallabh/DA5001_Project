"""feat-178 in the manuscript: the opponent axis across workloads, and the AlpacaEval ladder.

Every claim is rebuilt from the CSV that measured it and checked inside its own paragraph, never
anywhere in the file (caution (an)). Each adjective is guarded by the shape it describes, so a
re-run that changed the shape fails here by name rather than leaving a sentence that no longer
reads off its numbers (caution (ai)).
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import body  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _csv(name):
    return list(csv.DictReader(open(os.path.join(ROOT, "results", name), encoding="utf-8")))


def _para():
    t = body("appendix_selection.tex")
    i = t.index(r"\textbf{The one thing every workload shares is the opponent")
    return t[i:t.index(r"\paragraph{Does it survive a different judge", i)]


def _ladder_sentence():
    t = body("appendix_selection.tex")
    i = t.index("Nor does the slope transfer as measured")
    return t[i:t.index("\\textbf{", i)]


def test_the_cross_workload_correlations_are_the_committed_ones():
    p1 = _csv("opponent_by_workload.csv")[0]
    t = _para()
    for col in ("rho", "rho_normalised", "rho_gmet", "rho_gsel"):
        v = float(p1[col])
        assert f"${v:+.3f}$" in t or f"${v:.3f}$" in t, (col, v)


def test_the_withdrawal_follows_the_normalised_band():
    """L4 was registered to withdraw the unification claim if the ordering did not survive the
    headroom; the sentence must withdraw exactly when the CSV reads outside CONSISTENT."""
    verdict = _csv("opponent_by_workload.csv")[0]["verdict_normalised"]
    assert ("we withdraw the claim" in _para()) == (verdict != "CONSISTENT"), verdict


def test_the_meter_tracks_strength_and_selection_does_not():
    p1 = _csv("opponent_by_workload.csv")[0]
    assert float(p1["rho_gmet"]) == 1.0 and "exact rank order" in _para()
    assert abs(float(p1["rho_gsel"])) < 0.7 and "selection's does not" in _para()


def test_every_rung_is_quoted_as_the_meter_winning_only_while_every_interval_says_so():
    rows = _csv("oppalp_ladder.csv")
    assert len(rows) == 5
    assert all(float(r["d3_hi95"]) < 0 for r in rows), "a rung's interval reaches zero"
    t = _para()
    assert "against every one the meter beats selection" in t
    s = sorted(float(r["strength"]) for r in rows)
    assert f"${s[0]:.4f}$ to ${s[-1]:.4f}$" in t
    d = sorted(float(r["d3"]) for r in rows)
    assert f"${d[-1]:.4f}$ to ${d[0]:.4f}$" in t


def test_the_generator_gap_is_measured_and_exceeds_the_ladder():
    alp = next(r for r in _csv("opponent_by_workload.csv") if r["workload"] == "AlpacaEval")
    rows = _csv("oppalp_ladder.csv")
    lla = next(r for r in rows if r["tag"] == "llama8b")
    committed, ladder = float(alp["opponent_strength"]), float(lla["strength"])
    span = max(float(r["strength"]) for r in rows) - min(float(r["strength"]) for r in rows)
    assert ladder - committed > span, "the text says the gap is more than the whole ladder"
    t = _para()
    assert f"${committed:.4f}$ through our sweep and ${ladder:.4f}$" in t
    assert int(alp["n_prompts"]) == 805 and "same $805$ prompts" in t


def test_the_weakest_rung_is_what_three_times_in_four_says():
    weakest = min(float(r["strength"]) for r in _csv("oppalp_ladder.csv"))
    assert 0.70 <= weakest < 0.80 and "three times in four" in _para(), weakest


def test_the_slope_sentence_reads_off_the_ladder():
    r = _csv("oppalp_ladder.csv")[0]
    rho, rho_n = float(r["rho"]), float(r["rho_normalised"])
    t = _ladder_sentence()
    assert rho > 0 and "\\emph{rises} with strength" in t and f"${rho:+.3f}$" in t
    assert rho_n < 0 and "point the same way" in t and f"${rho_n:.3f}$" in t
    assert "neither is a significance claim" in t
