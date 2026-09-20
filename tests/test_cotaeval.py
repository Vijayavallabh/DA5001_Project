"""CoTaEval: the benchmark the Program Chairs named, and the negative result it produced.

feat-155/156. Every number is rebuilt from results/cotaeval_scoring.csv, and the CONCESSION -- that
selection anchoring under a pointwise reward loses utility as n grows on a standard benchmark -- is
guarded per site, because caution (ag) says a length edit deletes concessions first and this one
cost two body lines to seat.
"""
import csv
import os

from tests.manuscript import ROOT, body, tex

CSV = os.path.join(ROOT, "results", "cotaeval_scoring.csv")
REGISTERED = "pointwise reward (Qwen2.5-7B)"
SEC = "appendix_selection.tex"


def rows():
    assert os.path.exists(CSV), f"{CSV} missing; this guard must not pass by never running"
    return list(csv.DictReader(open(CSV, encoding="utf-8")))


def scored():
    return [r for r in rows() if r.get("rule") == REGISTERED and r.get("gate") == "PASS"]


def test_every_usable_anchor_turns_over_under_the_registered_selector():
    """The finding itself. If a future re-run changes it, this fails by name rather than letting
    the paper keep a sentence the data no longer support."""
    s = scored()
    assert len(s) == 4, f"expected the four anchors that passed G1, got {len(s)}"
    for r in s:
        assert r["verdict"] == "TURNS OVER", f"{r['anchor']} now reads {r['verdict']}"
        assert float(r["gain"]) < 0


def test_the_ladder_claim_is_recorded_as_refuted():
    """The band asked for at least two anchors climbing. Zero do."""
    climbs = [r for r in scored() if r["verdict"] == "CLIMBS"]
    assert len(climbs) == 0, f"anchors now climb: {[r['anchor'] for r in climbs]}"


def test_the_headline_number_and_its_replication_are_in_the_appendix():
    t = body(SEC)
    by = {r["anchor"]: r for r in scored()}
    head = by["Comma-7B (2T)"]
    rep = by["Comma-7B (2T), seed 5254"]
    for r in (head, rep):
        assert f"{float(r['gain']):+.4f}" in t, (
            f"the appendix does not print {r['anchor']}'s gain {float(r['gain']):+.4f}")
    assert float(rep["gain"]) < 0 and rep["verdict"] == "TURNS OVER", (
        "the disjoint re-draw no longer reproduces the turn-over")


def test_the_main_text_carries_the_concession():
    """Registered consequence of TURNS OVER: reported in the MAIN text, not only the appendix.
    It cost two body lines to seat, which is exactly the kind of sentence a page trim removes."""
    t = body("iclr_closing.tex")
    assert "CoTaEval" in t, "the main text no longer mentions CoTaEval at all"
    i = t.find("CoTaEval")
    assert "loses" in t[max(0, i - 120):i + 120], (
        "the main text mentions CoTaEval without saying the mechanism loses utility there")


def test_the_four_unusable_anchors_are_not_counted_as_evidence():
    """G1 failures say the anchor cannot read a news article, which is the capability floor the
    paper already concedes -- not evidence about selection anchoring. The scorer must refuse to
    compute their bands, and the appendix must say why."""
    failed = [r for r in rows() if r.get("gate") == "FAIL"]
    assert len(failed) == 4, f"expected four G1 failures, got {len(failed)}"
    for r in failed:
        assert "gain" not in r or not r.get("gain"), (
            f"a band was computed for {r['anchor']}, which failed G1")
    t = body(SEC)
    assert "cannot read a news article" in t, (
        "the appendix no longer explains why four anchors were excluded")


def test_the_scope_limits_survive():
    """Two things the result does NOT touch, both stated so the scope is not read wider than the
    measurement. Removing either would let a reader take this as a refutation of the certificate."""
    t = body(SEC)
    i = t.find("community-standard benchmark the scorer does worse than bind")
    assert i > 0, "the CoTaEval paragraph is gone"
    w = t[i:i + 3000]
    assert "does not depend on the served text being good" in w, (
        "the appendix no longer says the certificate is untouched by this result")
    assert "0.546" in w, "the appendix no longer says GSM8K majority vote is untouched"
