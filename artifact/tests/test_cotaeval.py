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


# --------------------------------------------------------------------------------------
# feat-157: is the turn-over the mechanism or the 7B scorer? The registered answer is
# SCORER-BOUND, which is a result AGAINST the prediction we filed, so every claim the
# appendix now makes about it is rebuilt here from the CSV rather than from the sentence.
# --------------------------------------------------------------------------------------
CSV14 = os.path.join(ROOT, "results", "cotaeval_scorer_scale.csv")
REGISTERED14 = "pointwise reward (Qwen2.5-14B)"


def scored14():
    assert os.path.exists(CSV14), f"{CSV14} missing; this guard must not pass by never running"
    rs = list(csv.DictReader(open(CSV14, encoding="utf-8")))
    s = [r for r in rs if r.get("rule") == REGISTERED14 and r.get("gate") == "PASS"]
    assert len(s) == 4, f"expected four scorer-scale arms, got {len(s)}"
    return s


def _para14():
    """The scorer-scale paragraph alone. Scoped because the words `turns over`, `marginal` and the
    anchor names all occur elsewhere in this file (caution (an))."""
    t = body(SEC)
    i = t.find("A larger scorer moves it")
    assert i > 0, "the scorer-scale paragraph is gone from the appendix"
    return t[i:i + 1400]


def test_the_scorer_scale_reading_is_the_one_the_data_give():
    """Two of four still turn over -> SCORER-BOUND by the table fixed before the run. If a re-run
    moves any verdict this fails by name instead of letting the paragraph stand."""
    v = [r["verdict"] for r in scored14()]
    assert v.count("TURNS OVER") == 2 and v.count("NO EFFECT") == 2, v
    assert "scorer-bound" in _para14().lower()


def test_the_two_that_survive_are_the_headline_anchor_and_its_redraw():
    by = {r["anchor"]: r for r in scored14()}
    for name in ("Comma-7B (2T)", "Comma-7B (2T), seed 5254"):
        assert by[name]["verdict"] == "TURNS OVER", f"{name} now reads {by[name]['verdict']}"
        assert float(by[name]["half_widths"]) >= 2.0, f"{name} is now marginal"
        # verdict and sign must agree: a stale label over a flipped number is exactly the defect
        # caution (ag) describes, and the appendix quotes only the verdict and the half-width.
        assert float(by[name]["gain"]) < 0 < -float(by[name]["hi95"]), (
            f"{name}: verdict says TURNS OVER but gain {by[name]['gain']} "
            f"/ hi95 {by[name]['hi95']} do not")
    p = _para14()
    for name in ("Comma-7B (2T)", "Comma-7B (1T)", "TinyComma-1.8B"):
        assert name.replace("-", "-") in p or name.split(" ")[0] in p


def test_the_rescue_is_not_a_climb():
    """The honest limit on a SCORER-BOUND reading: nothing climbed, so a bigger scorer bought the
    absence of harm and not a gain. Dropping that clause would read as a rescue."""
    assert [r for r in scored14() if r["verdict"] == "CLIMBS"] == []
    for r in scored14():
        if r["verdict"] == "NO EFFECT":
            assert r["marginal"] == "MARGINAL", f"{r['anchor']} is no longer marginal"
    p = _para14().lower()
    assert "began to climb" in p and "marginal" in p


def test_the_failed_prediction_survives_a_page_trim():
    """A registered prediction that was refuted is a concession (caution (ag))."""
    p = _para14().lower()
    assert "scorer-independent" in p and "wrong" in p


def test_the_instrument_check_claim_is_true_of_the_csv():
    """The appendix says each arm's n=1 F1 agrees with its counterpart's to four decimals. n=1 does
    not involve the scorer, so anything else would mean the pipeline changed."""
    ref = {"cta14_comma7b": "cta_news", "cta14_comma1t": "cta_comma1t",
           "cta14_tc18b": "cta_tc18b", "cta14_s5254": "cta_s5254"}
    for r in scored14():
        src = f"results/selection_verifiable_{ref[r['tag']]}.csv"
        rs = list(csv.DictReader(open(os.path.join(ROOT, src), encoding="utf-8")))
        base = [x for x in rs if x["arm"] == REGISTERED and int(x["n"]) == 1][0]
        assert round(float(base["acc"]), 4) == round(float(r["f1_n1"]), 4), (
            f"{r['anchor']}: n=1 moved {base['acc']} -> {r['f1_n1']}")
    assert "four decimals" in _para14()
