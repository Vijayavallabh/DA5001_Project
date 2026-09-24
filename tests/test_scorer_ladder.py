"""feat-158/160: reward overoptimisation on the judge-free tasks is a 7B-scorer effect.

This is a result that FLATTERS the mechanism, which is exactly the shape caution (ag) says gets
quietly strengthened and caution (aq) says gets deleted in the other direction. So the guards pin
BOTH halves: nothing turns over above 7B, AND the 7B numbers are still printed, still conceded, and
still the paper's operating point. A rescope that becomes a withdrawal fails here by name.
"""
import csv
import os

from tests.manuscript import ROOT, body

CSV = os.path.join(ROOT, "results", "scorer_ladder.csv")
SEC = "appendix_selection.tex"


def rows():
    assert os.path.exists(CSV), f"{CSV} missing; this guard must not pass by never running"
    return list(csv.DictReader(open(CSV, encoding="utf-8")))


def by_arm():
    return {r["arm"]: r for r in rows()}


def _cotaeval_section():
    """The CoTaEval paragraph of Appendix D together with the scorer-ladder table it reports into.

    v10 (2026-09-24) merged the v9 paragraphs ("On the community-standard benchmark ...", "A larger
    scorer moves it ...", "A third rung settles it ...") into one paragraph and moved every band into
    Table tab:scorerladder, which follows it. Located by the table's label, so a retitle cannot
    retire the guards: from the \\paragraph that precedes the float to the float's end."""
    t = body(SEC)
    k = t.index("\\label{tab:scorerladder}")
    return t[t.rfind("\\paragraph{", 0, k):t.index("\\end{table}", k)]


def _at_csv_precision(x):
    """A CSV cell printed at the precision the CSV holds it (caution (j): round once, from it)."""
    return f"{float(x):+.{len(x.split('.')[1])}f}"


def test_every_arm_passed_every_gate():
    """G0 in particular: majority vote identical at all seven cells is what licenses comparing
    these passes at all, because it proves the text is byte-identical (caution (ap))."""
    a = by_arm()
    assert set(a) == {"tqa14", "gsm14", "cta72", "cta72_comma1t", "cta72_tc18b", "tqa72"}, set(a)
    for k, r in a.items():
        assert r["gate"] == "PASS", f"{k} did not pass its gates: {r['gate']}"


def test_nothing_turns_over_above_the_7b_scorer():
    """The finding. If a re-run reverses it, the rescope in the appendix and in Limitations is no
    longer supported and this fails by name rather than letting those sentences stand."""
    for k, r in by_arm().items():
        assert r["verdict"] != "TURNS OVER", (
            f"{k} turns over again at a larger scorer; the rescope is not supported")


def test_the_gsm8k_control_still_climbs():
    """Registered in advance: if the control stops climbing both 14B arms are INVALID, so a
    convenient TriviaQA result cannot be kept while its own control is discarded."""
    r = by_arm()["gsm14"]
    assert r["verdict"] == "CLIMBS", r["verdict"]
    assert float(r["half_widths"]) >= 2.0, f"the control is now marginal: {r['half_widths']}"
    assert float(r["spearman"]) > 0.9, r["spearman"]


def test_both_spearmans_inverted():
    """The sharpest form of the result: under a larger scorer accuracy rises with n where it fell."""
    for k in ("tqa14", "cta72"):
        r = by_arm()[k]
        assert float(r["spearman_committed"]) < 0 < float(r["spearman"]), (
            f"{k}: committed {r['spearman_committed']} -> {r['spearman']} is no longer an inversion")


def test_the_refuted_prediction_survives_a_page_trim():
    """We registered that CoTaEval's turn-over would survive 72B. It did not. A refuted
    prediction is a concession (caution (ag))."""
    log = open(os.path.join(ROOT, "results", "onset_prediction_scorer_ladder.md"),
               encoding="utf-8").read()
    flat = " ".join(log.split("\n## Scoring log", 1)[1].split())
    assert "REFUTED" in flat and "did not" in flat, "the refuted prediction is no longer recorded"


def test_the_rescope_is_in_the_paper_and_is_not_a_withdrawal():
    """Both halves, in one place each. Removing either alone is the failure mode."""
    para = _cotaeval_section()
    assert "72B" in para.replace("$", ""), "the 72B rung is gone from the appendix"
    a = by_arm()
    # the rescope
    assert "nothing turns over anywhere" in para.lower()
    for k in ("cta72", "cta72_comma1t", "cta72_tc18b", "tqa14", "tqa72"):
        v = _at_csv_precision(a[k]["gain"])
        assert f"${v}$" in para, f"{k}'s gain {v} is not printed with the 72B rung"
    # ... and that it is NOT a withdrawal
    assert "rescoped, not withdrawn" in para, (
        "the paragraph no longer says the 7B concession stands")
    # every printed interval end too, not only the point estimate: perturbing an interval end was
    # invisible to the first version of this guard. Checked as the printed PAIR, "[lo, hi]".
    for k in ("cta72", "cta72_comma1t", "cta72_tc18b", "tqa14", "tqa72"):
        lo, hi = _at_csv_precision(a[k]["lo95"]), _at_csv_precision(a[k]["hi95"])
        assert f"$[{lo}, {hi}]$" in para, f"{k}'s interval [{lo}, {hi}] is not printed with the 72B rung"


def test_the_7b_concession_is_still_stated_where_it_was_made():
    """The rescope must not swallow the concession. SCOPED to the CoTaEval paragraph, because
    `43\%` now occurs twice in this file and the first version of this guard was satisfied by the
    other occurrence -- caution (an), inside a guard written about caution (an)."""
    para = _cotaeval_section()
    assert "CoTaEval" in para and "token-F1" in para, "the CoTaEval paragraph is gone"
    assert "43\\%" in para, (
        "the 7B CoTaEval loss is no longer stated in the paragraph that reports it")
    assert "-0.1836" in para and "-0.2334" in para, (
        "the 7B headline loss or its disjoint re-draw is gone")


def test_no_cost_number_above_7b_is_claimed():
    """The honest limit: a bigger scorer is a requirement, and requirements cost. No latency arm
    was run above 7B, so the paper must not imply one was."""
    import re
    t = body(SEC) + body("appendix_limitations.tex")
    # v10 (2026-09-24): "no latency arm was run above 7B" reads "no latency arm ran above 7B"
    assert re.search(r"no latency arm (was run|ran) above \$7\$b", t.lower()), \
        "the paper no longer says the larger scorer's cost is unmeasured"


def test_limitations_states_the_scorer_requirement():
    t = body("appendix_limitations.tex")
    assert "scorer-size requirement" in t, (
        "Limitations no longer states the scorer-size requirement the registration fixed")
    assert "worse off at every $n$ than serving the anchor once" in t, (
        "the deployer-facing consequence was dropped")
