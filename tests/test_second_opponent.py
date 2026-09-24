"""The second-opponent arm: the one robustness check in this paper that did not survive.

Bands: results/onset_prediction_second_opponent.md, committed before any generation. The paired
difference of gains against Qwen2.5-14B-Instruct is -0.0065 [-0.0385, +0.0255] -- the interval
contains zero, so the registered reading is UNRESOLVED and the registered consequence is that the
abstract's judged claim gains "against one fixed opponent".

Every check here is conditioned on the CSV, never on a phrase in another section. Caution (aq):
a withdrawal branch triggered by wording retires itself on the first reword, and a skipped branch
is a pass.
"""
import csv
import os

from tests.manuscript import ROOT, body, tex

CSV = os.path.join(ROOT, "results", "order_averaged_h2h__opp2.csv")
LOG = os.path.join(ROOT, "results", "onset_prediction_second_opponent.md")
SEC = "appendix_selection.tex"


def rows():
    assert os.path.exists(CSV), f"{CSV} missing; this guard must not pass by never running"
    return list(csv.DictReader(open(CSV, encoding="utf-8")))


def d(q):
    r = [x for x in rows() if x["quantity"].startswith(q)]
    assert len(r) == 1, f"expected one {q!r} row, got {len(r)}"
    return float(r[0]["value"]), float(r[0]["lo95"]), float(r[0]["hi95"])


def window():
    """The opponent paragraph alone, so a number in it cannot be guarded by a coincidence
    elsewhere in a long appendix. v10 (2026-09-24) folded v9's "Changing the opponent" paragraph
    into "Stronger opponents", whose Table~\\ref{tab:opponentladder} carries this arm as its top
    rung; the window is that paragraph through the end of its table."""
    t = body(SEC)
    i = t.find("\\paragraph{Stronger opponents.}")
    assert i > 0, "the opponent paragraph is gone from the appendix"
    j = t.find("\\end{table}", t.find("\\label{tab:opponentladder}", i))
    assert j > i, "the opponent ladder table is gone from the paragraph"
    return t[i:j]


def abstract():
    return " ".join(open(tex("iclr_2027.tex"), encoding="utf-8").read().split())


def test_the_difference_against_the_second_opponent_is_unresolved_and_the_paper_says_so():
    v, lo, hi = d("D3")
    assert lo <= 0 <= hi, (
        "the D3 interval no longer contains zero; the registered reading is no longer UNRESOLVED "
        "and every consequence in this file has to be re-derived from the bands")
    # Bare values: the interval is printed as one math group, `$[-0.0385, +0.0255]$`, so there is
    # no `$` around either end. Caution (an) learned the same thing on `$\\log 8 = 2.08$`.
    #
    # SCOPED TO ITS OWN PARAGRAPH. The first version searched the whole appendix and so could not
    # see the sign of D3 flipped: `-0.0065` is also the lower end of the compute-matched loss
    # `$-0.0395$ $[-0.0720, -0.0065]$` a few pages earlier, and that unrelated occurrence satisfied
    # the check. Caution (an), and the number it hid is the one this arm is about.
    t = window()
    # scoped further, to this opponent's own row of the ladder table, which also carries the reading
    # (the table prints family names without "-Instruct", which its caption states once)
    t = t[t.index("Qwen2.5-14B}", t.index("\\label{tab:opponentladder}")):]
    row = t[:t.index("\\\\")]
    for x in (v, lo, hi):
        assert f"{x:+.4f}" in row, f"the opponent's row does not print {x:+.4f} from its CSV"
    assert "unresolved" in row, "the row does not give the registered reading, UNRESOLVED"


def test_both_arms_still_beat_their_controls_and_that_is_reported_too():
    """Reporting only the failure would be as one-sided as reporting only the success.

    v10 (2026-09-24) no longer prints this opponent's two gains; it states the success once, for the
    whole ladder ("Both mechanisms beat their own controls against every opponent"). A claim about
    a set is checked against the set (caution (ai)): every rung's D1 and D2 must exclude zero."""
    import sys
    sys.path.insert(0, ROOT)
    from analysis.opponent_strength import ARMS
    for q in ("D1", "D2"):
        v, lo, hi = d(q)
        assert lo > 0, f"{q} no longer excludes zero; the appendix sentence about it is now false"
    for opp, suffix, _, _ in ARMS:
        f = os.path.join(ROOT, "results", f"order_averaged_h2h{suffix}.csv")
        assert os.path.exists(f), f"{f} missing; this guard must not pass by never running"
        for q in ("D1", "D2"):
            r = [x for x in csv.DictReader(open(f, encoding="utf-8")) if x["quantity"].startswith(q)]
            assert len(r) == 1 and float(r[0]["lo95"]) > 0, (
                f"{opp}: {q} no longer excludes zero; 'against every opponent' is now false")
    assert "beat their own controls against every opponent" in window(), (
        "the opponent paragraph no longer reports that both arms beat their controls")


def test_the_abstract_carries_the_opponent_qualifier_because_the_interval_contains_zero():
    """Conditioned on the DATA, not on a phrase. If a future pass resolves the difference the
    assertion inverts by itself and says so, rather than being satisfied by a reword."""
    _, lo, hi = d("D3")
    a = abstract()
    if lo <= 0 <= hi:
        assert "against one fixed opponent" in a, (
            "the judged head-to-head is unresolved against a second opponent and the abstract "
            "does not say the claim is against one fixed opponent")
    else:
        assert "against one fixed opponent" not in a, (
            "the difference now resolves against a second opponent; the qualifier is no longer "
            "the registered consequence and should be revisited deliberately")


def test_no_section_claims_the_judged_comparison_is_opponent_independent():
    every = body(SEC, "experiments.tex", "iclr_closing.tex", "selection.tex", "iclr_intro.tex")
    assert "not across opponents" in every or "opponent-dependent" in every, (
        "no section carries the opponent limit, and one of them must")
    for bad in ("across opponents and judges alike", "whatever the opponent",
                "independent of the opponent"):
        assert bad not in every, f"a section claims opponent-independence: {bad!r}"


def test_the_paper_reports_the_registered_reading_and_not_the_scripts_own_label():
    """analysis/order_averaged_h2h.py labels ANY negative point estimate 'REVERSAL REFUTED' without
    consulting the interval, while requiring a positive one to clear its interval before it says
    CONFIRMED. That asymmetric string is the script's, not the registration's. Caution (ag): a
    label a script writes into a CSV is not a measurement."""
    log = open(LOG, encoding="utf-8").read()
    head, sep, tail = log.partition("\n## Scoring log")
    assert sep, "no scoring log"
    assert "UNRESOLVED" in head, "the three-way taxonomy must be committed before the run"
    assert "UNRESOLVED" in tail, "the scored reading must be the registered one"
    assert "REVERSAL REFUTED" in tail, (
        "the scoring log must disclose the CSV's own softer-sounding label rather than quietly "
        "reporting only the registered one")
    assert d("D3")[0] < 0, "the point estimate is negative and the log must not imply otherwise"


def test_the_arm_did_not_touch_what_it_registered_it_would_not():
    """No certificate, leakage or s(x) number comes from this arm."""
    log = open(LOG, encoding="utf-8").read()
    _, _, tail = log.partition("\n## Scoring log")
    assert "certificate" in tail and "judge-free axis is untouched" in tail.replace("---", "--"), (
        "the scoring log must state what the arm leaves standing, not only what it overturns")


def test_the_intros_gain_ratio_is_qualified_by_the_opponent_because_it_is_a_gain_ratio():
    """The headline `$2.6\\times$` is g_sel/g_met on the registered pass. Against this opponent the
    same ratio is 0.89 -- below one. The certificate comparison beside it ($3.175$ against $171.3$
    nats) is NOT opponent-dependent and is deliberately not qualified; only the ratio is.

    Conditioned on the CSV, like every other check here: if a future pass resolves the difference
    the branch inverts and says to revisit the wording, rather than being satisfied by a reword.
    """
    _, lo, hi = d("D3")
    t = body("iclr_intro.tex")
    i = t.find(r"win by $2.6\times$")
    if i < 0:
        # Re-derived 2026-09-24: the intro dropped the ratio (measured on echo-carrying text, and a
        # ratio whose denominator's interval nears zero) and now states the paired difference. What
        # this guard protects -- an unqualified head-to-head -- is then the difference, and it must
        # carry the serving/opponent qualifier that Table 1 and the ladder measured.
        # v10 (2026-09-24) states the condition and the reversal rather than "depends on how the
        # risky model is served": better "whenever the risky model continues text", while "the meter
        # wins only where the judge rewards instruction following". Both halves must stay together.
        flat = " ".join(t.split())
        # v11 (2026-09-24): "best-of-$64$ is judged better than that model continuing text and worse
        # than it served as a chat assistant or following instructions". Both halves together.
        j = flat.find("judged better than that model continuing text")
        assert j >= 0, "the intro no longer states the head-to-head"
        w = flat[j: j + 300]
        assert "chat assistant" in w and "following instructions" in w, \
            "the intro states the head-to-head without its serving-configuration qualifier"
        return
    clause = t[i:i + 120]
    if lo <= 0 <= hi:
        assert "opponent" in clause, (
            "the gain ratio is not reproduced against a second opponent and the intro states it "
            "unqualified")
    else:
        assert "opponent" not in clause, (
            "the difference now resolves against a second opponent; the qualifier should be "
            "revisited deliberately rather than left standing")


def test_the_certificate_comparison_is_not_watered_down_by_the_opponent_result():
    """The nats are a property of the served law. An over-correction that hedged them would be as
    wrong as the missing hedge on the ratio was."""
    import re as _re
    # v10 (2026-09-24): the introduction compares certificates ($2.08$ against $2000$); the
    # bound-against-measured-spend comparison ($3.175$ against $171.3$ nats) moved to Section 4.2
    t = body("experiments.tex")
    assert "$171.3$" in t and "$3.175$" in t, "the certificate comparison has gone"
    # EVERY occurrence. The v9 intro printed $171.3$ three times and the first was Figure 1's
    # caption, so a guard on t.find() was inspecting a different sentence than the one it meant and
    # a deliberate hedge inserted at the second went undetected. Caution (an), same shape as the
    # -0.0065 collision above.
    hits = [m.start() for m in _re.finditer(_re.escape("$171.3$"), t)]
    # >= 1: Section 4.2 quotes the measured spend twice (the k=10 cell and the gains comparison), and
    # the check below runs on every occurrence there is.
    assert len(hits) >= 1, f"expected the measured spend in Section 4, found {len(hits)}"
    for i in hits:
        assert "opponent" not in t[max(0, i - 160):i], (
            "the measured-nats comparison has been hedged by opponent language near "
            f"char {i}; the spend is a property of the served law and depends on no opponent")
