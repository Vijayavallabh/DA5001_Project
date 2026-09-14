"""feat-101: the judge-free axis.

The parts worth pinning are the ones that can silently produce a wrong accuracy: answer
extraction from a base model that runs on into the next question, the majority-vote tie rule
(which must return an INDEX into the drawn samples, or Proposition 4 does not apply to what is
served), and the paired bootstrap.
"""
import math
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_verifiable import (  # noqa: E402
    N_GRID, boot, boot_gain, extract, gold, majority, spearman)
from tests.manuscript import tex  # noqa: E402


def test_gold_strips_the_marker_and_the_thousands_comma():
    assert gold("blah\n#### 1,234") == "1234"
    assert gold("#### 18") == "18"


@pytest.mark.parametrize("text,want", [
    ("Janet makes 16-3-4=9 eggs. 9*2=18.\n#### 18", "18"),
    ("... #### 18\n\nQuestion: something else\nAnswer: 99\n#### 99", "18"),
    ("The total is 42 dollars.", "42"),
    ("She sells them for $1,250 total.", "1250"),
    ("#### -7", "-7"),
    ("no numbers at all", None),
    ("", None),
])
def test_extract_handles_run_on_and_formatting(text, want):
    assert extract(text) == want


def test_extract_cuts_at_the_run_on_before_looking_for_the_marker():
    """A base model continues into the next few-shot question. Scoring the LAST '####' in the
    buffer would grade a different problem's answer and inflate or deflate accuracy silently."""
    t = "2+2=4.\n#### 4\n\nQuestion: unrelated\nAnswer: 1+1=2\n#### 2"
    assert extract(t) == "4"


def test_majority_returns_an_index_into_the_drawn_samples():
    """Proposition 4 bounds what is SERVED, and it is served only if it is one of the n draws."""
    ans = ["7", "3", "7", None]
    i = majority(ans)
    assert 0 <= i < len(ans) and ans[i] == "7"


def test_majority_breaks_ties_to_the_earliest_sample():
    assert majority(["5", "9", "5", "9"]) == 0
    assert majority(["9", "5", "9", "5"]) == 0


def test_majority_ignores_unextractable_samples_but_still_serves_one():
    assert majority([None, "4", None, "4", "6"]) == 1
    assert majority([None, None]) == 0  # nothing extractable: still an index, still bounded


def test_boot_gain_is_paired_and_zero_against_itself():
    c = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    g, lo, hi = boot_gain(c, c, 200, 1)
    assert (g, lo, hi) == (0.0, 0.0, 0.0)


def test_boot_gain_recovers_a_uniform_improvement_exactly():
    base = [0.0] * 20
    arm = [1.0] * 20
    g, lo, hi = boot_gain(arm, base, 200, 1)
    assert g == 1.0 and lo == 1.0 and hi == 1.0


def test_boot_interval_brackets_the_mean():
    c = [1.0] * 30 + [0.0] * 70
    acc, lo, hi = boot(c, 2000, 5)
    assert acc == pytest.approx(0.30)
    assert lo < acc < hi and 0.15 < lo and hi < 0.45


def test_spearman_is_one_on_a_monotone_series_and_minus_one_reversed():
    xs = [math.log(n) for n in N_GRID]
    assert spearman(xs, [0.1 * i for i in range(len(xs))]) == pytest.approx(1.0)
    assert spearman(xs, [-0.1 * i for i in range(len(xs))]) == pytest.approx(-1.0)


def test_the_grid_is_nested_powers_of_two_so_no_sample_is_scored_twice():
    assert N_GRID == [2 ** i for i in range(7)]


def test_the_preregistration_fixes_the_gate_before_the_bands():
    p = "results/onset_prediction_verifiable.md"
    txt = open(p).read()
    # line 3 mentions `## Scoring log` in backticks, so split on the HEADING, not the phrase
    head, _, log = txt.partition("\n## Scoring log")
    assert log is not None
    # the unfavourable half of the gate is on the record, not just the favourable half
    assert "**PASS**" in head and "FAIL" in head
    assert "no judge-free head-to-head against" in " ".join(head.split())
    for band in ("SC LIFTS", "SC FLAT", "REWARD LIFTS", "REWARD FLAT"):
        assert band in head, band
    # all four outcomes carry a fixed manuscript consequence
    assert "Both fail." in head and "the reward model is the weak link" in head


def test_the_preregistration_names_selfconsistency_as_covered_by_the_proposition():
    txt = " ".join(open("results/onset_prediction_verifiable.md").read().split())
    assert "self-consistency" in txt
    assert "returns one of the `n`" in txt or "returns one of the" in txt


@pytest.mark.skipif(not os.path.exists("results/selection_verifiable_comma7b.csv"),
                    reason="the judge-free arm has not been scored yet")
def test_the_scored_arm_agrees_with_the_manuscript():
    import csv
    rows = list(csv.DictReader(open("results/selection_verifiable_comma7b.csv")))
    assert rows, "empty result file"
    ns = {r["arm"]: [int(r["n"]) for r in rows if r["arm"] == r["arm"]] for r in rows}
    assert ns
    for r in rows:
        if r["gain"] and float(r["n"]) == 1:
            assert float(r["gain"]) == 0.0, "the n=1 control must be its own baseline"
    body = " ".join(open(tex("sections/experiments.tex")).read().split())
    assert "GSM8K" in body, "the judge-free arm is scored but Section 6 does not report it"
    mv = {int(r["n"]): r for r in rows if r["arm"].startswith("majority")}
    pw = {int(r["n"]): r for r in rows if r["arm"].startswith("pointwise")}
    # the three numbers Section 6 quotes, each rounded from the CSV once (caution (j))
    assert f"${float(mv[1]['acc']):.3f}$" in body, mv[1]["acc"]
    best_n = max(mv, key=lambda n: float(mv[n]["acc"]))
    assert f"${float(mv[best_n]['acc']):.3f}$" in body, mv[best_n]["acc"]
    g = mv[best_n]
    assert f"$+{float(g['gain']):.3f}$ $[+{float(g['gain_lo95']):.3f}, " \
           f"+{float(g['gain_hi95']):.3f}]$" in body, g
    r64 = pw[64]
    assert f"$+{float(r64['gain']):.3f}$" in body, r64["gain"]
    # and the paper must not present majority vote as the registered scorer
    assert "pointwise" in body or "reward" in body


def test_the_abstract_claims_the_judge_free_axis_only_because_it_was_measured():
    """The abstract's utility claim was 'judged' alone, which is the first thing a reviewer
    discounts. It may say 'with no judge at all' only while an exact-match arm exists and lifts."""
    from tests.manuscript import tex
    absr = " ".join(open(tex("iclr_2027.tex")).read().split())
    absr = absr.split("\\end{abstract}")[0]
    if "no judge at all" not in absr:
        return                      # the claim was withdrawn; nothing to pin
    import csv as _csv
    rows = list(_csv.DictReader(open("results/selection_verifiable_comma7b.csv")))
    for rule in ("majority", "pointwise"):
        arm = [r for r in rows if r["arm"].startswith(rule) and r["gain_lo95"]]
        best = max(arm, key=lambda r: float(r["gain"]))
        assert float(best["gain_lo95"]) > 0, (rule, best)
    assert "judged" in absr, "the abstract must still say which of the two metrics is judged"


@pytest.mark.skipif(not os.path.exists("results/selection_verifiable_tqa_comma7b.csv"),
                    reason="the TriviaQA arm has not been scored yet")
def test_the_knowledge_task_arm_agrees_with_the_manuscript():
    """feat-105. The TriviaQA arm is the one that cuts against the paper: majority vote lifts, the
    paper's own pointwise reward does not, and at n=16 it is significantly WORSE than the anchor's
    first draw. All three numbers are quoted in Section 6 and each must round from the CSV once."""
    import csv
    rows = list(csv.DictReader(open("results/selection_verifiable_tqa_comma7b.csv")))
    mv = {int(r["n"]): r for r in rows if r["arm"].startswith("majority")}
    pw = {int(r["n"]): r for r in rows if r["arm"].startswith("pointwise")}
    assert set(mv) == set(pw) == set(N_GRID), sorted(mv)

    # W1 SC LIFTS and W2 FLAT are the readings Section 6 is written against; if either flips, the
    # paragraph is false and this fails rather than the reader finding it.
    top = mv[max(N_GRID)]
    assert float(top["gain_lo95"]) > 0, ("W1 no longer lifts", top)
    assert float(pw[max(N_GRID)]["gain_lo95"]) <= 0, ("W2 now lifts; Section 6 says it does not",
                                                      pw[max(N_GRID)])
    worst = min(pw.values(), key=lambda r: float(r["gain"]))
    assert float(worst["gain_hi95"]) < 0, ("the reward is no longer significantly negative "
                                           "anywhere; Section 6's sign-flip sentence is now false")

    body = " ".join(open(tex("sections/experiments.tex")).read().split())
    assert "TriviaQA" in body, "the knowledge-task arm is scored but Section 6 does not report it"
    assert f"$+{float(top['gain']):.3f}$ $[+{float(top['gain_lo95']):.3f}, " \
           f"+{float(top['gain_hi95']):.3f}]$ at $n={top['n']}$" in body, top
    assert f"${float(worst['gain']):.3f}$ $[{float(worst['gain_lo95']):.3f}, " \
           f"{float(worst['gain_hi95']):.3f}]$ at $n={worst['n']}$" in body, worst

    # the ratio between the two tasks, quoted as the support ceiling, is not eyeballed
    gsm = list(csv.DictReader(open("results/selection_verifiable_comma7b.csv")))
    gmv = [r for r in gsm if r["arm"].startswith("majority") and r["gain_lo95"]]
    best = max(gmv, key=lambda r: float(r["gain"]))
    ratio = float(best["gain"]) / float(top["gain"])
    assert f"${ratio:.1f}\\times$ less" in body, (ratio, best["gain"], top["gain"])


def test_limitations_carries_both_tasks_worth_of_scorer_evidence():
    """W5's committed consequence: the limitation keeps 'the scorer binds before the anchor does'
    and names the sign flip, not just the 3.4x."""
    body = " ".join(open(tex("sections/iclr_closing.tex")).read().split())
    assert "The scorer binds before the anchor does" in body
    assert "TriviaQA" in body and "3.4" in body
