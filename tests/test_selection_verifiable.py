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
    # The two accuracy LEVELS stay in the prose (the figure plots gains, not levels); the two
    # bands moved into Figure 1's forest plot on 2026-09-17 and are checked wherever the paper
    # prints them. Every one still rounds from the CSV exactly once, caution (j).
    from tests.manuscript import carries_band
    assert f"${float(mv[1]['acc']):.3f}$" in body, mv[1]["acc"]
    best_n = max(mv, key=lambda n: float(mv[n]["acc"]))
    assert f"${float(mv[best_n]['acc']):.3f}$" in body, mv[best_n]["acc"]
    g = mv[best_n]
    assert carries_band(float(g["gain"]), float(g["gain_lo95"]), float(g["gain_hi95"]),
                        "experiments.tex"), g
    r64 = pw[64]
    assert carries_band(float(r64["gain"]), float(r64["gain_lo95"]), float(r64["gain_hi95"]),
                        "experiments.tex"), r64
    assert "selection_breadth_forest" in body, "the figure carrying these bands is not included"
    # and the paper must not present majority vote as the registered scorer
    assert "pointwise" in body or "reward" in body


def test_the_abstract_claims_the_judge_free_axis_only_because_it_was_measured():
    """The abstract's utility claim was 'judged' alone, which is the first thing a reviewer
    discounts. It may claim a judge-free axis only while an exact-match arm exists and lifts.

    The trigger matches on "no judge", not on one spelling of the sentence. It used to demand
    "no judge at all", and the 2026-09-17 abstract rewrite -- which says "needs no judge" -- turned
    this test into a no-op without failing anything. A conditional guard whose condition is a
    sentence someone will reword is a guard that retires itself, so a withdrawal now has to be
    deliberate: drop the claim from Section 3 as well, or edit this test."""
    from tests.manuscript import tex
    absr = " ".join(open(tex("iclr_2027.tex")).read().split())
    absr = absr.split("\\end{abstract}")[0]
    if "no judge" not in absr:
        body = open(tex("sections/experiments.tex"), encoding="utf-8").read()
        assert "Off the judge" not in body, \
            "the abstract dropped the judge-free claim while Section 3 still makes it"
        return                      # the claim was withdrawn everywhere; nothing to pin
    import csv as _csv
    rows = list(_csv.DictReader(open("results/selection_verifiable_comma7b.csv")))
    for rule in ("majority", "pointwise"):
        arm = [r for r in rows if r["arm"].startswith(rule) and r["gain_lo95"]]
        best = max(arm, key=lambda r: float(r["gain"]))
        assert float(best["gain_lo95"]) > 0, (rule, best)
    assert "judged" in absr.lower(), "the abstract must still say which of the two metrics is judged"


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
    from tests.manuscript import carries_band, _forest
    assert carries_band(float(top["gain"]), float(top["gain_lo95"]), float(top["gain_hi95"]),
                        "experiments.tex"), top
    assert carries_band(float(worst["gain"]), float(worst["gain_lo95"]), float(worst["gain_hi95"]),
                        "experiments.tex"), worst
    # the two bands carry their n, which used to be in the sentence and is now the row label
    labels = {lbl for lbl, *_ in _forest()}
    for r, rule in ((top, "majority vote"), (worst, "pointwise reward")):
        assert any(f"TriviaQA, {rule}, $n={r['n']}$" == l for l in labels), (rule, r["n"], labels)

    # the ratio between the two tasks, quoted as the support ceiling, is not eyeballed
    gsm = list(csv.DictReader(open("results/selection_verifiable_comma7b.csv")))
    gmv = [r for r in gsm if r["arm"].startswith("majority") and r["gain_lo95"]]
    best = max(gmv, key=lambda r: float(r["gain"]))
    ratio = float(best["gain"]) / float(top["gain"])
    # the judge-free detail now sits in the appendix figure caption (2026-09-19)
    both = body + " " + " ".join(open(tex("sections/appendix_selection.tex"),
                                      encoding="utf-8").read().split())
    assert f"${ratio:.1f}\\times$ less" in both, (ratio, best["gain"], top["gain"])


def test_limitations_carries_both_tasks_worth_of_scorer_evidence():
    """W5's committed consequence: the limitation keeps 'the scorer binds before the anchor does'
    and names the sign flip, not just the 3.4x."""
    body = " ".join(open(tex("sections/iclr_closing.tex")).read().split())
    # the heading was widened on 2026-09-14 when feat-106 landed: it now carries the scope
    # statement as well, so the phrase to pin is the claim, not the old heading text
    assert "the scorer binds first" in body
    assert "TriviaQA" in body and "3.4" in body


def test_every_cell_of_the_judgefree_table_rounds_from_its_csv():
    """Caution (j): a paper number rounds from the CSV once. 28 rows x 4 numbers plus the two
    baselines, checked mechanically rather than read."""
    import csv
    apx = " ".join(open(tex("sections/appendix_selection.tex"), encoding="utf-8").read().split())
    checked = 0
    for p in ("results/selection_verifiable_comma7b.csv",
              "results/selection_verifiable_tqa_comma7b.csv"):
        rows = list(csv.DictReader(open(p)))
        for r in rows:
            if r["arm"].startswith("risky"):
                assert f"${float(r['acc']):.3f}$" in apx, (p, r["arm"], r["acc"])
                checked += 1
                continue
            g, lo, hi = float(r["gain"]), float(r["gain_lo95"]), float(r["gain_hi95"])
            cell = (f"${float(r['budget_nats']):.3f}$ & ${float(r['acc']):.3f}$ & "
                    f"${g:+.3f}$ $[{lo:+.3f}, {hi:+.3f}]$")
            assert cell in apx or cell.split("& ", 1)[1] in apx, (p, r["arm"], r["n"], cell)
            checked += 1
        sp = {r["arm"].split(" ")[0]: r["spearman_acc_logn"] for r in rows if r["gain_lo95"]}
        for v in sp.values():
            assert f"${float(v):+.3f}$" in apx or f"${float(v):.3f}$" in apx, (p, v)
    assert checked == 32, checked


def test_the_appendix_states_both_metric_gates_as_measured():
    """Both arms are gated on an extraction rate and both gates are reported, not just the one
    that reads better. The TriviaQA figure is 1 - the logged no-answer fraction."""
    apx = " ".join(open(tex("sections/appendix_selection.tex"), encoding="utf-8").read().split())
    assert "$99.96\\%$" in apx and "$98.67\\%$" in apx, "a metric gate is missing from the appendix"


@pytest.mark.skipif(not os.path.exists("results/verifiable_metered_tqa.csv"),
                    reason="the judge-free head-to-head has not been scored yet")
def test_the_judgefree_headtohead_agrees_with_the_appendix():
    """feat-106. The arm the paper LOSES. H1 predicted METERED WINS before generation and it does,
    so what has to stay true is the reason: the winning arm is the one whose accuracy equals the
    unconstrained risky model's, at a certificate this paper calls vacuous."""
    import csv
    rows = list(csv.DictReader(open("results/verifiable_metered_tqa.csv")))
    met = {r["arm"]: r for r in rows if r["mechanism"] == "metered decoder"}
    sel = {r["arm"]: r for r in rows if r["mechanism"].startswith("selection")}
    best = max((r for a, r in met.items() if a != "k=-1"), key=lambda r: float(r["acc"]))
    assert best["arm"] == "k=20", best["arm"]
    # it wins by BECOMING the risky model; if that ever stops being true the framing is wrong
    assert float(best["acc"]) == float(met["k=-1"]["acc"]), (best["acc"], met["k=-1"]["acc"])
    # ... and where the certificate is small it buys nothing
    for arm in ("k=0.5", "k=1"):
        assert float(met[arm]["gain"]) <= 0.01, (arm, met[arm]["gain"])
    # H1 METERED WINS: the band is 0.03 with non-overlapping intervals
    top = max(sel.values(), key=lambda r: float(r["acc"]))
    assert float(best["acc"]) - float(top["acc"]) > 0.03
    assert float(best["acc_lo95"]) > float(top["acc_hi95"]), (best["acc_lo95"], top["acc_hi95"])

    apx = " ".join(open(tex("sections/appendix_selection.tex"), encoding="utf-8").read().split())
    # the table carries every metered arm and the three selection arms Section 6 and the
    # Limitations lean on; the intermediate n are in the nested grid of Table 6 already
    shown = [r for r in rows
             if r["mechanism"] == "metered decoder" or r["arm"] in ("n=8", "n=32", "n=64")]
    assert len(shown) == 9, [r["arm"] for r in shown]
    for r in shown:
        if r["arm"] == "k=-1":
            assert f"${float(r['acc']):.3f}$ $[{float(r['acc_lo95']):.3f}, " \
                   f"{float(r['acc_hi95']):.3f}]$" in apx, r["arm"]
            continue
        assert f"${float(r['acc']):.3f}$ $[{float(r['acc_lo95']):.3f}, " \
               f"{float(r['acc_hi95']):.3f}]$" in apx, (r["mechanism"], r["arm"], r["acc"])
    assert "$480.0$" in apx and "$44.8473$" in apx, "the winning arm's budget is not quoted"


@pytest.mark.skipif(not os.path.exists("results/verifiable_metered_tqa.csv"),
                    reason="the judge-free head-to-head has not been scored yet")
def test_h2_is_read_on_the_axis_the_preregistration_named():
    """H2 says "the same axes as Figure 1(b)", and those are Table 1's: selection's
    log n - (n-1)/n against the metered decoder's realised KL. The scorer used to write 0.0 for
    selection, which is true of a per-token meter and useless as a comparison axis."""
    import csv
    import math
    rows = list(csv.DictReader(open("results/verifiable_metered_tqa.csv")))
    sel = [r for r in rows if r["mechanism"].startswith("selection")]
    met = [r for r in rows if r["mechanism"] == "metered decoder" and r["arm"] != "k=-1"]
    assert all("kl_nats" in r for r in rows), "the registered axis is not in the CSV"
    for r in sel:
        n = int(r["arm"].split("=")[1])
        want = math.log(n) - (n - 1) / n if n > 1 else 0.0
        assert abs(float(r["kl_nats"]) - want) < 5e-4, (n, r["kl_nats"], want)
    # every accuracy selection BUYS is bought for under a tenth of the cheapest metered arm's spend
    ratios = []
    for s in sel:
        if float(s["gain"]) <= 0:
            continue                       # bought nothing; the ratio is against zero nats
        cand = [m for m in met if float(m["acc"]) >= float(s["acc"])]
        if not cand:
            continue
        c = min(cand, key=lambda m: float(m["realised_nats"]))
        ratios.append(float(s["kl_nats"]) / float(c["realised_nats"]))
    assert ratios and max(ratios) < 0.1, ratios       # FRONTIER HOLDS


def test_limitations_states_the_two_are_not_substitutes():
    """H3's committed consequence for METERED WINS + FRONTIER HOLDS: a scope statement in the MAIN
    text, carrying both halves -- the metered decoder wins, and wins by going vacuous."""
    body = " ".join(open(tex("sections/iclr_closing.tex"), encoding="utf-8").read().split())
    assert "not substitutes" in body
    assert "$0.618$" in body and "$0.190$" in body
    assert "$480$ nats" in body, "the scope statement omits the budget it was won at"


def test_the_scorer_scale_table_reads_by_column_from_its_four_csvs():
    """Thirty cells across four scorer CSVs plus the majority-vote row, checked by POSITION.

    Added 2026-09-18. audit_numbers.py already asks whether every literal is findable in some CSV --
    3,696 of them, one expected miss -- but membership is not placement: a right number in a wrong
    cell passes it, which is caution (j). This table was the largest in the appendix with no
    positional guard, and it carries the paper's scorer-saturation claim.

    The majority-vote row is the same numbers in all four runs by construction, which is what pins
    the cached-generation path; assert that too, because if the four ever disagree the table's own
    paragraph ("all four runs must reproduce it exactly --- they do") is false.
    """
    import csv as _csv
    import os as _os
    from tests.manuscript import body, tex as _tex
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    files = {"$0.5$B": "selection_verifiable_comma7b_qwen05b.csv",
             "$1.5$B": "selection_verifiable_comma7b_qwen15b.csv",
             "$3$B": "selection_verifiable_comma7b_qwen3b.csv",
             "$7.6$B": "selection_verifiable_comma7b.csv"}
    grid = [2, 4, 8, 16, 32, 64]
    txt = body("appendix_selection.tex")

    def _rows(s):
        """Table rows with their leading rule commands stripped -- a row can begin '\\midrule $0.5$B'
        once whitespace is normalised, which a bare startswith() never matches."""
        import re as _re
        for line in s.split("\\\\"):
            yield _re.sub(r"^(?:\\(?:top|mid|bottom)rule|\\cmidrule\{[^}]*\}|\s)+", "", line)

    majorities = {}
    for label, fn in files.items():
        rows = list(_csv.DictReader(open(_os.path.join(root, "results", fn), encoding="utf-8")))
        rew = {int(float(r["n"])): r for r in rows if "reward" in r["arm"].lower()}
        maj = {int(float(r["n"])): r for r in rows if "major" in r["arm"].lower()}
        majorities[label] = tuple(round(float(maj[n]["acc"]), 3) for n in grid)
        row = next((l for l in _rows(txt) if l.startswith(label + " &")), None)
        assert row, (label, "the scorer row is gone from the table")
        cols = [c.strip() for c in row.split("&")]
        assert len(cols) == len(grid) + 1, (label, cols)
        for j, n in enumerate(grid, start=1):
            want = f"{float(rew[n]['acc']):.3f}"
            got = cols[j].replace("\\mathbf{", "").replace("}", "").strip()
            assert got == f"${want}$", (label, f"n={n} column", got, want)

    assert len(set(majorities.values())) == 1, \
        ("the four runs disagree on majority vote, so the paragraph's 'they do' is false",
         majorities)
    maj_row = next((l for l in _rows(txt) if l.startswith("majority vote &")), None)
    assert maj_row, "the majority-vote row is gone"
    cols = [c.strip() for c in maj_row.split("&")]
    for j, v in enumerate(next(iter(majorities.values())), start=1):
        got = cols[j].replace("\\mathbf{", "").replace("}", "").strip()
        assert got == f"${v:.3f}$", (f"majority column {j}", got, v)
