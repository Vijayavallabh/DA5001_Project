"""The TriviaQA vacuity arm: every fraction in the paper is recomputed from the CSV.

The paragraph this guards used to say two things that could not both be true -- `72` nats
"already vacuous", `12` and `24` "the budgets whose certificate is not vacuous", and then a closing
sentence saying no vacuity number came from the arm at all. Nothing in results/ measured S(x) for a
short factual answer, so all three were adjectives about a set of numbers that did not exist
(caution (ai), and the contradiction inside one file is caution (ao)).

results/onset_prediction_tqa_vacuity.md carries the bands. Two of four were REFUTED.
"""
import csv
import math
import os
import re

import pytest

from tests.manuscript import ROOT, body

SEC = "appendix_selection.tex"
CSV = os.path.join(ROOT, "results", "tqa_vacuity.csv")
METERED = [(0.5, 12.0), (1.0, 24.0), (3.0, 72.0), (20.0, 480.0)]


def sx():
    assert os.path.exists(CSV), f"{CSV} missing; this guard must not pass by never running"
    v = [float(r["s_anchor_nats"]) for r in csv.DictReader(open(CSV, encoding="utf-8"))]
    assert len(v) == 500, f"expected the 500 served questions, got {len(v)}"
    return v


def frac(v, K):
    return sum(1 for s in v if s <= K) / len(v)


def pct(text, value):
    """Does the paper print this percentage, to one decimal or as an integer?

    The integer spelling is accepted ONLY when the value IS an integer percentage. The first
    version allowed it always, so a mutation of `$89.6\\%$` fell through to `$90\\%$` -- which this
    appendix prints elsewhere about something unrelated -- and the guard passed on a number that
    had been deleted. Caution (an): a guard satisfied by a different occurrence of its phrase is
    not guarding its sentence.
    """
    p = value * 100
    if f"${p:.1f}\\%$" in text:
        return True
    return abs(p - round(p)) < 1e-9 and f"${round(p)}\\%$" in text


def test_every_metered_vacuous_fraction_in_the_paper_is_the_one_in_the_csv():
    v, t = sx(), body(SEC)
    for k, K in METERED:
        assert pct(t, frac(v, K)), (
            f"k={k} (K={K:g} nats) is vacuous on {frac(v, K):.3f} and the paper does not print it")


def test_the_selection_concession_is_printed_at_both_ends_of_its_grid():
    """Selection is vacuous on a third of these questions at n=64. That argues against this paper,
    which by caution (aq) is exactly the kind of sentence a length edit deletes first."""
    v, t = sx(), body(SEC)
    for n in (4, 64):
        f = frac(v, math.log(n))
        assert pct(t, f), f"selection at n={n} is vacuous on {f:.3f} and the paper does not say so"


def test_the_median_surprisal_is_the_csv_median():
    v, t = sx(), body(SEC)
    v = sorted(v)
    med = 0.5 * (v[249] + v[250])
    assert f"${med:.2f}$" in t, f"median S(x) is {med:.2f}"
    assert f"${v[50]:.2f}$" in t and f"${v[450]:.2f}$" in t, "p10/p90 must match the CSV"


def test_the_ordering_claim_matches_the_numbers_and_not_just_the_wording():
    """The paper says selection's WORST budget is vacuous less often than the meter's BEST. That is
    a claim about a set of numbers, which is the thing nothing checks (caution (ai)). Rebuild both
    series and check the shape, so that if it ever stops holding the guard fails by name rather
    than the sentence quietly surviving."""
    v = sx()
    sel_worst = max(frac(v, math.log(n)) for n in (4, 8, 16, 32, 64))
    met_best = min(frac(v, K) for _, K in METERED)
    assert sel_worst < met_best, (
        f"selection's worst {sel_worst:.3f} is not below the meter's best {met_best:.3f}; "
        "the sentence in appendix_selection.tex has to be rewritten")
    t = body(SEC)
    # The COMPARISON, not the two words. The first version asked only whether "worst" and "best"
    # both appeared, so rewriting "is vacuous less often than" to "is about the same as" left it
    # passing on a sentence that no longer made the claim -- caution (ai), the claim about a set of
    # numbers being the thing nothing checks.
    assert re.search(r"\\emph\{worst\}[^.]{0,120}vacuous less often[^.]{0,80}\\emph\{best\}", t), (
        "the ordering sentence no longer says selection's worst budget is vacuous LESS OFTEN "
        "than the meter's best, but its numbers are still here")


def test_no_metered_budget_on_this_grid_is_called_non_vacuous():
    """The withdrawn phrasings. Every budget here is vacuous on at least 89.6% of the questions,
    so a sentence naming some of them as the non-vacuous ones is false."""
    v, t = sx(), body(SEC)
    assert min(frac(v, K) for _, K in METERED) > 0.5, (
        "a metered budget is now non-vacuous on most questions; re-read the withdrawn phrasings "
        "before trusting this guard")
    for dead in ("budgets whose certificate is not vacuous", "already vacuous"):
        assert dead not in t, f"withdrawn phrasing is back: {dead!r}"
    # The BODY carried the same shape and nothing had caught it: iclr_closing.tex said "at every
    # non-vacuous budget it is the anchor", which quantifies over an empty set and so implies
    # non-vacuous budgets exist. A guard scoped to the appendix would have missed it -- caution
    # (af), a test written for a repair must scan every section file, not the one the arm was about.
    every = body("iclr_closing.tex", "experiments.tex", "selection.tex", "iclr_intro.tex")
    assert "non-vacuous budget" not in every, (
        "a section names a non-vacuous metered budget on the judge-free grid; there are none")


def test_the_arm_reports_its_two_instrument_gates():
    """G2 (the better model must be less surprised) and G3 (the prompt must matter) are what
    separate this measurement from a pipeline artefact. Caution (au) is three arms lost to a scorer
    nobody checked against real output."""
    log = open(os.path.join(ROOT, "results", "onset_prediction_tqa_vacuity.md"),
               encoding="utf-8").read()
    head, sep, tail = log.partition("\n## Scoring log")
    assert sep, "no scoring log"
    for g in ("G1", "G2", "G3"):
        assert g in head and g in tail, f"{g} must be registered before the run and scored after it"
    rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8"))]
    med = lambda c: sorted(float(r[c]) for r in rows)[250]  # noqa: E731
    assert med("s_risky_nats") < med("s_anchor_nats"), "G2 fails on the committed CSV"
    assert med("s_perm_nats") - med("s_anchor_nats") >= 5.0, "G3 fails on the committed CSV"


def test_the_fractions_are_lower_bounds_and_the_paper_says_so():
    """S is scored as -log max_alias P, which overstates S(x) and so understates vacuity. The
    direction was fixed in the pre-registration before the run; it must reach the reader."""
    t = body(SEC)
    # Scoped to ITS OWN sentence: this appendix says "lower bound" elsewhere, about contamination
    # recall, and the first version of this guard was satisfied by that occurrence (caution (an)).
    assert re.search(r"[Bb]oth figures are lower bounds", t), (
        "the conservative direction is not disclosed beside the two vacuity tables")
