"""The paper's conceded weaknesses, pinned to the data that conceded them.

Caution (ag): a page-budget trim deletes concessions first, because they are the sentences that
argue against the paper and so read as cuttable. Caution (aq): a green suite is evidence about the
edits that were guarded, not the edit you just made.

Mutation-testing the eight concessions on 2026-09-19 -- break each in turn, require a named
failure -- found SIX that could be deleted or sign-flipped with nothing failing. Two were already
guarded (the two-anchor climb, and "no joint table is built"). These four close the gap for the
ones whose source data supports a derivation; each asserts the SHAPE the CSV has, so a withdrawal
has to survive the data rather than a rewording.
"""
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from tests.manuscript import tex  # noqa: E402


def _rows(name):
    p = os.path.join(ROOT, "results", name)
    assert os.path.exists(p), p
    return list(csv.DictReader(open(p, encoding="utf-8")))


def _tex(name):
    return " ".join(open(tex(name), encoding="utf-8").read().split())


def test_the_compute_matched_loss_is_stated_with_its_sign_and_its_interval():
    """The paper's most damaging concession: held to the meter's own compute, selection LOSES.

    Nothing read it. `0.0395` appeared in no test in the repository: test_compute_matched.py pins
    the arm's COST (0.92x) and the worst ratio (61.3x), so the sentence could keep its cost and
    silently flip its result. Both the sign and the interval are derived from the F4 band here,
    and the point estimate is cross-checked against the two gains it is the difference of.
    """
    f4 = [r for r in _rows("compute_matched_bands.csv") if r["band"].startswith("F4")]
    assert len(f4) == 1, f4
    val, lo, hi = (float(f4[0][k]) for k in ("value", "lo95", "hi95"))
    assert val < 0 and hi < 0, (val, hi, "F4 is no longer a loss; the paper's sentence must change")
    assert f4[0]["reading"] == "MATCHED-COMPUTE LOSS", f4[0]["reading"]

    # the point estimate IS the difference of the two arms, not a third arithmetic (caution (j))
    g = {r["arm"]: float(r["gain"]) for r in _rows("compute_matched.csv")}
    assert abs((g["sel05b_n4"] - g["metered_k10"]) - val) < 5e-4, (g["sel05b_n4"], g["metered_k10"])

    body = _tex("sections/selection.tex")
    assert f"${val:+.4f}$" in body, f"Section 3 lost the matched-compute loss {val:+.4f}"
    assert f"$[{lo:+.4f}, {hi:+.4f}]$" in body, f"Section 3 lost its interval [{lo:+.4f}, {hi:+.4f}]"
    close = _tex("sections/iclr_closing.tex")
    i = close.find("held to the meter's own compute")
    assert i != -1, "the Conclusion no longer concedes the compute-matched comparison"
    assert "loses" in close[i:i + 80], \
        "the Conclusion states the compute-matched comparison without conceding the loss"


def test_the_overoptimisation_concession_is_stated_at_both_sites():
    """The pointwise reward turns over on TriviaQA, and the appendix says so twice.

    Deleting it from the Appendix~I prose fired nothing, because test_overoptimisation_claim.py is
    satisfied by the OTHER occurrence in the judge-free table caption -- caution (an)'s "a guard
    satisfied by a different occurrence of its phrase is not guarding its sentence". So locate the
    caption and require the direction on both sides of it.
    """
    pw = [r for r in _rows("selection_verifiable_tqa_comma7b.csv")
          if r["arm"].startswith("pointwise")]
    worst = min(pw, key=lambda r: float(r["gain"]))
    assert float(worst["gain_hi95"]) < 0, \
        (worst["n"], worst["gain_hi95"], "no arm excludes zero on the wrong side any more")
    assert float(worst["spearman_acc_logn"]) < 0, worst["spearman_acc_logn"]

    apx = _tex("sections/appendix_selection.tex")
    assert f"at $n={worst['n']}$" in apx, f"the appendix no longer names n={worst['n']}"
    label = apx.index(r"\label{tab:judgefree}")
    occ = [m.start() for m in re.finditer("wrong side", apx)]
    assert any(abs(o - label) < 1500 for o in occ), \
        "the judge-free table caption lost the direction of the TriviaQA turnover"
    assert any(abs(o - label) >= 1500 for o in occ), \
        "the Appendix I prose lost the direction of the TriviaQA turnover"


def test_mt_bench_is_reported_as_resolving_nothing():
    """80 prompts, both judges' intervals containing zero at a half-width of about 0.09.

    A pre-registered arm that decides nothing is the easiest thing in the paper to quietly drop,
    and dropping it would leave AlpacaEval looking like the whole of the benchmark evidence.
    """
    rows = [r for r in _rows("selection_scaling_mtbench.csv") if int(float(r["n"])) == 8]
    assert len(rows) == 2, rows
    hw = []
    for r in rows:
        lo, hi = float(r["gain_lo95"]), float(r["gain_hi95"])
        assert lo < 0 < hi, (r["judge"], lo, hi, "this judge now resolves MT-Bench; rewrite the text")
        hw.append((hi - lo) / 2)
    n_prompts = {int(float(r["n_prompts"])) for r in rows}
    assert n_prompts == {80}, n_prompts

    lim = _tex("sections/appendix_limitations.tex")
    assert "resolve nothing in either direction" in lim, \
        "Limitations no longer concedes that MT-Bench resolves nothing"
    assert f"${max(hw):.2f}$" in lim, f"the conceded half-width is not the CSV's {max(hw):.2f}"
    assert f"MT-Bench's ${min(n_prompts)}$ prompts" in lim, "the prompt count is not stated"


def test_the_single_corpus_scope_is_conceded_with_the_count_the_corpus_has():
    """Every extraction measurement is sixteen English genre novels, and the number is the
    committed corpus's own. Derived rather than spelled: if a corpus is ever added, this fails and
    says to revisit the wording instead of letting a stale 'sixteen' stand (caution (j))."""
    import glob
    import json
    novels = set()
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "copybench_*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            novels.add(json.loads(line)["source_novel"])
    word = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen"][len(novels)]
    lim = _tex("sections/appendix_limitations.tex")
    assert f"{word} English genre novels" in lim, \
        (len(novels), f"Limitations must concede the corpus scope as '{word} English genre novels'")
    close = _tex("sections/iclr_closing.tex")
    assert f"{word} English novels" in close or f"{word} English genre novels" in close, \
        f"the Conclusion no longer concedes that extraction evidence is {word} novels"


def test_the_onset_interval_width_is_the_one_the_table_currently_holds():
    """"More passages do not narrow the interval" -- and the number saying so had gone stale.

    The mean width was written as $0.256$ and matched the table when it held FIVE pairs at 100
    passages (commit d8f1879, mean 0.2569). Four pairs were added afterwards and the sentence was
    never re-derived: the mean is 0.2854 over the eight pairs at 100 passages. Nothing caught it.
    No analysis script computes the number, no test read it, and audit_numbers.py passed it because
    the literal 0.256 does occur in a committed CSV -- in an unrelated scorer-cost table.

    This is caution (ai) with the set GROWING under a claim about it: adding a pair silently
    invalidates every mean quoted over the pairs. Derive it here so the next pair fails this test
    instead of ageing the sentence.

    NOTE for whoever adds pair ten: the companion figure $0.219$, Pleias-350M's width at n=100, has
    no committed source -- it predates onset_table.csv's git history, which already has that pair
    at n=458. It is internally consistent (0.219 * sqrt(100/458) = 0.102, the quoted prediction)
    and its n=458 twin, $0.215$, matches the table exactly, so it is not asserted here.
    """
    import math
    rs = [r for r in _rows("onset_table.csv") if not r["pair"].startswith("ALL")]
    at100 = [r for r in rs if int(r["n_passages"]) == 100]
    assert at100, "no pair is measured at 100 passages; the sentence needs rewriting"
    widths = [float(r["ratio_hi"]) - float(r["ratio_lo"]) for r in at100]
    mean = sum(widths) / len(widths)

    lim = _tex("sections/appendix_limitations.tex")
    assert f"mean width of ${mean:.3f}$" in lim, \
        (f"the conceded mean width is not the table's {mean:.3f} over {len(at100)} pairs",
         "a pair was probably added without re-deriving it")

    # and the shape: quadrupling the passages left the width far above sqrt-n scaling
    big = [r for r in rs if int(r["n_passages"]) > 100]
    assert len(big) == 1, big
    w_big, n_big = (float(big[0]["ratio_hi"]) - float(big[0]["ratio_lo"]),
                    int(big[0]["n_passages"]))
    assert f"${w_big:.3f}$ at $n={n_big}$" in lim, (w_big, n_big, "the wide arm is misquoted")
    predicted = mean / math.sqrt(n_big / 100)
    assert w_big > predicted, \
        (w_big, predicted, "the interval now DOES shrink with passages; the concession must change")

    # The adjective is the claim, so it is asserted too. A phrase list is normally caution (an)'s
    # mistake, but here it fails in the SAFE direction: a reword breaks the test loudly instead of
    # retiring it silently, which is what "make withdrawal deliberate" asks for.
    i = lim.index(f"mean width of ${mean:.3f}$")
    w = lim[i:i + 320]
    assert any(p in w for p in ("does not shrink", "do not shrink",
                                "does not narrow", "do not narrow")), \
        "the sentence no longer says the interval fails to narrow; the data still says it does not"


# ---------------------------------------------------------------------------------------------
# Five more, found by mutation-testing the CONCESSIONS on 2026-09-22 (caution (ag): a length edit
# deletes these first; caution (aq): a green suite is evidence about the edits that were guarded).
# Eight were broken one at a time and five left the suite passing. In four of the five the NUMBER
# was guarded and the VERDICT WORD was not -- caution (av) exactly: a verdict is a string that
# outlives the number under it, and softening it is the cheapest way to delete a concession.
# ---------------------------------------------------------------------------------------------

def _app_sel():
    from tests.manuscript import body
    return body("appendix_selection.tex")


def test_the_second_opponent_verdict_word_matches_its_own_interval():
    """`The difference between them does not survive` could be softened to `is smaller` with the
    whole suite green: the band is pinned, the word is not. `is smaller` is a comparison of point
    estimates; `does not survive` is the registered reading for an interval containing zero, and
    those are different claims."""
    import csv as _csv
    import os as _os
    rows = [r for r in _csv.DictReader(open(_os.path.join(
        ROOT, "results", "order_averaged_h2h__opp2.csv"), encoding="utf-8"))
        if r["quantity"].startswith("D3")]
    assert len(rows) == 1
    lo, hi = float(rows[0]["lo95"]), float(rows[0]["hi95"])
    txt = _app_sel()
    # SCOPE IT (caution (an)): `does not survive` occurs twice in this file, once in the
    # paragraph heading two sentences earlier, so a bare `in txt` passed the mutation that
    # softened the claim itself. Anchor on the sentence that carries the band.
    claim = "The difference between them does not survive}: "
    if lo <= 0 <= hi:
        assert claim in txt, (
            f"the interval [{lo}, {hi}] contains zero, so the registered reading is UNRESOLVED; "
            "the appendix must say the difference does not survive, not that it is smaller")
        i = txt.find(claim)
        assert f"{lo:+.4f}" in txt[i:i + 200], \
            "the verdict and its own interval were separated"
    else:
        assert claim not in txt, \
            "the interval no longer contains zero; the concession's wording must be revisited"


def test_the_self_preference_disclosure_survives():
    """Judge C is the fixed opponent's own checkpoint. Caution (aa) records that this was nowhere
    stated until 2026-09-15; it is a disclosure the paper chose to make and it deletes cleanly.
    Conditioned on the fact itself -- the judge panel and the opponent name the same model."""
    from tests.manuscript import body
    txt = _app_sel() + body("appendix_limitations.tex")
    assert "Llama-3.1-8B-Instruct" in txt, "the fixed opponent is no longer named"
    # The phrase occurs twice in appendix_selection -- once about the judge panel, once about the
    # second-opponent arm -- so a bare membership test passed the mutation that removed one of
    # them. Both are disclosures and both must stay (caution (an)).
    assert txt.count("opponent's own checkpoint") >= 2, (
        "a self-preference disclosure was deleted; judge C is still the opponent's own checkpoint "
        "and the paper still reports judge C in both the panel and the opponent arm")


def test_the_workload_verdict_word_matches_its_own_interval():
    """`the reversal \\emph{fails}` could become `\\emph{narrows}` with nothing failing. On
    AlpacaEval the paired difference is negative with its interval excluding zero, which is a
    failure and not a narrowing."""
    import csv as _csv
    import os as _os
    rows = [r for r in _csv.DictReader(open(_os.path.join(
        ROOT, "results", "order_averaged_h2h__mixpowk_judgeB.csv"), encoding="utf-8"))
        if r["quantity"].startswith("D3")]
    assert len(rows) == 1
    g, hi = float(rows[0]["value"]), float(rows[0]["hi95"])
    assert g < 0 and hi < 0, f"AlpacaEval no longer reverses ({g}, hi {hi}); revisit the wording"
    assert "the reversal \\emph{fails}" in _app_sel(), \
        "the appendix softened the AlpacaEval verdict; its own interval excludes zero on that side"


def test_the_order_consistency_concession_keeps_below_chance():
    """`the latter below chance agreement between presentation orders` could become `close to
    chance`. Below chance is the damaging reading and it is what the number says."""
    import csv as _csv
    import os as _os
    rows = [r for r in _csv.DictReader(open(_os.path.join(
        ROOT, "results", "order_averaged_h2h__opp2.csv"), encoding="utf-8"))
        if r["quantity"] == "order consistency"]
    met = [float(r["value"]) for r in rows if r["arm"].startswith("metered")]
    assert met, "the metered arm's order consistency is gone from that pass"
    assert met[0] < 0.5, f"order consistency is {met[0]}, no longer below chance; revisit"
    assert "below chance agreement" in _app_sel(), \
        f"the appendix softened a consistency of {met[0]}, which is below chance"


def test_the_audited_anchors_empty_rate_is_quoted_with_its_own_threshold():
    """The audited anchor emits an empty completion on 6.8% of prompts, above the 5% its own
    registration set (caution (p)). Both the rate and the ladder it sits in deleted cleanly."""
    import csv as _csv
    import os as _os
    from tests.manuscript import body
    rows = list(_csv.DictReader(open(_os.path.join(
        ROOT, "results", "selection_breadth.csv"), encoding="utf-8")))
    empties = sorted({round(float(r[c]) * 100, 1) for r in rows
                      for c in r if c.startswith("empty") and r[c] not in ("", None)})
    txt = body("appendix_limitations.tex")
    # `$6.8\%$` appears twice -- once as the anchor's own rate, once in the ladder -- so a bare
    # membership test passed the mutation that deleted the first (caution (an)).
    assert "emits one on $6.8\\%$ of the $500$ prompts" in txt, (
        "the audited anchor's empty rate was deleted; it is the one that exceeds the 5% threshold "
        f"its own registration set (rates on record: {empties})")
    for lit in ("$0.0\\%$", "$0.2\\%$", "$3.0\\%$"):
        assert lit in txt, f"the empty-fraction ladder lost {lit}, so 6.8% has nothing to sit in"
