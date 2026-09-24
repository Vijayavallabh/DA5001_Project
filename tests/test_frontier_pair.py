"""The head-to-head at a second pair, and the two things that make it honest.

The paper's central comparison -- selection buys more judged utility than the metered decoder at a
hundredth of the divergence -- was one (anchor, risky) pair. This pins the second one: that both
mechanisms are judged against the SAME opponent with ONE shared control, that the pair carries no
certificate claim, and that the numbers in the scoring log round from the CSV.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "frontier_pair_llama321b.csv")
LOG = os.path.join(ROOT, "results", "onset_prediction_frontier_second_pair.md")
SCORER = "Phi-3.5-mini-instruct"


def rows():
    with open(CSV, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def by_judge(j):
    return {r["arm"]: r for r in rows() if r["judge"] == j}


def passage():
    """The two further pairs' paragraph. v10 (2026-09-24) moved it from appendix_proofs.tex to
    appendix_onset.tex (App. H, "Two anchors closer to the risky model", label app:frontier3); it is
    located by its label and cut at the next paragraph, so a phrase elsewhere in the file (that
    file's scaling paragraph says "no shared vocabulary is needed") cannot satisfy a guard on it."""
    from tests.manuscript import body
    t = body("appendix_onset.tex")
    i = t.index("\\label{app:frontier3}")
    j = t.find("\\paragraph", i)
    return t[i: j if j > i else len(t)]


def test_both_judges_scored_every_arm():
    js = {r["judge"] for r in rows()}
    assert len(js) == 2 and SCORER in js, js
    arms = {r["arm"] for r in rows() if r["judge"] == SCORER}
    assert "anchor alone (control)" in arms
    assert sum(1 for a in arms if a.startswith("metered")) == 4, arms
    assert sum(1 for a in arms if a.startswith("selection")) == 4, arms
    for r in rows():
        assert int(r["n_prompts"]) == 500, r["arm"]


def test_selection_beats_the_best_metered_arm_at_a_fraction_of_the_spend():
    """F1 REPLICATES, and the band is checked here rather than trusted from the log."""
    for j in {r["judge"] for r in rows()}:
        a = by_judge(j)
        met = [v for k, v in a.items() if k.startswith("metered")]
        best = max(met, key=lambda r: float(r["gain"]))
        sel = a["selection, n=8"]
        assert float(sel["gain_lo95"]) > 0, (j, sel["gain_lo95"])
        assert float(sel["gain"]) >= float(best["gain"]) - 0.03, (j, sel["gain"], best["gain"])
        assert float(sel["spend_nats"]) < 0.1 * float(best["spend_nats"]), (j, sel, best)


def test_the_metered_spend_saturates_as_proposition_3_says():
    """k=3 and k=20 differ by a factor of seven in the published cap and by 0.2 nats in what is
    actually spent. If that ever separates, Proposition 3's measurement here has changed."""
    a = by_judge(SCORER)
    s3, s20 = float(a["metered, k=3"]["spend_nats"]), float(a["metered, k=20"]["spend_nats"])
    assert abs(s20 - s3) < 1.0, (s3, s20)
    assert abs(float(a["metered, k=20"]["gain"]) - float(a["metered, k=3"]["gain"])) < 0.01


def test_one_shared_control_and_the_within_pool_gain_beside_it():
    """The pre-registration's parenthetical was wrong and the fix is a single shared control, with
    selection's own n=1 reported beside it. Both columns must exist and differ."""
    a = by_judge(SCORER)
    sel = a["selection, n=8"]
    assert sel["gain_vs_own_n1"] != "", "the within-pool column is gone"
    assert abs(float(sel["gain"]) - float(sel["gain_vs_own_n1"])) > 0.01, sel
    assert float(sel["gain"]) < float(sel["gain_vs_own_n1"]), "the reported number is the smaller"
    assert a["metered, k=3"]["gain_vs_own_n1"] == "", "a metered arm has no n=1 of its own"


def test_the_log_reports_the_conservative_number_and_says_so():
    # newlines collapsed: the phrases below wrap in the source
    t = " ".join(open(LOG, encoding="utf-8").read().split())
    assert "the more conservative of the two" in t
    assert "parenthetical is wrong" in t, "the protocol correction must stay disclosed"


def test_no_certificate_claim_is_made_at_this_pair():
    """Llama-3.2-1B is not a safe model. Excluded alternative 2 is enforced here, not just written."""
    t = open(LOG, encoding="utf-8").read()
    assert "is not a safe model" in t
    for banned in ("s(x)", "near-verbatim", "vacuit"):
        pass  # the words may appear in the prohibition itself; what must not appear is a number
    import re
    body = t.split("## Scoring, 2026-09-12")[1]
    assert not re.search(r"recall (is|of) `?0\.\d", body), "a leakage number leaked into this arm"


def test_the_scoring_log_rounds_from_the_csv():
    t = open(LOG, encoding="utf-8").read().split("## Scoring, 2026-09-12")[1]
    for r in rows():
        if r["arm"] == "anchor alone (control)" or r["arm"] == "selection, n=1":
            continue
        g, lo, hi = float(r["gain"]), float(r["gain_lo95"]), float(r["gain_hi95"])
        assert f"`{g:+.3f} [{lo:+.3f}, {hi:+.3f}]`" in t, (r["judge"], r["arm"], g)


def test_the_appendix_table_rounds_from_the_csv():
    """Rescoped 2026-09-19. The twelve-cell table left the manuscript with the appendix reduction
    (appendix 30 -> 25 pages); the arm, its scoring log and its CSV are unchanged. What the paper
    still prints is the reading -- which arms resolve, in which direction, and what they spent --
    so that is what is checked against the CSV here, cell by cell where a cell survives.
    """
    apx = passage()
    metered = [r for r in rows() if r["arm"].startswith("metered") and r["judge"] == SCORER]
    sel = [r for r in rows() if r["arm"].startswith("selection, n=8") and r["judge"] == SCORER]
    assert len(metered) == 4 and len(sel) == 1, (len(metered), len(sel))
    # the metered arms' realised spend is still quoted -- that is the paper's point about this
    # pair, and it is what the prose reading rests on. v10 (2026-09-24) quotes it as a range over
    # the four arms, "for $88.1$ to $123.4$ nats", so the cells that survive are its two ends.
    spends = [float(r["spend_nats"]) for r in metered]
    assert f"${min(spends):.1f}$ to ${max(spends):.1f}$ nats" in apx, spends
    # and the direction of every metered gain is still stated correctly
    gains = [float(r["gain"]) for r in metered]
    if all(g < 0 for g in gains):
        assert "resolves at no budget on either judge" in apx, gains
        assert f"${min(gains):+.3f}$" in apx and f"${max(gains):+.3f}$" in apx, gains
    else:
        assert "the meter is useful" in apx, gains     # v9: "genuinely useful here"
        for r in metered:
            if float(r["gain"]) > 0 and r["arm"] != "metered, k=0.5":
                assert f"${float(r['gain']):+.3f}$" in apx, r["arm"]
    # selection at n=8 beats them for 1.204 nats, quoted with its interval
    r = sel[0]
    assert f"${float(r['gain']):+.3f}$ $[{float(r['gain_lo95']):+.3f}, {float(r['gain_hi95']):+.3f}]$" \
        in apx, (r["arm"], "selection's own cell is no longer quoted")


def test_the_second_pair_is_reported_only_beside_its_not_a_safe_model_caveat():
    """Re-derived 2026-09-24. This guard used to REQUIRE Section 4 to quote the pair, and a referee
    showed why that was wrong: Llama-3.2-1B is not a safe model, so a main-text reader takes
    "+0.076 for 1.204 nats" as further evidence for the COPYRIGHT claim when it is a
    vocabulary-matched ablation of the meter. The numbers stay -- in the appendix, in the same
    passage that says neither anchor is safe -- and Section 4 may quote them only with that caveat.
    """
    import re
    from tests.manuscript import tex
    a = by_judge(SCORER)
    sel = a["selection, n=8"]
    met = [v for k, v in a.items() if k.startswith("metered")]
    best = max(met, key=lambda r: float(r["gain"]))
    # v10 (2026-09-24): the passage lives in appendix_onset.tex and opens with the caveat, "Neither
    # anchor here is a safe model"; emphasis and the "here" are spelling, the caveat is the claim
    text = passage()
    assert re.search(r"[Nn]either anchor (?:here )?is a (?:\\emph\{safe\}|safe) model", text), \
        "the caveat left the pair's own passage"
    assert f"${float(sel['gain']):+.3f}$ $[" in text, "selection's cell left the appendix passage"
    # the meter's best arm: its gain is quoted, and its spend lies inside the quoted range
    spends = [float(r["spend_nats"]) for r in met]
    assert f"${float(best['gain']):+.3f}$" in text, "the meter's best arm left the appendix passage"
    assert f"${min(spends):.1f}$ to ${max(spends):.1f}$ nats" in text, \
        "the meter's spend left the appendix passage"
    body = " ".join(open(tex("sections/experiments.tex"), encoding="utf-8").read().split())
    for needle in ("Llama-3.2-1B", f"${float(sel['gain']):+.3f}$ for ${float(sel['spend_nats']):.3f}$ nats"):
        j = body.find(needle)
        if j >= 0:
            assert "not" in body[max(0, j - 250):j + 250] and "safe" in body[max(0, j - 250):j + 250], \
                "Section 4 quotes the non-safe pair without saying it is not a safe model"


def test_the_shared_vocabulary_constraint_is_stated_where_it_bites():
    """A reader must not conclude we simply did not bother repeating the comparison.

    v10 (2026-09-24): the pairs' passage moved from appendix_proofs.tex to appendix_onset.tex and
    is checked there, scoped to the passage itself (see passage())."""
    from tests.manuscript import tex
    for f, t in (("sections/experiments.tex",
                  " ".join(open(tex("sections/experiments.tex"), encoding="utf-8").read().split())),
                 ("appendix_onset.tex, the further-pairs passage", passage())):
        assert "shared vocabulary" in t or "the risky model's tokenizer" in t, f


def test_the_two_nominally_identical_control_arms_are_reported_not_smoothed():
    """The pass judges the metered run's k=0 arm and the selection run's n=1 arm, which are the
    same thing generated twice. A reviewer reading the released CSV finds the discrepancy whether
    or not we mention it, so the appendix reports it as this paper's generation-run noise floor."""
    n1 = {r["judge"]: r for r in rows() if r["arm"] == "selection, n=1"}
    assert len(n1) == 2, "both judges must score the duplicated control"
    # v10 (2026-09-24) reports the duplicated arms of BOTH further pairs as one bound, "differ by at
    # most $0.034$", and names it the cross-pass floor (v9: the "generation-run" floor, each gain
    # printed). So the bound is rebuilt from both pairs' CSVs, and every interval must still hold 0.
    both = list(n1.values())
    third = os.path.join(ROOT, "results", "frontier_pair_llama323bi.csv")
    assert os.path.exists(third), f"{third} missing; this guard must not pass by never running"
    both += [r for r in csv.DictReader(open(third, encoding="utf-8")) if r["arm"] == "selection, n=1"]
    assert len(both) == 4, "both judges must score the duplicated control at both pairs"
    for r in both:
        lo, hi = float(r["gain_lo95"]), float(r["gain_hi95"])
        assert lo < 0 < hi, ("the two anchor-alone arms differ significantly; the appendix "
                             "sentence reading this as a noise floor is now false")
    text = passage()
    assert "cross-pass floor" in text, "the noise floor is measured but not reported"
    assert f"differ by at most ${max(abs(float(r['gain'])) for r in both):.3f}$" in text, \
        [r["gain"] for r in both]
