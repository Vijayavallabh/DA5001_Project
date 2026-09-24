"""feat-104: the head-to-head at a third pair, the closest anchor on record.

The second-pair test pins that both mechanisms are judged against one opponent with one shared
control. This one pins the thing that is new here and that the pre-registration did NOT predict:
the metered decoder resolves at no budget on either judge, so the paper's sentence about it is a
statement of non-resolution and fails here if any arm ever resolves.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "frontier_pair_llama323bi.csv")
LOG = os.path.join(ROOT, "results", "onset_prediction_frontier_third_pair.md")
SCORER = "Phi-3.5-mini-instruct"


def rows():
    with open(CSV, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def by_judge(j):
    return {r["arm"]: r for r in rows() if r["judge"] == j}


def test_both_judges_scored_every_arm_on_one_shared_control():
    js = {r["judge"] for r in rows()}
    assert len(js) == 2 and SCORER in js, js
    for j in js:
        a = by_judge(j)
        assert "anchor alone (control)" in a
        assert sum(1 for k in a if k.startswith("metered")) == 4, sorted(a)
        assert sum(1 for k in a if k.startswith("selection")) == 4, sorted(a)
    for r in rows():
        assert int(r["n_prompts"]) == 500, r["arm"]


def test_the_metered_decoder_resolves_at_no_budget_on_either_judge():
    """The appendix says all eight intervals contain zero and all eight estimates are negative.
    If that stops being true the trivial-horn reading of this pair is false."""
    for j in {r["judge"] for r in rows()}:
        for arm, r in by_judge(j).items():
            if not arm.startswith("metered"):
                continue
            lo, hi, g = float(r["gain_lo95"]), float(r["gain_hi95"]), float(r["gain"])
            assert lo < 0 < hi or hi <= 0, (j, arm, lo, hi)
            assert lo <= 0 <= hi, (j, arm, "a metered arm now resolves", lo, hi)
            assert g < 0, (j, arm, "a metered arm's point estimate is no longer negative", g)


def test_selection_wins_by_more_than_the_band_asked_at_a_fiftieth_of_the_spend():
    for j in {r["judge"] for r in rows()}:
        a = by_judge(j)
        best = max((v for k, v in a.items() if k.startswith("metered")),
                   key=lambda r: float(r["gain"]))
        sel = a["selection, n=8"]
        assert float(sel["gain_lo95"]) > 0, (j, sel)
        assert float(sel["gain"]) >= float(best["gain"]) - 0.03, (j, sel["gain"], best["gain"])
        assert float(sel["spend_nats"]) < 0.1 * float(best["spend_nats"]), (j, sel, best)


def test_the_spend_saturates_while_the_cap_grows_fortyfold():
    a = by_judge(SCORER)
    lo = float(a["metered, k=0.5"]["spend_nats"])
    hi = float(a["metered, k=20"]["spend_nats"])
    assert hi / lo < 1.15, (lo, hi, "the spend no longer saturates; Prop 3's reading here changes")


def test_the_duplicated_control_is_reported_not_smoothed():
    n1 = {r["judge"]: r for r in rows() if r["arm"] == "selection, n=1"}
    assert len(n1) == 2
    for r in n1.values():
        lo, hi = float(r["gain_lo95"]), float(r["gain_hi95"])
        assert lo < 0 < hi, (r["judge"], "the two anchor-alone arms now differ significantly")


def _closer_anchors_paragraph():
    """The 'Two anchors closer to the risky model' paragraph (\\label{app:frontier3}), heading
    included. Through v9 it sat in appendix_proofs.tex; the v10 restructure (2026-09-24) moved it to
    Appendix H in appendix_onset.tex. Scoped to the paragraph so a number elsewhere in the appendix
    cannot satisfy a guard meant for this pair (caution (an))."""
    from tests.manuscript import body
    txt = body("appendix_onset.tex")
    assert txt.count(r"\label{app:frontier3}") == 1, "the closer-anchors paragraph is gone"
    i = txt.index(r"\label{app:frontier3}")
    end = txt.find(r"\paragraph{", i)
    return txt[txt.rindex(r"\paragraph{", 0, i): end if end >= 0 else len(txt)]


def test_the_scoring_log_and_the_appendix_round_from_the_csv():
    """Rescoped 2026-09-19. The twelve-cell table left the manuscript with the appendix reduction
    (appendix 30 -> 25 pages); the arm, its scoring log and its CSV are unchanged. What the paper
    still prints is the reading -- which arms resolve, in which direction, and what they spent --
    so that is what is checked against the CSV here, cell by cell where a cell survives.

    Rescoped again for v10 (2026-09-24). The paragraph now prints the four realised spends as their
    range ("spends $54.8$ to $60.6$ nats over a $40\\times$ range of caps") and the direction of the
    eight metered gains as "all eight point estimates negative"; v9's per-arm spends ($59.3$, $60.3$)
    and per-judge gain ranges ($-0.038$ to $-0.019$, $-0.030$ to $-0.009$) are no longer printed.
    Both surviving statements are rebuilt from the CSV, and selection's cell is now checked under
    BOTH judges, since the paragraph prints both.
    """
    log = " ".join(open(LOG, encoding="utf-8").read().split())
    apx = _closer_anchors_paragraph()
    assert "REPLICATES" in log
    metered = [r for r in rows() if r["arm"].startswith("metered") and r["judge"] == SCORER]
    sel = [r for r in rows() if r["arm"].startswith("selection, n=8") and r["judge"] == SCORER]
    assert len(metered) == 4 and len(sel) == 1, (len(metered), len(sel))
    # the realised spends are still quoted, as the range over every metered arm -- that saturation
    # is the paper's point about this pair, and it is what the prose reading rests on
    spends = [float(r["spend_nats"]) for r in metered]
    assert f"spends ${min(spends):.1f}$ to ${max(spends):.1f}$ nats" in apx, spends
    # and the direction of every metered gain, on both judges, is still stated correctly
    every = [r for r in rows() if r["arm"].startswith("metered")]
    gains = [float(r["gain"]) for r in every]
    assert len(gains) == 8, len(gains)
    if all(g < 0 for g in gains):
        assert "resolves at no budget on either judge" in apx, gains
        assert "all eight point estimates negative" in apx, gains
    else:
        assert "resolves at no budget" not in apx and "all eight point estimates negative" not in apx, \
            gains
        for r in every:
            if float(r["gain"]) > 0:
                assert f"${float(r['gain']):+.3f}$" in apx, (r["judge"], r["arm"])
    # selection at n=8 beats them for 1.204 nats, quoted with its interval, under both judges
    for r in (x for x in rows() if x["arm"] == "selection, n=8"):
        assert (f"${float(r['gain']):+.3f}$ $[{float(r['gain_lo95']):+.3f}, "
                f"{float(r['gain_hi95']):+.3f}]$") in apx, (r["judge"], "selection's cell is gone")


def test_the_third_pair_is_reported_as_one_of_two_non_safe_pairs():
    """Re-derived 2026-09-24, for the reason test_frontier_pair.py gives: Llama-3.2-3B-Instruct is
    not a safe model, so its numbers belong beside that caveat and not in Section 4's copyright
    comparison. The appendix must still call them two further pairs and quote both mechanisms."""
    from tests.manuscript import tex
    # v10 (2026-09-24): the paragraph is headed "Two anchors closer to the risky model" and opens on
    # the caveat "Neither anchor here is a safe model"; v9 said "Two further pairs". Same claim.
    passage = _closer_anchors_paragraph()
    assert "Two anchors closer to the risky model" in passage, \
        "the appendix no longer says there are two further pairs"
    assert "Neither anchor here is a safe model" in passage, "the not-a-safe-model caveat is gone"
    a = by_judge(SCORER)
    sel, best = a["selection, n=8"], max(
        (v for k, v in a.items() if k.startswith("metered")), key=lambda r: float(r["gain"]))
    assert f"${float(sel['gain']):+.3f}$" in passage, sel["gain"]
    assert f"${float(best['spend_nats']):.1f}$" in passage, best
    body = " ".join(open(tex("sections/experiments.tex"), encoding="utf-8").read().split())
    j = body.find("Llama-3.2-3B")
    if j >= 0:
        assert "not" in body[max(0, j - 250):j + 250] and "safe" in body[max(0, j - 250):j + 250], \
            "Section 4 quotes the non-safe third pair without saying it is not a safe model"


def test_no_certificate_claim_is_made_at_this_pair():
    """Llama-3.2-3B-Instruct is not a safe model. T3 is enforced, not just written."""
    import re
    t = open(LOG, encoding="utf-8").read()
    assert "is not a safe model" in t
    body = t.split("## Scoring, 2026-09-14")[1]
    assert not re.search(r"recall (is|of) `?0\.\d", body), "a leakage number leaked into this arm"
