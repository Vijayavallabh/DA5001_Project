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


def test_the_scoring_log_and_the_appendix_round_from_the_csv():
    """Rescoped 2026-09-19. The twelve-cell table left the manuscript with the appendix reduction
    (appendix 30 -> 25 pages); the arm, its scoring log and its CSV are unchanged. What the paper
    still prints is the reading -- which arms resolve, in which direction, and what they spent --
    so that is what is checked against the CSV here, cell by cell where a cell survives.
    """
    from tests.manuscript import tex
    log = " ".join(open(LOG, encoding="utf-8").read().split())
    apx = " ".join(open(tex("sections/appendix_proofs.tex"), encoding="utf-8").read().split())
    assert "REPLICATES" in log
    metered = [r for r in rows() if r["arm"].startswith("metered") and r["judge"] == SCORER]
    sel = [r for r in rows() if r["arm"].startswith("selection, n=8") and r["judge"] == SCORER]
    assert len(metered) == 4 and len(sel) == 1, (len(metered), len(sel))
    # every metered arm's realised spend is still quoted -- that is the paper's point about this
    # pair, and it is what the prose reading rests on
    for r in metered:
        assert f"${float(r['spend_nats']):.1f}$" in apx, r["spend_nats"]
    # and the direction of every metered gain is still stated correctly
    gains = [float(r["gain"]) for r in metered]
    if all(g < 0 for g in gains):
        assert "resolves at no budget on either judge" in apx, gains
        assert f"${min(gains):+.3f}$" in apx and f"${max(gains):+.3f}$" in apx, gains
    else:
        assert "genuinely useful here" in apx, gains
        for r in metered:
            if float(r["gain"]) > 0 and r["arm"] != "metered, k=0.5":
                assert f"${float(r['gain']):+.3f}$" in apx, r["arm"]
    # selection at n=8 beats them for 1.204 nats, quoted with its interval
    r = sel[0]
    assert f"${float(r['gain']):+.3f}$ $[{float(r['gain_lo95']):+.3f}, {float(r['gain_hi95']):+.3f}]$" \
        in apx, (r["arm"], "selection's own cell is no longer quoted")


def test_the_third_pair_is_reported_as_one_of_two_non_safe_pairs():
    """Re-derived 2026-09-24, for the reason test_frontier_pair.py gives: Llama-3.2-3B-Instruct is
    not a safe model, so its numbers belong beside that caveat and not in Section 4's copyright
    comparison. The appendix must still call them two further pairs and quote both mechanisms."""
    from tests.manuscript import tex
    apx = " ".join(open(tex("sections/appendix_proofs.tex"), encoding="utf-8").read().split())
    assert "Two further pairs" in apx, "the appendix no longer says there are two further pairs"
    i = apx.index("\\label{app:frontier3}")
    passage = apx[i:i + 1600]
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
