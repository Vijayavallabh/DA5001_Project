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
    """Every cell of the second-pair table comes from frontier_pair_llama321b.csv, once."""
    from tests.manuscript import tex
    apx = " ".join(open(tex("sections/appendix_proofs.tex"), encoding="utf-8").read().split())
    quoted = 0
    for r in rows():
        if r["arm"] in ("anchor alone (control)", "selection, n=1", "selection, n=2"):
            continue
        g, lo, hi = float(r["gain"]), float(r["gain_lo95"]), float(r["gain_hi95"])
        cell = f"${g:+.3f}$ $[{lo:+.3f}, {hi:+.3f}]$"
        assert cell in apx, (r["judge"], r["arm"], cell)
        quoted += 1
    assert quoted == 12, quoted
    for r in rows():
        if r["arm"].startswith("metered") and r["judge"] == SCORER:
            assert f"${float(r['spend_nats']):.1f}$" in apx, r["spend_nats"]


def test_section6_carries_the_second_pair_reversal():
    from tests.manuscript import tex
    body = " ".join(open(tex("sections/experiments.tex"), encoding="utf-8").read().split())
    a = by_judge(SCORER)
    sel = a["selection, n=8"]
    best = max((v for k, v in a.items() if k.startswith("metered")), key=lambda r: float(r["gain"]))
    assert f"${float(sel['gain']):+.3f}$ for ${float(sel['spend_nats']):.3f}$ nats" in body
    assert f"${float(best['gain']):+.3f}$ for ${float(best['spend_nats']):.1f}$" in body
    assert "the reversal repeats" in body


def test_the_shared_vocabulary_constraint_is_stated_where_it_bites():
    """A reader must not conclude we simply did not bother repeating the comparison."""
    from tests.manuscript import tex
    for f in ("sections/experiments.tex", "sections/appendix_proofs.tex"):
        t = " ".join(open(tex(f), encoding="utf-8").read().split())
        assert "shared vocabulary" in t or "the risky model's tokenizer" in t, f
