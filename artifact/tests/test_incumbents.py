"""The incumbent comparison, and the three concessions it forced into the paper.

Caution (ag): a length edit deletes concessions first, because they read as cuttable. Every guard
here derives its claim from the measurement that produced it, so the sentence cannot be softened
without the CSV changing too.
"""
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import body  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _decode_rows():
    p = os.path.join(ROOT, "results", "blocklist_decode.csv")
    assert os.path.exists(p), "the decode-time blocklist arm is missing; guard cannot be vacuous"
    return {r["arm"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}


def test_the_rule_actually_bound_and_the_control_did_not():
    """Caution (p): a rule that never fires makes 'it suppresses copying' unfalsifiable."""
    rows = _decode_rows()
    assert int(rows["memfree"]["blocked_steps"]) > 0, rows["memfree"]
    assert int(rows["-1"]["blocked_steps"]) == 0, rows["-1"]


def test_the_incumbent_beats_us_on_leakage_reduction_from_its_own_baseline():
    """The leakage half of the measurement, kept as data rather than as prose.

    The BODY claim that the incumbent concedes is checked by
    test_the_incumbent_concession_is_in_the_body_and_is_the_TRUE_one, which replaced an earlier
    guard here that looked for "it wins" anywhere in two files. That guard was satisfied by the
    appendix's "selection anchoring wins NEITHER comparison" -- the opposite meaning -- and the
    sentence it was written for turned out to be false anyway (caution (an), then (v)).
    """
    rows = _decode_rows()
    assert float(rows["memfree"]["nv_recall_mean"]) < float(rows["-1"]["nv_recall_mean"]) / 10


def test_the_paraphrase_claim_is_withdrawn_because_it_was_measured_false():
    """H3 predicted >= 10 of 100 at ROUGE-L >= 0.5 under the rule and measured 0. The old sentence
    'silent on ... paraphrase' may not come back while that is true."""
    rows = _decode_rows()
    assert int(rows["memfree"]["rouge_ge_0p5_count"]) == 0, rows["memfree"]
    txt = body("appendix_related.tex", "related_work_v4.tex")
    assert "silent on an unlisted work, on paraphrase" not in txt, \
        "the refuted paraphrase claim is back in the manuscript"


def test_the_blocklist_is_a_no_op_on_ordinary_text_so_its_utility_number_is_not_quoted_as_its_own():
    """The rule fired on 0 of 850 ordinary prompts, so +0.272 measures the unconstrained model,
    not the blocklist. The paper must not present that gain as a property of the rule."""
    pat = os.path.join(ROOT, "results", "order_averaged_h2h__memfree.csv")
    nor = os.path.join(ROOT, "results", "order_averaged_h2h__norule.csv")
    assert os.path.exists(pat) and os.path.exists(nor)
    g = {}
    for tag, p in (("rule", pat), ("norule", nor)):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["quantity"].startswith("D4"):
                g[tag] = float(r["value"])
    assert g["rule"] == g["norule"], g       # byte-identical text, so identical gains
    body_txt = body("experiments.tex", "selection.tex", "iclr_closing.tex")
    assert "0.272" not in body_txt, \
        "a gain produced by a no-op reached the body as if it measured the blocklist"


def test_cpfuse_reproduced_before_its_contrast_was_stated():
    """H4: the composition contrast may only be made if H1 passed."""
    p = os.path.join(ROOT, "results", "cpfuse_audit_rebuild.csv")
    assert os.path.exists(p), "the CP-Fuse rebuild is missing"
    rows = [r for r in csv.DictReader(open(p, encoding="utf-8")) if r["mode"] == "single"]
    own = {(r["arm"], r["shard"]): float(r["nv_recall_mean"]) for r in rows}
    assert own[("a", "0")] > 0.50 and own[("a", "1")] < 0.10, own
    assert own[("b", "1")] > 0.50 and own[("b", "0")] < 0.10, own
    assert own[("cpfuse", "0")] < 0.10 and own[("cpfuse", "1")] < 0.10, own


def test_the_window_vacuity_table_matches_its_csv():
    """Caution (j): a paper number rounds from the CSV, once."""
    p = os.path.join(ROOT, "results", "window_vacuity.csv")
    rows = {r["event"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}
    txt = body("appendix_proofs.tex").replace("$", "")
    for ev, cells in (("10-token window", ("32.0", "0.160", "7.7")),
                      ("20-token window", ("63.9", "0.320", "15.4")),
                      ("50-token window", ("159.8", "0.799", "38.4")),
                      ("100-token window", ("319.7", "1.598", "76.9"))):
        r = rows[ev]
        assert abs(float(r["S_median_nats"]) - float(cells[0])) < 0.05, (ev, r)
        assert abs(float(r["k_at_vacuity"]) - float(cells[1])) < 5e-4, (ev, r)
        for c in cells:
            assert c in txt.replace("$", ""), f"{ev}: {c} is not in the appendix table"


def test_the_50_token_window_is_the_one_the_extraction_metric_uses():
    """The claim that makes the window threshold the RELEVANT one. If the metric ever stops being
    a 50-token window, the sentence must be revisited rather than left standing."""
    src = open(os.path.join(ROOT, "analysis", "window_vacuity.py"), encoding="utf-8").read()
    assert "50-token windows" in src
    # the manuscript writes $50$-token, so strip math delimiters before matching (caution (an):
    # a bare value carries no $ around it and a guard on the spelling misses the sentence)
    txt = body("appendix_proofs.tex", "frontier.tex").replace("$", "")
    assert "50-token window" in txt


def test_the_incumbent_concession_is_in_the_body_and_is_the_TRUE_one():
    """Two things at once, because the first draft of this concession was wrong in the harsher
    direction and that is still an error.

    TRUE:  on a listed work the blocklist costs NO UTILITY -- it fired on 0 of 850 ordinary
           prompts -- which selection cannot approach.
    FALSE: that it suppresses MORE. Under the same 20-token seed protocol on the same corpus,
           selection reads 0.0000 near-verbatim recall at every n <= 64 and the blocklist 0.0201.

    Both halves are derived from the CSVs, so the sentence cannot drift in either direction
    without the data moving.
    """
    dec = _decode_rows()
    sel = os.path.join(ROOT, "results", "selection_extraction.csv")
    rows = list(csv.DictReader(open(sel, encoding="utf-8")))
    sel_leak = max(float(r["nv_recall_mean"]) for r in rows
                   if r["n"].strip() and r["n"] != "-1")
    blk_leak = float(dec["memfree"]["nv_recall_mean"])
    # the claim the paper must NOT make
    assert sel_leak <= blk_leak, (
        f"selection now leaks MORE than the blocklist ({sel_leak} vs {blk_leak}); the appendix "
        "sentence saying it does not must be revisited")
    # the claim the paper MUST make, in the body. v9: "We measure one as a decoder, and on ordinary
    # text it costs nothing"; v10 (2026-09-24): "Measured as decoders on listed works ..., a
    # blocklist costs no utility". Same concession, reworded.
    rw = body("related_work_v4.tex")
    i = rw.find("Measured as decoders")
    assert i >= 0, "Related Work no longer says the incumbent was measured"
    assert "a blocklist costs no utility" in rw[i:i + 200], rw[i:i + 200]
    # and the appendix must carry the precise form, not the harsher one: no utility cost on a listed
    # work AND no stronger suppression, with the blocklist's own recall quoted from the CSV
    apx = body("appendix_related.tex")
    assert "wins neither the utility comparison nor the leakage one" not in apx, \
        "the appendix is back to the overcorrected claim, which is false on leakage"
    assert "the blocklist costs no utility" in apx, apx[:200]
    assert f"does not suppress more (${blk_leak:.4f}$ against its own" in apx, blk_leak


def test_the_judgefree_compute_claim_matches_its_whole_grid():
    """Caution (ai): a claim ABOUT a set of numbers is checked against nothing unless a test
    rebuilds the set. The appendix says 'no n on the grid closes the gap' and 'at 11.5x the
    compute selection is still at 0.190', which are statements about every row, not one cell."""
    p = os.path.join(ROOT, "results", "compute_matched_judgefree.csv")
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    assert len(rows) >= 6, rows
    met = float(rows[0]["best_metered_acc"])
    assert all(float(r["best_metered_acc"]) == met for r in rows), "the meter's best moved per row"
    # the claim: no selection arm reaches the meter's best, at any cost on the grid
    assert all(float(r["selection_acc"]) < met for r in rows), \
        "a selection arm now matches the meter; 'no n on the grid closes the gap' must be revisited"
    # and the top of the grid is the one the appendix quotes
    top = max(rows, key=lambda r: float(r["cost_vs_metered"]))
    apx = body("appendix_selection.tex").replace("$", "")
    assert f"{float(top['cost_vs_metered']):.1f}" in apx, (top["cost_vs_metered"], "not quoted")
    assert f"{float(top['selection_acc']):.3f}" in apx, (top["selection_acc"], "not quoted")
    # the certificate ratio the paper calls 115x
    assert abs(float(top["certificate_ratio"]) - 115.4) < 0.1, top


def test_the_meters_cost_really_is_flat_in_k_which_is_why_the_comparison_works():
    """The whole comparison rests on it: the meter runs both models at every step whatever k is,
    so its entire accuracy range is available at one serving cost."""
    src = open(os.path.join(ROOT, "analysis", "compute_matched_judgefree.py"),
               encoding="utf-8").read()
    assert "P_ANCHOR + P_RISKY" in src
    assert 'r["arm"] != "k=-1"' in src, \
        "the unconstrained baseline is back in the metered set; it would credit the meter for free"
