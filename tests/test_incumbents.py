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


def test_the_paper_concedes_the_incumbent_wins_on_a_listed_work():
    """Measured: recall 0.4192 -> 0.0201 and 0/100 at ROUGE-L >= 0.5. The paper must not imply
    selection leaks less on a work the blocklist was given."""
    rows = _decode_rows()
    assert float(rows["memfree"]["nv_recall_mean"]) < float(rows["-1"]["nv_recall_mean"]) / 10
    # Scoped to the Scope paragraph's own sentence. An unscoped search for "wins" is satisfied by
    # the appendix's "selection anchoring wins NEITHER comparison" -- the opposite meaning, in
    # another file (caution (an)); mutation-testing found exactly that.
    rw = body("related_work_v4.tex")
    i = rw.find("We measure one")
    assert i >= 0, "the Scope paragraph no longer says the incumbent was measured"
    sentence = rw[i:i + 220]
    assert ("it wins" in sentence or "comes out ahead" in sentence), \
        f"the Scope paragraph must concede the incumbent beat us: {sentence[:160]!r}"


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


def test_limitations_carries_the_incumbent_concession():
    """Caution (ag): a length edit deletes concessions first. This one is in the Conclusion's
    Limitations, which is where a careful reader looks, and it is derived from the measurement --
    the rule's recall reduction -- so softening the sentence needs the CSV to change too."""
    rows = _decode_rows()
    beat_on_leakage = (float(rows["memfree"]["nv_recall_mean"])
                       < float(rows["-1"]["nv_recall_mean"]))
    assert beat_on_leakage, rows
    txt = body("iclr_closing.tex")
    i = txt.find("blocklist")
    assert i >= 0, "Limitations no longer concedes that a blocklist beats both mechanisms"
    assert "beats both" in txt[i - 120:i + 120], txt[max(0, i - 150):i + 150]
