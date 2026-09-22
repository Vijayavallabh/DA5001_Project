"""The two inference-time baselines the Program Chairs named, and the concession beside them.

The PC report asked for a comparison with, or explicit discussion of, MemFree Decoding
(Ippolito et al. 2023), TokenSwap (Prashant et al. 2025) and TRBS (Fu et al. 2026). The first was
already measured as a decoder (Appendix J). The other two were absent from the manuscript AND from
references.bib until 2026-09-20.

BOTH CITATIONS WERE VERIFIED TO EXIST BEFORE BEING ADDED. The report is LLM-generated and a
fabricated reference is a desk-reject under the CFP, so neither was written from the report's own
wording: TokenSwap is arXiv:2502.05159 (NeurIPS 2025) and TRBS is Findings of ACL 2026. This test
pins the venue and year so a later edit cannot quietly turn a checked citation into a guessed one.

The paragraph concedes that neither was measured and says why. That is a sentence arguing against
this paper, which caution (ag) says a page-budget trim deletes first, so it is guarded per claim.
"""
import os
import re

from tests.manuscript import DIR, body

SEC = "appendix_related.tex"


def bib():
    return open(os.path.join(DIR, "references.bib"), encoding="utf-8").read()


def test_both_named_baselines_are_cited_in_a_live_section():
    t = body(SEC)
    for key in ("prashant2025tokenswap", "fu2026duplicate"):
        assert f"\\citep{{{key}}}" in t or f"\\citet{{{key}}}" in t, (
            f"{key} is not cited in {SEC}; the PC report named it by name")
    for name in ("TokenSwap", "TRBS"):
        assert name in t, f"the paper does not name {name}, which is how the report asked for it"


def test_the_bibliography_entries_are_the_verified_ones():
    """Venue and year as checked against the published record, not as the report phrased them."""
    b = bib()
    m = re.search(r"@inproceedings\{prashant2025tokenswap,(.*?)\n\}", b, re.S)
    assert m, "TokenSwap entry missing"
    assert "2502.05159" in m.group(1), "the verified arXiv id is gone"
    assert "2025" in m.group(1) and "NeurIPS" in m.group(1)
    m = re.search(r"@inproceedings\{fu2026duplicate,(.*?)\n\}", b, re.S)
    assert m, "TRBS entry missing"
    assert "2026" in m.group(1) and "Association for Computational Linguistics" in m.group(1)


def test_the_paper_reports_the_tokenswap_measurement_and_that_it_loses():
    """This guard used to assert the OPPOSITE -- that no results/ artefact existed and the paper
    said so. feat-165/167/169 measured TokenSwap and the guard fired, which is what it was for.

    What it pins now is the uncomfortable half. TokenSwap beats this paper's own mechanism on
    utility and matches it on suppression, so these are the sentences a length edit deletes first
    (caution (ag)), and every number is rebuilt from the CSV that produced it.
    """
    import csv as _csv
    import glob
    from tests.manuscript import ROOT
    assert glob.glob(os.path.join(ROOT, "results", "*tokenswap*")), \
        "the TokenSwap measurement artefacts are gone; the paragraph claims a measurement we lack"
    assert not glob.glob(os.path.join(ROOT, "results", "*trbs*")), \
        "a TRBS measurement now exists; the paragraph still says it is unmeasured"

    def d5(tag):
        rows = [r for r in _csv.reader(open(os.path.join(
            ROOT, "results", f"order_averaged_h2h__{tag}.csv"), encoding="utf-8"))
            if r and r[0].startswith("D5")]
        assert len(rows) == 1, tag
        return float(rows[0][2]), float(rows[0][3]), float(rows[0][4]), rows[0][7].strip()

    t = body(SEC)
    # The paragraph, not the first mention: since 2026-09-23 Appendix J opens with a table that
    # names TokenSwap first, and a window from there never reached the concession (caution (an)).
    i = t.find(r"\textbf{TokenSwap}") + len(r"\textbf{")
    assert i > len(r"\textbf{")
    w = t[i:i + 4400]  # the paragraph grew when the measurement replaced the concession

    # It must say the measurement happened, and that our mechanism is the expensive one.
    assert "have now measured TokenSwap" in w, "the paper no longer says the arm was run"
    assert "most expensive of the three" in w, \
        "the concession that our mechanism costs the most was trimmed"

    # The three costs, each from its own judging pass, and the head-to-head with its verdict.
    ts, tslo, tshi, verdict = d5("tokenswap")
    assert verdict == "INCUMBENT WINS", f"TokenSwap's verdict is now {verdict!r}"
    assert ts > 0, "TokenSwap no longer beats selection; the paragraph's framing is stale"
    assert f"${ts:+.4f}$ $[{tslo:+.4f}, {tshi:+.4f}]$" in w, "the head-to-head band left the paper"
    sel, sello, selhi, _ = d5("norule")
    assert f"$-{sel:.4f}$ $[-{selhi:.4f}, -{sello:.4f}]$" in w, \
        "selection's own cost against the shared control left the paper"

    # The auxiliary qualification: at THEIR auxiliary it is a tie, not a win.
    dg, dglo, dghi, dgv = d5("ts_distilgpt2")
    assert dgv == "TIE", f"the DistilGPT-2 reading is now {dgv!r}"
    assert f"${dg:+.4f}$ $[{dglo:+.4f}, {dghi:+.4f}]$" in w, \
        "the tie at their own auxiliary was trimmed; it is what makes the +0.0615 honest"

    # The fragility, and that |G| is not the axis.
    kl = [r for r in _csv.DictReader(open(os.path.join(
        ROOT, "results", "blocklist_decode__tsleak_kl3m170m.csv"), encoding="utf-8"))
        if r["arm"] == "tokenswap"][0]
    assert f"${float(kl['nv_recall_mean']):.4f}$" in w, "the KL3M leak was trimmed"
    assert f"${kl['rouge_ge_0p5_count']} of $100$" in w or \
        f"${kl['rouge_ge_0p5_count']}$ of $100$" in w, "the KL3M passage count was trimmed"
    assert "is not the axis" in w, "the paper no longer says |G| fails to predict the failure"
    assert "vetting requirement" in w, \
        "the paper no longer records that its own stated blocker was discharged"


def test_the_scope_limit_on_trbs_is_kept():
    """TRBS is evaluated on code. Dropping that would let a reader read it as a prose baseline we
    declined to run on equal terms."""
    t = body(SEC)
    i = t.find("TRBS")
    assert i > 0
    # SCOPED TO THE SENTENCE that introduces TRBS. A window check passed when that sentence was
    # reworded to "evaluated broadly", because the word "code" occurs again further down in the
    # clause explaining why we did not port it -- caution (an), a guard satisfied by a different
    # occurrence of its own word.
    sent = t[i:]
    end = sent.find(". ")
    sent = sent[:end if end > 0 else 400]
    assert "code" in sent, (
        f"the sentence introducing TRBS no longer says what it was evaluated on: {sent[:160]!r}")


def test_neither_is_claimed_to_be_in_this_papers_class():
    """The whole point of the paragraph is the level: these bound a named catalogue, not the served
    law. An edit that blurred that would make the paper's own axis look arbitrary."""
    t = body(SEC)
    i = t.find(r"\textbf{TokenSwap}") + len(r"\textbf{")   # the paragraph, not the table's row
    assert i > len(r"\textbf{")
    w = t[i:i + 2200]
    m = re.search(r"neither is in this paper's class|not in this paper's class", w)
    assert m, "the paragraph no longer says these are outside the certified class"
    # the distinction must be IN that sentence: the paragraph's later "a bound on the served law"
    # satisfied a paragraph-wide check with the distinction deleted (caution (an), 2026-09-23)
    assert "served law" in w[m.start(): w.index(". ", m.start())], \
        "the level distinction has gone from the paragraph"
