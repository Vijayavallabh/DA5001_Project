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


def test_the_paper_concedes_it_did_not_measure_them_and_says_why():
    """Derived from the fact itself: no results/ artefact exists for either method, so the paper
    must not imply one does, and must carry the concession."""
    import glob
    from tests.manuscript import ROOT
    assert not glob.glob(os.path.join(ROOT, "results", "*tokenswap*")), (
        "a TokenSwap measurement now exists; this guard and the paragraph must be revisited")
    assert not glob.glob(os.path.join(ROOT, "results", "*trbs*")), (
        "a TRBS measurement now exists; this guard and the paragraph must be revisited")
    t = body(SEC)
    i = t.find("TokenSwap")
    assert i > 0
    w = t[i:i + 2200]
    assert "did not measure either one" in w, "the concession that neither was measured is gone"
    assert "next comparison" in w, "the paper no longer says measuring them is the next step"


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
    i = t.find("TokenSwap")
    w = t[i:i + 2200]
    assert "served law" in w, "the level distinction has gone from the paragraph"
    assert re.search(r"neither is in this paper's class|not in this paper's class", w), (
        "the paragraph no longer says these are outside the certified class")
