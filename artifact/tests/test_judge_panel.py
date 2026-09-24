"""The five-judge panel, and the claims the manuscript makes from it.

Caution (aq): a suite that was green before an edit and green after it has told you only that the
edit changed nothing the OLD edits guarded. These guards are written against the edit --- the panel
result and the three sentences it put in the paper --- and each was mutation-checked when written.

Caution (aq) again, rule 1: a withdrawal branch conditions on MEASURED DATA, never on a phrase in
another section. Every guard here reads the five committed CSVs and derives what the paper may say,
so rewording a sentence cannot retire a check and deleting a claim cannot pass one.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import body, tex  # noqa: E402


def section_text(name):
    """Whitespace-normalised text of one live section file, or of the main .tex."""
    if name == "iclr_2027":
        return " ".join(open(tex("iclr_2027.tex"), encoding="utf-8").read().split())
    return body(name + ".tex")

TAGS = {
    "": ("B", "Phi-3.5-mini-instruct", 3.8, "clean"),
    "__qwen72b": ("D", "Qwen2.5-72B-Instruct", 72.0, "scorer"),
    "__mixtral": ("E", "Mixtral-8x7B-Instruct-v0.1", 47.0, "clean"),
    "__qwen14b": ("F", "Qwen2.5-14B-Instruct", 14.0, "scorer"),
    "__gemma27b": ("G", "gemma-2-27b-it", 27.0, "clean"),
}


def _panel():
    """(letter, size, family, d3, lo, hi) for every judge that has a committed CSV."""
    out = []
    for tag, (letter, _model, size, fam) in TAGS.items():
        p = f"results/order_averaged_h2h{tag}.csv"
        if not os.path.exists(p):
            raise AssertionError(f"{p} missing; this guard must not pass by never running")
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        d3 = [r for r in rows if r["quantity"].startswith("D3")]
        assert len(d3) == 1, p
        r = d3[0]
        out.append((letter, size, fam, float(r["value"]), float(r["lo95"]), float(r["hi95"])))
    return out


def _resolves(row):
    return row[3] > 0 and row[4] > 0


def test_the_panel_is_five_judges_and_four_of_them_resolve_the_difference():
    panel = _panel()
    assert len(panel) == 5, [p[0] for p in panel]
    r = sum(1 for p in panel if _resolves(p))
    assert r == 4, [(p[0], p[3], p[4]) for p in panel]
    # every judge puts it positive -- the weaker claim the paper also makes
    assert all(p[3] > 0 for p in panel), panel


# The panel the v10 manuscript quotes (2026-09-24). Caution (bc): every judged number recorded before
# 2026-09-24 was judged on echo-carrying text, and the paper now reports the head-to-head on the
# RECOVERED text, re-judged by six judges -- B and the five that re-score the headline's texts, C
# (the opponent's own checkpoint) included (Figure fig:h2h). The recorded-text five-judge panel above
# is still what Table tab:h2hrepeat prints, and the guards above still pin it.
REPAIRED = {
    "": ("B", "Phi-3.5-mini-instruct"),
    "__opp_committed_judgeC": ("C", "Meta-Llama-3.1-8B-Instruct"),
    "__qwen72b": ("D", "Qwen2.5-72B-Instruct"),
    "__mixtral": ("E", "Mixtral-8x7B"),
    "__qwen14b": ("F", "Qwen2.5-14B-Instruct"),
    "__gemma27b": ("G", "gemma-2-27b-it"),
}


def _panel_repaired():
    """(letter, model, d3, lo, hi) on the recovered text, one row per judge of the v10 panel."""
    out = []
    for tag, (letter, model) in REPAIRED.items():
        p = f"results/order_averaged_h2h{tag}_deecho.csv"
        if not os.path.exists(p):
            raise AssertionError(f"{p} missing; this guard must not pass by never running")
        d3 = [r for r in csv.DictReader(open(p, encoding="utf-8")) if r["quantity"].startswith("D3")]
        assert len(d3) == 1, p
        r = d3[0]
        out.append((letter, model, float(r["value"]), float(r["lo95"]), float(r["hi95"])))
    return out


def test_the_paper_quotes_the_measured_fraction_and_not_an_unqualified_claim():
    """H3's r>=4 branch: 'The claim does not return to an unqualified judged better.'

    v10 (2026-09-24) quotes the fraction on the recovered text over a six-judge panel ("under five of
    six judges" in the abstract, "five of six judges exclude zero, and the sixth, Mixtral-8x7B,
    straddles it at $-0.0015$" in Section 4.2), where v9 quoted four of five on the recorded text. The
    fraction is derived here from the six committed *_deecho CSVs, so it cannot drift from them."""
    panel = _panel_repaired()
    r = sum(1 for p in panel if p[2] > 0 and p[3] > 0)
    words = {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
    claim = f"{words[r]} of {words[len(panel)]} judges"
    for f in ("iclr_2027", "experiments"):
        txt = section_text(f)
        if f == "iclr_2027":
            assert claim in txt, f"the abstract must carry the measured fraction '{claim}'"
        else:
            # GUARD THE PROPERTY, NOT THE SPELLING (caution (an)): the section must state the
            # fraction, name every judge that does not resolve it with its own reading, and make no
            # 'judged better' claim without the fraction beside it.
            low = txt.lower()
            assert claim in low, f"Section 4.2 must state the measured fraction ({claim})"
            for letter, model, d3, lo, hi in panel:
                if not (d3 > 0 and lo > 0):
                    assert model.lower() in low and f"${d3:+.4f}$" in txt, \
                        (letter, model, d3, "the judge that does not resolve it is not named")
            assert "judged better" not in low or claim in low, \
                "Section 4.2 makes an unqualified 'judged better' claim"


def test_family_does_not_explain_the_exception_and_the_paper_says_so():
    """The registered H2 band. Derived from the CSVs, so a reworded sentence cannot retire it."""
    panel = _panel()
    fam = [p for p in panel if p[2] == "scorer"]
    clean = [p for p in panel if p[2] == "clean"]
    assert len(fam) == 2 and len(clean) == 3, panel
    fam_r = sum(1 for p in fam if _resolves(p))
    clean_r = sum(1 for p in clean if _resolves(p))
    # FAMILY EXPLAINS IT would need all of fam and none of clean
    assert not (fam_r == len(fam) and clean_r == 0), \
        "family now explains the split; the manuscript's claim that it does not must be revisited"
    # and the single non-resolving judge is family-CLEAN, which is the paper's sentence
    nonres = [p for p in panel if not _resolves(p)]
    assert len(nonres) == 1 and nonres[0][2] == "clean", nonres
    txt = section_text("appendix_selection")
    assert "clean" in txt and "family" in txt.lower(), \
        "the appendix must state that the exception is family-clean"


def test_no_covariate_we_measured_orders_the_outcome():
    """Size does not separate resolvers from the exception -- the paper's explicit claim."""
    panel = _panel()
    res = sorted(p[1] for p in panel if _resolves(p))
    non = sorted(p[1] for p in panel if not _resolves(p))
    assert non, "nothing to explain; revisit the sentence"
    # a size THRESHOLD would separate them; assert none does
    assert not (max(res) < min(non) or min(res) > max(non)), \
        f"size now separates resolvers {res} from non-resolvers {non}; the claim must be revisited"


def test_levels_are_never_quoted_across_judges():
    """Caution (ap). The levels move by a factor of three, which is why only D3 is compared."""
    mets = []
    for tag in TAGS:
        rows = list(csv.DictReader(open(f"results/order_averaged_h2h{tag}.csv", encoding="utf-8")))
        mets.append(float([r for r in rows if r["quantity"].startswith("D2")][0]["value"]))
    assert max(mets) / min(mets) > 2.5, mets      # the fact the paper asserts
    body = section_text("experiments")
    # the two judge-D/E/F/G LEVELS must not appear in the body; only the difference may
    for lvl in ("0.1950", "0.1330", "0.1680", "0.1975", "0.1185"):
        assert lvl not in body, f"a panel LEVEL ({lvl}) reached the body; only D3 may be quoted"


def test_the_exception_is_named_so_it_cannot_be_aggregated_away():
    txt = section_text("experiments") + section_text("appendix_selection")
    assert "Mixtral" in txt, "the judge that does not resolve must be named in the paper"
