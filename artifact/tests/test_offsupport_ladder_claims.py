"""feat-172's consequences in the manuscript, each guarded against results/offsupport_ladder.csv.

The registered reading is STILL CLIMBING on both workloads, at about one half-width; measured from
n=64 the two doublings contain zero (post hoc). So the appendix withdrew "ceiling" and "optimum" and
put NO slope in their place. Every guard here is conditioned on the CSV, never on a phrase in
another section, so a withdrawal has to survive the data rather than a rewording (caution (aq)).
"""
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import DIR, body, carries_band  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROWS = list(csv.DictReader(open(os.path.join(ROOT, "results", "offsupport_ladder.csv"), encoding="utf-8")))


def _row(arm, quantity):
    got = [r for r in ROWS if r["arm"] == arm and r["quantity"] == quantity]
    assert len(got) == 1, (arm, quantity, got)
    r = got[0]
    return r, *(float(r[c]) if r[c] else None for c in ("value", "lo95", "hi95"))


def _para(start):
    """The paragraph of appendix_selection.tex that holds `start`, whitespace-normalised."""
    txt = body("appendix_selection.tex")
    i = txt.index(start)
    j = txt.find("\\paragraph{", i + 1)
    return txt[txt.rfind("\\paragraph{", 0, i + 1):j if j > 0 else None]


# v10 (2026-09-24) retitled the paragraph ("How far $n$ pays.") and moved its bands into a table;
# it is located by the reference to that table, so a retitle cannot retire the guards.
LADDER = "Table~\\ref{tab:ladder}"


def test_every_ladder_band_the_appendix_quotes_rounds_from_the_csv():
    for arm, q in (("A", "g(128)-g(64)"), ("A", "g(256)-g(128)"), ("B", "g(128)-g(64)"),
                   ("B", "g(256)-g(128)"), ("A", "g(256)-g(64)"), ("B", "g(256)-g(64)"),
                   ("A", "D3 n=128"), ("A", "D3 n=256")):
        _, g, lo, hi = _row(arm, q)
        assert carries_band(g, lo, hi, "appendix_selection.tex"), (arm, q, "band not printed")
    _, g, lo, hi = _row("A", "g(256)-g(64) order-averaged")
    assert carries_band(g, lo, hi, "appendix_selection.tex"), "order-averaged total not printed"


def test_the_ceiling_and_the_optimum_stay_withdrawn_while_the_next_doubling_climbs():
    climbs = [_row(a, "g(256)-g(128)")[0]["reading"] == "STILL CLIMBING" for a in ("A", "B")]
    if not any(climbs):
        return  # a re-run that saturates would license the old wording again; revisit by hand
    live = [f for f in glob.glob(os.path.join(DIR, "sections", "*.tex"))
            if "_v" not in os.path.basename(f) and "preflow" not in f]
    for f in live:
        txt = " ".join(open(f, encoding="utf-8").read().split())
        for phrase in ("ceiling between $64$ and $128$", "ceiling is between $64$ and $128$",
                       "has an optimum beyond which", "anchor-dependent} ceiling"):
            assert phrase not in txt, (os.path.basename(f), phrase)


def test_no_slope_is_claimed_where_the_two_doublings_contain_zero():
    totals = [_row(a, q) for a, q in (("A", "g(256)-g(64)"), ("B", "g(256)-g(64)"),
                                      ("A", "g(256)-g(64) order-averaged"))]
    para = _para(LADDER)
    if all(lo < 0 < hi for _, _, lo, hi in totals):
        assert "we claim neither a ceiling nor a slope" in para.lower(), "a slope crept back in"
        assert "post hoc" in para, "the two-doubling checks are no longer labelled post hoc"
    for a in ("A", "B"):
        _, g, lo, hi = _row(a, "g(256)-g(128)")
        if g / ((hi - lo) / 2) < 1.5:
            assert "about one half-width from zero" in para, (a, "the marginality concession left")


def test_the_reversal_is_said_to_fail_off_support_at_256_only_while_d3_is_clear_of_zero():
    _, g, lo, hi = _row("A", "D3 n=256")
    txt = body("appendix_selection.tex")
    said = "the reversal fails off-support at $256$ as at $64$" in txt
    assert said == (hi < 0), (g, lo, hi, said)
    cert = _row("A", "certificate log 256")[0]
    assert f"$\\log 256 = {float(cert['value']):.2f}$" in txt
    assert f"realised ${float(cert['reading']):.1f}$" in txt


def test_limitations_names_where_the_scorer_does_turn_over():
    txt = body("appendix_limitations.tex")
    assert "Our extraction arm reaches $n=64$ and our judged arm $n=8$" not in txt, "stale reach"
    over = [r for r in csv.DictReader(open(os.path.join(ROOT, "results", "cotaeval_scoring.csv"),
                                           encoding="utf-8"))
            if r.get("reading", "").strip() == "TURNS OVER" or r[list(r)[-1]] == "TURNS OVER"]
    if over:
        i = txt.index("\\emph{Overoptimisation.}")
        para = txt[i:txt.index("\\emph{", i + 10)]
        assert "TriviaQA" in para and "CoTaEval" in para, "a measured turnover left Limitations"
