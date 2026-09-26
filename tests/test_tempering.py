"""feat-216 (results/onset_prediction_tempering.md): tempering the anchor against selecting. The table and the
closing sentence are pinned to the two judge passes, and the closing's wording to the readings, so a re-run that
moved a difference to zero could not leave "buys only part of the gain" standing (caution (av))."""
import csv
import os

from tests.manuscript import body

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pass(jd):
    return {(r["quantity"], r["arm"]): r for r in
            csv.DictReader(open(os.path.join(ROOT, "results", f"matched_h2h_tempering_{jd}.csv"), encoding="utf-8"))}


def _iv(r):
    f4 = lambda x: f"{float(x):+.4f}"
    return f"${f4(r['value'])}$ $[{f4(r['lo95'])}, {f4(r['hi95'])}]$"


def test_the_tempering_table_is_the_two_passes():
    B, G = _pass("B"), _pass("G")
    app = body("appendix_selection.tex")
    t = app[app.index("\\label{tab:tempering}"): app.index("\\end{tabular}", app.index("\\label{tab:tempering}"))]
    for n in ("sel_n64", "anchor_t07", "anchor_t05", "anchor_t07pen", "anchor_k0"):
        assert f"${B[('level', n)]['n_empty']}$ & {_iv(B[('gain', f'{n} - sel_n1')])} & {_iv(G[('gain', f'{n} - sel_n1')])}" in t, n
    for n in ("anchor_t07", "anchor_t05", "anchor_t07pen"):
        assert f"{_iv(B[('difference', f'sel_n64 - {n}')])} & {_iv(G[('difference', f'sel_n64 - {n}')])}" in t, n


def test_the_closing_quotes_the_registered_differences_and_its_word_follows_them():
    B, G = _pass("B"), _pass("G")
    c = body("iclr_closing.tex")
    d = {(j, n): P[("difference", f"sel_n64 - anchor_{n}")] for j, P in (("B", B), ("G", G)) for n in ("t07", "t05")}
    assert all(r["reading"] == "CONFIRMED" for r in d.values()), "a difference is no longer CONFIRMED: rewrite the sentence"
    f = lambda r: f"{float(r['value']):+.4f}".rstrip("0")
    s = (f"best-of-$64$ leads it at $T=0.7$ and $0.5$ by ${f(d[('B', 't07')])}$ and ${f(d[('B', 't05')])}$ (B), "
         f"${f(d[('G', 't07')])}$ and ${f(d[('G', 't05')]).ljust(6, '0')}$ (G; Appendix~\\ref{{app:tempering}})")
    assert "Tempering buys only part of the gain: " + s in c, s
    # the null (a second untempered draw) must stay unresolved under both judges, or the control is broken
    for P in (B, G):
        assert P[("gain", "anchor_k0 - sel_n1")]["reading"] == "UNRESOLVED"
