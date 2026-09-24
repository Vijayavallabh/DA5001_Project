"""The contaminated-anchor amplification range is a min/max over a set -- caution (ai)'s class.

"across twelve deliberately contaminated anchors the realised amplification at n = 64 runs between
1.0 and 4.0, against the 64 the certificate permits" is three claims about a set: the anchor count,
the endpoints, and that every value is under the certificate. Rebuild all three from the CSV.

This one IS a realisation, unlike the selection bounds guarded in test_selection_claims.py:
analysis/contaminated_anchor.py computes rate / base_rate for a named event. The column is empty for
a clean anchor because the anchor's own base rate is 0 there, which is why the clean arms have a
bound and no measured amplification.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import body, caption_of, tex  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rows():
    """feat-179's corrected-selector arm. contaminated_anchor.csv was measured with a selector that
    ranked partly on padding (caution (ba)); its n > 1 numbers are no longer quoted anywhere."""
    p = os.path.join(ROOT, "results", "selector_n256.csv")
    return list(csv.DictReader(open(p, encoding="utf-8")))


def _measured_at_64():
    return [r for r in _rows() if int(float(r["n"])) == 64 and r["amplification_vs_n1"] not in ("", None)]


def test_twelve_distinct_contaminated_anchors():
    assert len({r["anchor"] for r in _rows()}) == 12, sorted({r["anchor"] for r in _rows()})


def _ethics():
    main = " ".join(open(tex("iclr_2027.tex"), encoding="utf-8").read().split())
    main = main[main.index(r"\section*{Ethics Statement}"):]
    return main[:main.index(r"\section*{", 1)]


def _leakage_section():
    """Section 4.4's prose, without the fig:safety float (its caption is a site of its own)."""
    exp = body("experiments.tex")
    sec = exp[exp.index("\\label{sec:leakage}"):exp.index("\\subsection{", exp.index("\\label{sec:leakage}"))]
    a, b = sec.index("\\begin{figure}"), sec.index("\\end{figure}")
    return sec[:a] + sec[b:]


def _contaminated_paragraph():
    apx = body("appendix_selection.tex")
    sec = apx[apx.index("\\label{app:contaminated}"):]
    return sec[:sec.index("\\paragraph{")]


def test_the_quoted_endpoints_are_the_actual_min_and_max():
    """v10 (2026-09-24): the introduction and Section 2 no longer quote the range; it lives in
    Section 4.4, the Ethics Statement and Appendix E, and each must quote the CSV's own min and max
    as a range."""
    vals = [float(r["amplification_vs_n1"]) for r in _measured_at_64()]
    assert vals, "no measured amplification at n=64"
    lo, hi = min(vals), max(vals)
    for f, txt in (("Section 4.4", _leakage_section()), ("Ethics Statement", _ethics()),
                   ("app:contaminated", _contaminated_paragraph())):
        assert _quotes(txt, lo, hi), (f, lo, hi)


def test_every_measured_amplification_is_under_the_certificate():
    """The claim is 'against the 64 the certificate permits' -- so none may exceed 64."""
    for r in _rows():
        if r["amplification_vs_n1"]:
            assert float(r["amplification_vs_n1"]) <= int(r["n"]) + 1e-9, r


def _range(fname, event="E_08"):
    vals = [float(r["amplification_vs_n1"]) for r in
            csv.DictReader(open(os.path.join(ROOT, "results", fname), encoding="utf-8"))
            if r["n"] == "64" and r["event"] == event and r["amplification_vs_n1"]]
    return min(vals), max(vals)


def _quotes(txt, lo, hi):
    return any(f"${lo:.1f}$ {w} ${hi:.1f}" in txt for w in ("to", "and"))


def test_the_intro_quotes_the_range_and_the_anchor_count():
    """v10 (2026-09-24): the introduction no longer carries this sentence. Its count and both draws'
    ranges now sit together in the Ethics Statement, and in Section 4.4 (count in the fig:safety
    caption, both ranges in the prose); both sites are checked, the count rebuilt from the CSV."""
    count = {12: "twelve"}[len({r["anchor"] for r in _rows()})]
    eth = _ethics()
    assert f"{count} deliberately contaminated anchors" in eth, "the anchor count was trimmed"
    assert f"{count} anchors fine-tuned on the passages" in caption_of("fig:safety").lower(), \
        "Section 4.4 no longer states how many contaminated anchors there are"
    for f in ("selector_n256.csv", "selector_redraw.csv"):
        assert _quotes(eth, *_range(f)), (f, "the Ethics Statement no longer quotes this draw's range")
        assert _quotes(_leakage_section(), *_range(f)), (f, "Section 4.4 no longer quotes this draw's range")


def test_every_site_quotes_the_factor_per_draw_both_draws():
    """feat-182 (results/onset_prediction_selector_redraw.md): 'the amplification quoted at n = 64 and
    n = 256 is given per draw, both draws, never averaged into one rate', and a verdict that does not
    replicate is stated as draw-dependent. Every site that quotes the first draw must quote the second."""
    # v10 (2026-09-24): the sites are Section 4.4's prose, the fig:safety caption (which moved into
    # Section 4.4), the Ethics Statement and Appendix E's contaminated-anchor paragraph; the intro
    # and Section 2 no longer quote the factor. The rule itself is checked on EVERY live file too:
    # any file that quotes the first draw must quote the second beside it.
    sites = {"Section 4.4": _leakage_section(), "ethics": _ethics(),
             "fig:safety caption": caption_of("fig:safety"),
             "app:contaminated": _contaminated_paragraph()}
    one, two = _range("selector_n256.csv"), _range("selector_redraw.csv")
    for name, txt in sites.items():
        assert _quotes(txt, *one), (name, "first draw not quoted")
        assert _quotes(txt, *two), (name, "the second draw must be quoted beside the first")
    import re
    main = open(tex("iclr_2027.tex"), encoding="utf-8").read()
    swept = {"abstract": " ".join(main.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0].split())}
    swept.update({n: body(n + ".tex") for n in re.findall(r"^\\input\{sections/([^}]+)\}", main, re.M)})
    assert len(swept) > 10, sorted(swept)
    for name, txt in swept.items():
        if _quotes(txt, *one):
            assert _quotes(txt, *two), (name, "quotes the first draw without the second")
    rep = {r["part_a_band"]: r["reading"] for r in
           csv.DictReader(open(os.path.join(ROOT, "results", "selector_redraw_replication.csv"),
                               encoding="utf-8"))}
    if rep["B1"] == "DOES NOT REPLICATE":
        # v10 words the draw-dependent verdict without the label: the band is crossed on one draw
        # and not the other, "so we give both readings".
        assert "crossed by one anchor on the first draw and by none on the second" in sites["app:contaminated"]
        assert "we give both readings" in sites["app:contaminated"]
