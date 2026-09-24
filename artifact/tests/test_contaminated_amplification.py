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
from tests.manuscript import body  # noqa: E402

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


def test_the_quoted_endpoints_are_the_actual_min_and_max():
    vals = [float(r["amplification_vs_n1"]) for r in _measured_at_64()]
    assert vals, "no measured amplification at n=64"
    lo, hi = min(vals), max(vals)
    for f in ("iclr_intro.tex", "selection.tex", "experiments.tex"):
        txt = body(f)
        assert f"${lo:.1f}$" in txt and f"${hi:.1f}$" in txt, (f, lo, hi)


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
    txt = body("iclr_intro.tex")
    assert "twelve deliberately contaminated anchors" in txt, "the anchor count was trimmed"
    for f in ("selector_n256.csv", "selector_redraw.csv"):
        assert _quotes(txt, *_range(f)), (f, "the intro no longer quotes this draw's range")


def test_every_site_quotes_the_factor_per_draw_both_draws():
    """feat-182 (results/onset_prediction_selector_redraw.md): 'the amplification quoted at n = 64 and
    n = 256 is given per draw, both draws, never averaged into one rate', and a verdict that does not
    replicate is stated as draw-dependent. Every site that quotes the first draw must quote the second."""
    import tests.manuscript as ms
    eth = " ".join(open(ms.tex("iclr_2027.tex"), encoding="utf-8").read().split())
    eth = eth[eth.index(r"\section*{Ethics Statement}"):]
    apx = body("appendix_selection.tex")
    cap = apx[apx.index("\\caption{"):apx.index("\\label{fig:safety}")]
    sec = apx[apx.index("\\label{app:contaminated}"):]
    sites = {"intro": body("iclr_intro.tex"), "selection": body("selection.tex"),
             "experiments": body("experiments.tex"), "ethics": eth, "fig:safety caption": cap,
             "app:contaminated": sec[:sec.index("\\subsection{")]}
    one, two = _range("selector_n256.csv"), _range("selector_redraw.csv")
    for name, txt in sites.items():
        assert _quotes(txt, *one), (name, "first draw not quoted")
        assert _quotes(txt, *two), (name, "the second draw must be quoted beside the first")
    rep = {r["part_a_band"]: r["reading"] for r in
           csv.DictReader(open(os.path.join(ROOT, "results", "selector_redraw_replication.csv"),
                               encoding="utf-8"))}
    if rep["B1"] == "DOES NOT REPLICATE":
        assert "\\emph{draw-dependent}" in sites["app:contaminated"]
        assert "crossed by that one anchor on the first draw and by none on the second" in sites["app:contaminated"]
