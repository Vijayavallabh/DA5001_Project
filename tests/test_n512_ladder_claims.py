"""feat-181's consequences in the manuscript (results/onset_prediction_n512_ladder.md, read 2026-09-23).

Every band Appendix I quotes from the n = 512 pass rounds from results/n512_ladder.csv, and every
sentence that states a READING is conditioned on the reading in that CSV rather than on its own
wording, so a re-run that reads differently fails here instead of leaving stale prose (caution (aq)).
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import body, carries_band  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rows():
    with open(os.path.join(ROOT, "results", "n512_ladder.csv"), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _row(arm, band, quantity):
    got = [r for r in _rows() if (r["arm"], r["band"], r["quantity"]) == (arm, band, quantity)]
    assert len(got) == 1, (arm, band, quantity, got)
    return got[0]


def _para(start):
    txt = body("appendix_selection.tex")
    i = txt.index(start)
    j = txt.find("\\paragraph{", i + 1)
    return txt[txt.rfind("\\paragraph{", 0, i + 1):j if j > 0 else None]


# v10 (2026-09-24) retitled the paragraph ("How far $n$ pays.") and moved its bands into a table;
# it is located by the reference to that table, so a retitle cannot retire the guards.
LADDER = "Table~\\ref{tab:ladder}"


def test_every_n512_band_the_appendix_quotes_rounds_from_the_csv():
    for arm, band, q in (("A", "C2", "g(512)-g(256)"), ("B", "C2", "g(512)-g(256)"),
                         ("A", "C3", "g(512)-g(64)"), ("B", "C3", "g(512)-g(64)"),
                         ("A", "C1", "D3 n=512"), ("A", "C1", "g(512)-g(256) order-averaged"),
                         ("A", "C1", "g(512)-g(64) order-averaged")):
        r = _row(arm, band, q)
        assert carries_band(float(r["value"]), float(r["lo95"]), float(r["hi95"]),
                            "appendix_selection.tex"), (arm, band, q)


def test_the_slope_sentence_follows_c3_and_the_order_averaged_pair_carries_its_margin():
    c3 = [_row(a, "C3", "g(512)-g(64)")["reading"] for a in ("A", "B")]
    para = _para(LADDER)
    if all(w == "NOT RESOLVED" for w in c3):
        assert "we claim neither a ceiling nor a slope" in para.lower()
        assert "we had predicted a marginal climb" in para, "the wrong prediction was trimmed"
    else:
        assert "neither a ceiling nor a slope" not in para, f"C3 reads {c3}; the sentence is stale"
    c2 = [_row(a, "C2", "g(512)-g(256)")["reading"] for a in ("A", "B")]
    if all(w == "SATURATED" for w in c2):
        assert "the last doubling is flat on both" in para
    hw = [float(_row("A", "C1", q)["half_widths"]) for q in
          ("g(512)-g(256) order-averaged", "g(512)-g(64) order-averaged")]
    assert f"at ${hw[0]:.2f}$ and ${hw[1]:.2f}$ half-widths" in para
    if min(hw) < 1.7:
        assert "the margin at which a paired difference has failed to replicate" in para


def test_the_off_support_sentence_uses_the_reading_c1_returned():
    d3 = _row("A", "C1", "D3 n=512")
    txt = body("appendix_selection.tex")
    word = {"UNRESOLVED": "\\textsc{unresolved}", "STILL BEHIND": "\\textsc{still behind}",
            "CATCHES": "\\textsc{catches}"}[d3["reading"]]
    assert f"reads as {word}" in txt, d3["reading"]
    if d3["reading"] != "STILL BEHIND":
        assert "against our prediction that it would still trail" in txt
    assert "fails off-support at $512$" not in txt or float(d3["hi95"]) < 0
    c4 = _row("A", "C4", "certificate log 512")
    assert f"$\\log 512 = {float(c4['value']):.2f}$ nats" in txt
    assert f"${float(c4['reading']):.1f}\\times$ below the meter's realised spend" in txt


def test_limitations_reaches_as_far_as_the_judged_ladders_did():
    """v10 (2026-09-24) states the reach inside the reading ("To $n=512$ no judged ladder turns
    over") instead of in a separate "our judged arms $n=512$" clause; the reach is read off the CSV."""
    import re
    lim = body("appendix_limitations.tex")
    low = lim.lower()
    reach = max(int(m) for r in _rows() for m in re.findall(r"g\((\d+)\)", r["quantity"]))
    c2 = [_row(a, "C2", f"g({reach})-g({reach // 2})")["reading"] for a in ("A", "B")]
    assert f"to $n={reach}$" in low and "judged arms reach $n=256$" not in lim
    if "TURNS OVER" not in c2:
        assert f"to $n={reach}$ no judged ladder turns over" in low
    else:
        assert "no judged ladder turns over" not in low, f"a judged ladder turned over at {reach}"
