"""v12 (2026-09-25): the review items first skipped and then pursued. Each guard reads the CSV that
conceded or measured the claim and checks the sentence that states it, so a reword or a re-run that
changes the number fails here by name (cautions (ai), (aq))."""
import csv
import os
import re

from tests.manuscript import ROOT, body, caption_of

R = os.path.join(ROOT, "results")


def _csv(name):
    return list(csv.DictReader(open(os.path.join(R, name))))


def test_holm_over_the_forest_rows_is_stated_as_measured():
    import sys
    from analysis.forest_multiplicity import holm, p_from_interval
    sys.path.insert(0, os.path.join(ROOT, "figures"))
    from make_figures_v4 import h2h_forest_rows
    rows = _csv("forest_multiplicity.csv")
    fig = h2h_forest_rows()
    assert len(rows) == len(fig) == 21
    # the CSV is the figure's own rows, and its flags are Holm recomputed, never a stored label
    assert [(r["row"], float(r["value"]), float(r["lo95"]), float(r["hi95"])) for r in rows] == \
        [(lab, *v) for _, lab, v in fig]
    flags = holm([p_from_interval(*v) for _, _, v in fig])
    assert [r["holm_reject"] == "True" for r in rows] == flags
    rej = [r for r in rows if r["holm_reject"] == "True"]
    head = next(r for r in rows if r["row"].startswith("headline"))
    assert len(rej) == 8 and sum(r["sign"] == "+" for r in rej) == 7
    assert [r["row"] for r in rej if r["sign"] == "-"] == ["AlpacaEval-805, $k=1$"]
    assert head["holm_reject"] == "False" and abs(float(head["p_normal"]) - 0.005) < 0.0005
    t = body("appendix_selection.tex")
    assert "one family of $21$ under Holm's correction" in t
    assert "eight of Figure~\\ref{fig:h2h}'s rows still exclude zero, seven in selection's favour and AlpacaEval against it" in t
    assert "the headline row alone ($p \\approx 0.005$) does not survive a $21$-way correction" in t


def test_order_consistency_ranges_and_the_claim_about_the_most_consistent_judges():
    rows = {r["judge"]: r for r in _csv("judge_order_consistency.csv")}
    assert set(rows) == set("BCDEFG")
    t = body("appendix_selection.tex")
    arms = ("sel_n64", "sel_n1", "metered_k10", "anchor_k0")
    lows = {}
    for j in "BCDEFG":
        v = [float(rows[j][f"consistency_{a}"]) for a in arms]
        assert (min(v), max(v)) == (float(rows[j]["consistency_min"]), float(rows[j]["consistency_max"])), j
        assert f"${min(v):.2f}$--${max(v):.2f}$" in t, j
        lows[j] = min(v)
    top2 = sorted(lows, key=lambda j: -lows[j])[:2]
    assert sorted(top2) == ["D", "G"]
    assert all(rows[j]["d3_reading"] == "REVERSAL CONFIRMED" for j in top2)
    assert "the two most consistent judges both place the repaired difference above zero" in t


def test_the_containment_union_is_counted_and_charged_to_neither_certificate():
    t = body("appendix_proofs.tex")
    assert f"at most ${200 - 50 + 1}$ positions" in t
    assert "no position factor enters either certificate" in t


def test_the_onset_shading_is_not_called_an_interval():
    cap = caption_of("fig:horns")
    assert "(shaded: their range, not an interval)" in cap
    assert not re.search(r"shaded\)[^.]*confidence", cap)


def _row(name, band, start):
    return next(r for r in _csv(name) if r["band"] == band and r["quantity"].startswith(start))


def test_whole_response_judging_is_quoted_from_its_csv():
    t = body("appendix_selection.tex")
    r1, ref = _row("full_response.csv", "R1", "D3 uncut"), _row("full_response.csv", "ref", "D3 cut")
    assert r1["reading"] == "REVERSAL CONFIRMED"
    assert f"${float(r1['value']):+.3f}$ $[{float(r1['lo95']):+.4f}, {float(r1['hi95']):+.4f}]$ against ${float(ref['value']):+.3f}$ cut" in t
    bio = _row("full_response.csv", "desc", "D3 uncut, biographies")
    assert f"${float(bio['value']):+.4f}$ $[{float(bio['lo95']):+.4f}, {float(bio['hi95']):+.4f}]$ on the biographies alone" in t
    inc = {r["quantity"].split(":")[0]: int(r["value"]) for r in _csv("full_response.csv") if "longer than 1,200" in r["quantity"]}
    assert f"The cut reaches ${inc['opponent']}$ of the opponent's $500$ texts, ${inc['meter k=10']}$ of the meter's at $k=10$, ${inc['selection n=64']}$ of selection's and no prompt" in t
    assert inc["prompt"] == 0


def test_chat_onset_is_quoted_with_its_no_crossing_share():
    t = body("appendix_selection.tex")
    c2 = _row("chat_onset.csv", "C2", "chat onset")
    g1 = _row("chat_onset.csv", "G1", "memoriser alone")
    nc = _row("chat_onset.csv", "desc", "bootstrap resamples with no crossing")
    lo, hi = float(c2["lo95"]), float(c2["hi95"])
    assert c2["reading"] == ("ABOVE" if lo > 3 else "BELOW" if hi < 3 else "STRADDLES") == "STRADDLES"
    assert f"$k={float(c2['value']):.3f}$ $[{lo:.3f}, {hi:.3f}]$" in t
    assert f"near-verbatim recall ${float(g1['value']):.4f}$" in t and float(g1["value"]) >= 0.10
    assert f"with ${float(nc['value']):.1f}\\%$ of resamples never reaching it" in t
    assert "coincide within that interval" in t
