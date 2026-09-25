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
    # v13 (2026-09-25, Review 3 W10): the registered reading (the first win lies inside the onset's
    # interval) stays, and so does what the point estimate says -- at k=3 recall is still below 0.01
    assert "lies within that interval" in t or "coincide within that interval" in t
    assert float(c2["value"]) > 3 and "the meter wins there before any measured extraction" in t
    assert "the win comes before measured extraction, though within its interval" in body("experiments.tex")


def _bw(band, arm, start=""):
    return next(r for r in _csv("blockwise.csv") if r["band"] == band and r["arm"] == arm
                and r["quantity"].startswith(start))


def _iv(r):
    f = lambda x: f"{float(x):+.4f}".rstrip("0")
    return f"${f(r['value'])}$ $[{f(r['lo95'])}, {f(r['hi95'])}]$"


def test_every_cell_of_the_installments_table_is_its_csv_row():
    """tab:blockwise is generated from results/blockwise.csv; each row must still be that row (caution (j))."""
    import re
    src = body("appendix_onset.tex")
    tab = src[src.index("\\label{tab:blockwise}"):src.index("\\end{tabular}", src.index("\\label{tab:blockwise}"))]
    rows = {"blk200n64": ("once", 200, 64), "blk50n64": ("installments", 50, 64), "blk25n64": ("", 25, 64),
            "blk10n64": ("", 10, 64), "blk100n8": ("one $\\log 64$", 100, 8), "blk67n4": ("", 67, 4),
            "blk34n2": ("", 34, 2), "blk50n64_reward": ("prefix only", 50, 64),
            "blk10n64_planner": ("planner", 10, 64)}
    for arm, (lab, L, n) in rows.items():
        cert = _bw("desc", arm, "certificate")["value"].split(" / ")
        gain = (_bw("B1", arm, "gain") if arm in ("blk200n64", "blk50n64", "blk25n64", "blk10n64", "blk100n8",
                                                  "blk67n4", "blk34n2")
                else _bw("B4", arm, "gain") if arm.endswith("planner") else _bw("desc", arm, "gain over the anchor"))
        diff = next((r for r in _csv("blockwise.csv") if r["arm"] == f"{arm} - blk200n64"), None)
        line = f"{lab} & ${L}$ & ${n}$ & ${cert[0]}$ / ${cert[1]}$ & {_iv(gain)} & {_iv(diff) if diff else '---'} \\\\"
        assert line.strip() in tab, (arm, line)
    assert len(re.findall(r"\\\\", tab)) == len(rows) + 1          # the header plus one line per arm


def _d(x):
    """Four decimals with one trailing zero dropped, the precision feat-208's readings are printed at."""
    s = f"{float(x):+.4f}"
    return s[:-1] if s.endswith("0") else s


def _iv3(r):
    return f"${_d(r['value'])}$ $[{_d(r['lo95'])}, {_d(r['hi95'])}]$"


def _rep(band, start):
    return next(r for r in _csv("blockwise_replication.csv") if r["band"] == band and r["quantity"].startswith(start))


def test_the_installments_claims_follow_their_verdicts():
    t = body("frontier.tex")
    w10 = _bw("B2", "blk10n64 - blk200n64")
    assert w10["reading"] == "INSTALLMENTS WIN"
    p1 = _rep("P1", "judge B, new draw: L=10")
    assert p1["reading"] == "INSTALLMENTS WIN", "the fresh draw no longer reproduces; Section 3 must say so"
    assert (f"$10$-token installments beat one choice by {_iv(w10)}, and by {_iv3(p1)} on a fresh draw, "
            "at $25$ nats a window") in t
    assert round(float(_bw("desc", "blk10n64", "certificate")["value"].split(" / ")[1])) == 25
    b3 = [_bw("B3", f"{a} - blk200n64") for a in ("blk100n8", "blk67n4", "blk34n2")]
    assert all(r["reading"] == "ONCE WINS" for r in b3)
    assert "while held to one whole-output $\\log 64$ they lose" in t
    a = body("appendix_onset.tex")
    assert "against our registered prediction of a tie" in a
    b4 = _bw("B4", "blk10n64_planner - blk10n64")
    assert b4["reading"] == "VALUE WINS" and f"loses to the reward by ${-float(b4['value']):.4f}$ $[{-float(b4['hi95']):.3f}, {-float(b4['lo95']):.3f}]$" in a
    assert "reads near-verbatim recall $0.0000$ on all $100$ passages it memorised" in a
    assert _bw("B6", "blk10n64_memoriser", "near-verbatim recall, maximum")["value"] == "0.0"
    c = body("iclr_closing.tex")
    assert "we did not measure selection in installments" not in c
    assert "in installments, which buy more" in c


def test_the_notation_table_quotes_its_numbers_from_their_sources():
    """Reviews 3 and 4 asked for a notation table; its numbers are the paper's, not new ones."""
    from tests.manuscript import tex
    t = " ".join(open(tex("iclr_2027.tex"), encoding="utf-8").read().split())
    tab = t[t.index("\\caption{Notation and terms.}"):t.index("\\bottomrule", t.index("\\caption{Notation and terms.}"))]
    w = next(r for r in _csv("window_vacuity.csv") if r["window_tokens"] == "50")
    assert f"median ${float(w['S_median_nats']):.1f}$ nats" in tab
    assert "$\\gamma = 0.857$ nats per token" in tab and "$\\gamma = 0.857$" in body("frontier.tex")
    for label in ("prop:selection", "prop:threshold", "prop:sparse", "prop:imitation", "thm:nfl"):
        assert f"\\ref{{{label}}}" in tab, label
    assert "(notation: Table~\\ref{tab:notation})" in body("selection.tex")


def test_the_factscore_oracle_and_the_factuality_scorer_are_quoted_from_their_csv():
    t = body("appendix_selection.tex")
    rows = _csv("factscore_oracle.csv")
    get = lambda band, q: next(r for r in rows if r["band"] == band and r["quantity"] == q)
    o = get("F1", "oracle precision, first 64 drafts")
    o5 = get("F1", "oracle precision, first 64 drafts, drafts with >= 5 facts")
    f2 = get("F2", "factuality pick minus committed pick, paired")
    f3 = get("F3", "factuality pick minus the meter at k=10, paired")
    assert f"reaches ${float(o['value']):.4f}$ $[{float(o['lo95']):.4f}, {float(o['hi95']):.4f}]$" in t
    assert f"(${float(o5['value']):.4f}$ among draws claiming at least five facts)" in t
    assert f2["reading"] == "TIE" and f"paired ${float(f2['value']):+.4f}$ $[{float(f2['lo95']):+.4f}, {float(f2['hi95']):+.4f}]$ over the committed pick" in t
    assert f3["reading"] == "BELOW" and f"${float(f3['value']):+.4f}$ $[{float(f3['lo95']):+.4f}, {float(f3['hi95']):+.4f}]$ below the meter" in t
    assert "a scorer that asks for facts did not find them" in t
    assert "we did not test a factuality scorer" not in t


def test_the_planner_is_compared_with_the_meter_only_through_its_interval():
    r = _bw("desc", "blk10n64_planner - metered_k10")
    a = body("appendix_onset.tex")
    assert f"{_iv(r)} more than the meter does at $k=10$ (post hoc)" in a
    assert float(r["lo95"]) > 0, "the planner no longer exceeds the meter; 'more than' is stale"
    assert "above the meter's" not in a


def test_the_fresh_draw_and_the_second_judge_are_quoted_from_their_csv():
    """feat-208: the installment advantage re-drawn and read by judge G, each reading quoted beside the others and
    none pooled; P2 was registered as a tie, and the appendix says so."""
    a = body("appendix_onset.tex")
    rows = _csv("blockwise_replication.csv")
    assert all(r["reading"] == "PASS" for r in rows if r["band"] in ("G0", "G1", "G2"))
    p1, p2 = _rep("P1", "judge B, new draw: L=10"), _rep("P2", "judge B, new draw: L=25")
    p3, g25 = _rep("P3", "judge G, new draw: L=10"), _rep("desc", "judge G, new draw: L=25 minus once")
    assert all(r["reading"] == "INSTALLMENTS WIN" for r in (p1, p2, p3, g25))
    assert (f"reproduces it, where we had predicted a tie at $L = 25$: {_iv3(p1)} and {_iv3(p2)} at $L = 10$ and "
            f"$25$ under judge~B, {_iv3(p3)} and {_iv3(g25)} under judge~G") in a
    ph = [_rep("posthoc", f"judge G, feat-201's draw: L={L} minus once") for L in (10, 25, 50)]
    assert all(r["reading"] == "INSTALLMENTS WIN" for r in ph), "a post-hoc 'advantage' whose interval reaches zero"
    assert (f"read post hoc by judge~G, the first draw's advantage is ${_d(ph[0]['value'])}$, ${_d(ph[1]['value'])}$ "
            f"and ${_d(ph[2]['value'])}$ at $L = 10$, $25$ and $50$") in a


def test_vetting_at_the_deployed_temperature_is_quoted_from_its_csv():
    """feat-203: every rung of the ladder drew at 1.0; the judge-free runs sample at 0.7. The 0.7 rungs may be
    quoted as zeros only because their positive control leaked (G1), and the TriviaQA shares the body quotes are
    the tempered anchor's, the temperature those runs sampled at."""
    import math
    from tests.manuscript import tex
    v = _csv("vetting_t07.csv")
    get = lambda band, start: next(r for r in v if r["band"] == band and r["quantity"].startswith(start))
    g1 = get("G1", "positive control")
    assert g1["reading"] == "PASS" and int(g1["value"]) >= 10, "the 0.7 screen has no power; no zero from it may be quoted"
    assert all(r["reading"] == "ZERO" for r in v if r["band"] == "V1") and sum(r["band"] == "V1" for r in v) == 6
    a = body("appendix_selection.tex")
    ladder = a[a.index("\\label{tab:vetladder}"):a.index("\\end{tabular}", a.index("\\label{tab:vetladder}"))]
    at100 = re.search(r"Llama-3\.1-70B & [^&]+ & \$1\$ & \$2\$ & \$11\$ & \$20\$ & \$(\d+)\$", ladder).group(1)
    assert (f"Every rung drew at temperature $1.0$. At the judge-free runs' $0.7$, TinyComma and Comma-7B read $0$ of "
            f"$50$ at $20$, $100$ and $200$ tokens while the $70$B leaks on ${int(g1['value'])}$ of $50$ at $100$ "
            f"(${at100}$ at $1.0$)") in a
    assert "temperature $1.0$:" in caption_of("tab:vetladder")
    tc, c7 = get("V2", "tinycomma"), get("V2", "comma7b")
    assert all(r["reading"].startswith("ABOVE") for r in (tc, c7)), "'conservative' needs S_w to rise under tempering"
    ref = lambda r: float(r["reading"].split()[1])
    assert (f"$S_w$ from ${ref(tc):.1f}$ to ${float(tc['value']):.1f}$ nats and, over $50$ of Comma-7B's own tokens, "
            f"from ${ref(c7):.1f}$ to ${float(c7['value']):.1f}$, so margins quoted against the untempered anchor are "
            "conservative") in a
    # TriviaQA at the runs' temperature: the ordering is rebuilt from the tempered CSV, not taken on trust
    s = [float(r["s_anchor_nats"]) for r in _csv("tqa_vacuity_t07.csv")]
    share = lambda K: 100 * sum(x <= K for x in s) / len(s)
    met = {r["k"]: 100 * float(r["vacuous_frac"]) for r in _csv("tqa_vacuity_t07_summary.csv")}
    assert max(share(math.log(n)) for n in (4, 8, 16, 32, 64)) < min(met.values())
    sel64 = float(get("V3", "TriviaQA: share of questions with S(x) <= log 64, temperature 0.7")["value"])
    assert abs(sel64 - share(math.log(64))) < 0.05
    assert f"and still is under the anchor tempered to the runs' $0.7$, at ${sel64:.1f}\\%$ against ${met['0.5']:.1f}\\%$" in a
    assert f"$\\log 64$ reaches $S(x)$ on ${sel64:.1f}\\%$ of questions, against ${met['0.5']:.1f}\\%$ for the meter's smallest budget" in body("iclr_closing.tex")
    ethics = " ".join(open(tex("iclr_2027.tex"), encoding="utf-8").read().split())
    assert "every prefix length the deployment accepts, and at the temperature it samples at" in ethics


def test_the_nonempty_rule_across_table2_is_quoted_from_its_csv():
    """feat-209: the non-empty rule re-judged at eight configurations of Table 2. The body may say no configuration
    changes its reading only while every row keeps its label; if one ever does, the registration requires the body
    to name it, and this guard fails until it does."""
    from analysis.nonempty_tables import TABLE2
    rows = _csv("nonempty_tables.csv")
    assert len(rows) == 8 and all(int(r["n"]) == 500 for r in rows)
    f = lambda x: f"{float(x):+.4f}".rstrip("0")
    iv = lambda r, p: f"${f(r[p])}$ $[{f(r[p + '_lo95'])}, {f(r[p + '_hi95'])}]$"
    a = body("appendix_selection.tex")
    tab = a[a.index("\\label{tab:nonempty}"):a.index("\\end{tabular}", a.index("\\label{tab:nonempty}"))]
    for r in rows:
        assert r["same_label"] == str(r["D3_reading"] == r["D3_nonempty_reading"])
        shift = f(r["D3_shift"]) if float(r["D3_shift"]) else "0"
        assert f"& {iv(r, 'D3')} & {iv(r, 'D3_nonempty')} & ${shift}$ \\\\" in tab, r["row"]
        t2 = next(x for x in _csv(f"order_averaged_h2h_{TABLE2[r['row']]}.csv") if x["quantity"].startswith("D3"))
        assert abs(float(r["D3"]) - float(t2["value"]) - float(r["D3_shift"])) < 1e-9, r["row"]
    assert len(re.findall(r"\\\\", tab)) == 9
    h = next(r for r in rows if r["row"] == "headline")
    assert h["D3_nonempty_reading"] == "CONFIRMED"
    assert (f"the headline at {iv(h, 'D3_nonempty')} against the committed rule's {iv(h, 'D3')} in the same pass") in a
    exp = body("experiments.tex")
    if all(r["same_label"] == "True" for r in rows):
        # v13 (2026-09-25) states the same claim as "keeps +0.056 [...] and every reading of Table served"
        assert ("and under it no re-judged configuration of Table~\\ref{tab:served} changes the difference's reading" in exp
                or re.search(r"keeps \$\+[\d.]+\$ \$\[[^\]]*\]\$ and every reading of Table~\\ref\{tab:served\}", exp)), \
            "the body no longer says the non-empty rule keeps every reading of Table served"
        assert "the difference keeps its reading under the non-empty rule at all eight configurations" in a
    else:
        changed = [r["row"] for r in rows if r["same_label"] != "True"]
        raise AssertionError(f"rows {changed} change their reading; the body must name them (feat-209's registration)")


def test_anchoredbyte_at_the_authors_settings_is_quoted_from_its_csv():
    """feat-205: the authors' byte-level decoder at temperature 0.7 and penalty 1.1. A1 may be quoted only because the
    replacement control passed G1' (the anchor at every byte); if A1 ever reads SURVIVES, the registration requires
    the temperature-1.0 'indistinguishable' sentence and Section 4 to be scoped, and this guard fails until they are.
    The binding shares are the pooled construction on both sides (caution (ai): one word, one denominator)."""
    rows = _csv("anchoredbyte_t07.csv")
    get = lambda band, start: next(r for r in rows if r["band"] == band and r["quantity"].startswith(start))
    for band in ("G0", "G1", "G1'", "G2"):
        gs = [r for r in rows if r["band"] == band]
        assert gs and all(r["reading"] == "PASS" for r in gs), band
    f = lambda x: f"{float(x):+.4f}".rstrip("0")
    iv = lambda r: f"${f(r['value'])}$ $[{f(r['lo95'])}, {f(r['hi95'])}]$"
    a = body("appendix_selection.tex")
    a1, d3 = get("A1", "the meter's gain"), get("desc", "TinyComma selection")
    ab10 = next(r for r in _csv("anchoredbyte.csv") if r["k"] == "0.1")
    assert (f"binds on ${100 * float(ab10['binding_share']):.1f}\\%$ of byte steps and is judged indistinguishable from "
            "Comma-7B alone") in a
    if a1["reading"] != "DISSOLVES":
        raise AssertionError("A1 no longer dissolves: scope the temperature-1.0 sentence and Section 4 (feat-205)")
    bind = get("posthoc", "k=0.1: binding share pooled over bytes")
    assert f"the meter at $k=0.1$ binds on ${100 * float(bind['value']):.1f}\\%$ of byte steps and again gains nothing over its anchor, {iv(a1)}" in a
    assert d3["reading"] == "REVERSAL CONFIRMED" and float(d3["lo95"]) > 0
    assert f"TinyComma's selection at those settings gains {iv(d3)} more over its own" in a
    sw07, sw10 = get("desc", "S_w under Comma-7B at 0.7/1.1"), get("desc", "S_w under Comma-7B at 1.0")
    ratio = 80 / float(sw07["value"])
    assert sw07["reading"] == f"K/S_w {ratio:.3f}"
    assert (f"$S_w = {float(sw07['value']):.1f}$ nats against ${float(sw10['value']):.1f}$ untempered, both over the "
            f"$758$ protected works, so $K/S_w = {ratio:.3f}$") in a
