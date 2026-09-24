"""Guards for the round that ran the review points skipped on 2026-09-24 (feat-187..193 and the notes).

Every number the new paragraphs print is re-derived here from the CSV or the arithmetic that produced
it, and each guard reads its own paragraph by label, not the file at large (caution (an)).
"""
import csv
import math
import os
import re

import pytest

from tests.manuscript import body

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def para(label, *files):
    """The whitespace-normalised paragraph that carries \\label{label}, up to the next heading."""
    t = body(*files)
    i = t.index(f"\\label{{{label}}}")
    j = min([k for k in (t.find("\\paragraph{", i), t.find("\\section", i), t.find("\\subsection", i))
             if k > 0] or [len(t)])
    return t[i:j]


def rows(name):
    return list(csv.DictReader(open(os.path.join(RES, name), encoding="utf-8")))


def test_the_per_work_certificate_arithmetic():
    p = para("app:perwork", "appendix_selection.tex")
    assert "$19{,}200$" in p and round(3 * 64 / 0.01) == 19200
    assert "$3 \\times 64/25{,}600 = 0.75\\%$" in p and abs(3 * 64 / 25600 - 0.0075) < 1e-12
    assert "$e^{100} \\approx 2.7 \\times 10^{43}$" in p and 2.65e43 < math.exp(100) < 2.75e43
    assert "$S_w = 159.8$" in p and "$10^{-60}$" in p and "$m=10^4$" in p
    assert math.log10(64 * 1e4) - 159.8 / math.log(10) < -60


def test_the_pools_the_per_work_bound_rests_on_have_no_event():
    r = [x for x in rows("rouge_threshold.csv")
         if x["file"] == "selfix_clean_n256_per_passage.csv" and x["source"] == "pool maximum"]
    assert r and all(int(x["count"]) == 0 and int(x["n"]) == 100 for x in r)
    assert {x["theta"] for x in r} >= {"0.3", "0.5", "0.7"}


def test_the_no_separation_remark_names_its_counterexamples():
    p = para("app:noseparation", "appendix_proofs.tex")
    for s in ("false as a statement about laws", "V_t = \\mathbb{E}_{p_s}[e^{\\lambda U} \\mid y_{\\le t}]",
              "information", "Proposition~\\ref{prop:imitation}"):
        assert s in p, s


def test_rouge_threshold_clean_anchors_zero_at_every_theta():
    r = [x for x in rows("rouge_threshold.csv") if x["kind"] == "clean" and x["source"] != "memoriser alone"]
    assert r and all(int(x["count"]) == 0 for x in r)
    mem = {x["theta"]: int(x["count"]) for x in rows("rouge_threshold.csv")
           if x["file"] == "selfix_clean_grid64_per_passage.csv" and x["source"] == "memoriser alone"}
    assert mem["0.5"] == 47 and mem["0.3"] >= mem["0.5"] >= mem["0.7"]


def test_the_covert_channel_never_beats_its_certificate():
    for r in rows("covert_channel.csv"):
        n = int(r["n"])
        assert float(r["bits_per_response"]) < math.log2(n)
        assert int(r["odometer_max_responses"]) == int(400 // math.log(n))
    r64 = next(r for r in rows("covert_channel.csv") if r["n"] == "64")
    assert abs(float(r64["efficiency"]) - float(r64["bits_per_response"]) / 6) < 1e-3


def test_adaptive_rules_are_certified_at_log_n_max():
    for r in rows("adaptive_n.csv"):
        assert abs(float(r["certificate_nats"]) - math.log(int(r["n_max"]))) < 1e-4
        assert 1 <= float(r["mean_draws"]) <= int(r["n_max"])


def _gains(tag):
    d = {r["quantity"][:2]: r for r in rows(f"order_averaged_h2h__{tag}.csv")}
    return {k: (float(v["value"]), float(v["lo95"]), float(v["hi95"])) for k, v in d.items()
            if k in ("D1", "D2", "D3")}


def test_every_older_reading_survives_the_repair_as_the_appendix_says():
    r = rows("deecho_rejudge.csv")
    d3 = [x for x in r if x["quantity"].startswith("D3")]
    assert len(d3) == 20 and all(x["survives"] == "YES" for x in d3)
    panel = [x for x in d3 if x["pass"].startswith("panel")]
    assert sum(x["reading_deecho"] == "POSITIVE" for x in panel) == 4 and len(panel) == 5
    moved = sorted((x["pass"], x["quantity"][:2]) for x in r if x["survives"] == "NO")
    assert moved == [("workload CoTaEval-QA k=1.4", "D1"), ("workload Gutenberg k=0.9", "D2"),
                     ("workload MT-Bench k=1.0", "D1")], moved
    t = body("appendix_selection.tex")
    i = t.index("Every older order-averaged reading was re-judged")
    p = t[i:i + 1100]
    assert "all twenty keep their sign class" in p and "four positive judges of five" in p
    for tag, q in (("mixtral_deecho", "D3"), ("mixpowk_judgeB_deecho", "D3"), ("mtb_conc_bind_deecho", "D1"),
                   ("cotaeval_qa_conc_bind_deecho", "D1"), ("gutenberg_conc_bind_deecho", "D2")):
        g, lo, hi = _gains(tag)[q]
        assert f"${g:+.4f}$ $[{lo:+.4f}, {hi:+.4f}]$" in p or f"${g:+.4f}$" in p, (tag, q, g)


def test_the_decomposition_on_repaired_text_is_rebuilt_from_its_csvs():
    ours, alp, mtb = (_gains(f"{t}_deecho") for t in ("wscope_c", "mixpowk_judgeB", "mtb_conc_bind"))
    sel = [x["D1"][0] for x in (ours, alp, mtb)]
    met = [x["D2"][0] for x in (ours, alp, mtb)]
    ss, sm = round(max(sel) - min(sel), 3), round(max(met) - min(met), 3)
    assert mtb["D1"][1] < 0 < mtb["D1"][2], "MT-Bench's selection gain no longer covers zero"
    assert met[0] < sel[0] and met[1] > sel[1] and met[2] > sel[2], "the meter no longer crosses"
    t = body("appendix_selection.tex")
    i = t.index("On the repaired text that decomposition does not survive")
    p = t[i:i + 520]
    assert f"${ss:.3f}$" in p and f"${sm:.3f}$" in p, (ss, sm, p[:200])
    for v in sel + met:
        assert f"${v:+.4f}$" in p, v


def test_the_main_text_quotes_alpacaeval_on_the_repaired_text_like_its_headline():
    g, lo, hi = _gains("mixpowk_judgeB_deecho")["D3"]
    assert hi < 0
    t = body("experiments.tex")
    assert f"${g:+.4f}$ $[{lo:+.4f}, {hi:+.4f}]$" in t, "the main text's AlpacaEval band is not the repaired one"
    assert "$-0.0339$" not in t, "the echo-carrying AlpacaEval number is back in the main text"


def test_the_threshold_paragraph_quotes_the_recount():
    p = para("app:rougethreshold", "appendix_selection.tex")
    mem = {x["theta"]: x["count"] for x in rows("rouge_threshold.csv")
           if x["file"] == "selfix_clean_grid64_per_passage.csv" and x["source"] == "memoriser alone"}
    assert f"${mem['0.3']}$, ${mem['0.4']}$, ${mem['0.5']}$, ${mem['0.6']}$ and ${mem['0.7']}$" in p
    assert "reaches none of the $100$ passages" in p


def test_the_channel_paragraph_quotes_its_csv():
    p = para("app:channel", "appendix_selection.tex")
    c = {r["n"]: r for r in rows("covert_channel.csv")}
    for n in ("8", "16", "64"):
        assert f"${float(c[n]['bits_per_response']):.2f}$" in p, n
    assert f"${float(c['64']['analytic_bits']):.2f}$" in p
    assert f"${1 - float(c['64']['distinct_draw_frac']):.1%}$".replace("%", "\\%") in p
    w = {(r["n"], r["encoding"]): r for r in rows("covert_channel_windows.csv")}
    lz = w[("64", "lzma")]
    assert f"${int(float(lz['mean_bits'])):,}$".replace(",", "{,}") in p
    assert f"${round(float(lz['mean_responses']))}$" in p and f"${round(float(lz['responses_at_capacity']))}$" in p
    assert f"= {c['64']['odometer_max_responses']}$ responses" in p
    assert f"= {float(c['64']['odometer_max_bits']):.0f}$ bits" in p


def test_the_adaptive_paragraph_quotes_its_contrasts():
    p = para("app:adaptive", "appendix_selection.tex")
    c = {r["contrast"]: r for r in rows("levels_adaptive_contrasts.csv")}
    for k in ("q75_vs_n64", "q75_vs_n7", "q90_vs_n64"):
        v, lo, hi = (float(c[k][x]) for x in ("value", "lo95", "hi95"))
        assert f"${v:+.4f}$ $[{lo:+.4f}, {hi:+.4f}]$" in p, k
    a = {(r["n_max"], r["q"]): r for r in rows("adaptive_n.csv")}
    assert f"${float(a[('64', '0.75')]['mean_draws']):.1f}$" in p and f"${float(a[('64', '0.9')]['mean_draws']):.1f}$" in p


def test_the_hybrid_paragraph_quotes_its_pass():
    p = para("app:hybrid", "appendix_selection.tex")
    c = {r["contrast"]: r for r in rows("levels_hybrid_contrasts.csv")}
    for k in c:
        v, lo, hi = (float(c[k][x]) for x in ("value", "lo95", "hi95"))
        assert f"${v:+.4f}$ $[{lo:+.4f}," in p, k
    lv = {r["arm"]: float(r["level"]) for r in rows("levels_hybrid.csv")}
    seq = [lv[a] for a in ("pw1", "hyb2", "hyb8", "sel64")]
    assert seq == sorted(seq), "the paragraph says the level rises monotonically with n"
    assert f"${seq[0]:.4f}$, ${seq[1]:.4f}$, ${seq[2]:.4f}$ and ${seq[3]:.4f}$" in p
    assert lv["pw1"] < lv["sel1"] and f"${lv['sel1']:.4f}$" in p
    assert "BELOW ZERO" == next(r["reading"] for r in rows("levels_hybrid_contrasts.csv") if r["contrast"].startswith("H1"))


def test_the_empty_preference_paragraph_quotes_its_csv():
    p = para("app:empties", "appendix_selection.tex")
    e = {(r["quantity"], r["n"]): r for r in rows("empty_preference.csv")}
    assert f"${100 * float(e[('draws empty', '64')]['value']):.1f}\\%$" in p
    med_e, med_n = (float(e[(q, "64")]["value"]) for q in ("median reward, empty draws",
                                                           "median reward, non-empty draws"))
    assert med_e > med_n and f"${med_e:.2f}$ against ${med_n:.2f}$" in p
    served = [100 * float(e[("served empty", n)]["value"]) for n in ("1", "8", "64")]
    assert f"${served[0]:.1f}\\%$ of\nprompts".replace("\n", " ") in p or f"${served[0]:.1f}\\%$" in p
    for v in served[1:]:
        assert f"${v:.1f}\\%$" in p
    le = [float(e[("judged level, served empty", n)]["value"]) for n in ("1", "8", "64")]
    ln = [float(e[("judged level, served non-empty", n)]["value"]) for n in ("1", "8", "64")]
    assert f"${le[0]:.4f}$, ${le[1]:.4f}$ and ${le[2]:.4f}$" in p
    assert f"${ln[0]:.4f}$, ${ln[1]:.4f}$ and ${ln[2]:.4f}$" in p
    # "costs selection level at n >= 8": the served-empty prompts sit below the rest there
    assert all(a < b for a, b in zip(le[1:], ln[1:]))


def test_the_seed52_draw_on_repaired_text_is_the_row_the_table_prints():
    d = {r["quantity"][:2]: r for r in rows("order_averaged_h2h_seed52_deecho.csv")}
    v = {q: float(d[q]["value"]) for q in ("D1", "D2", "D3")}
    lo, hi = float(d["D3"]["lo95"]), float(d["D3"]["hi95"])
    t = body("appendix_selection.tex")
    row = (f"seed $52$, repaired & B & ${v['D1']:+.4f}$ & ${v['D2']:+.4f}$ & ${v['D3']:+.4f}$ "
           f"$[{lo:+.4f},{hi:+.4f}]$")
    assert row in t, row
    head = {r["quantity"][:2]: float(r["value"]) for r in rows("order_averaged_h2h_deecho.csv")}
    assert f"read ${head['D3']:+.4f}$ and ${v['D3']:+.4f}$" in t


def test_the_workload_paragraph_carries_alpacaeval_on_the_repaired_text_too():
    g, lo, hi = _gains("mixpowk_judgeB_deecho")["D3"]
    p = para("app:workload", "appendix_selection.tex")
    assert f"${g:+.4f}$ $[{lo:+.4f}, {hi:+.4f}]$ on the repaired text" in p


def test_the_he_metrics_paragraph_quotes_its_csv():
    p = para("app:hemetrics", "appendix_selection.tex")
    h = {(r["metric"], r["arm"] or r["contrast"]): r for r in rows("he_metrics.csv")}
    lvl = lambda m, a: float(h[(m, a)]["value"])
    band = lambda m, c: tuple(float(h[(m, c)][k]) for k in ("value", "lo95", "hi95"))
    F, P = "factscore_precision", "prometheus_fluency_nonempty"
    assert f"${lvl(P, 'sel64'):.2f}$ against the meter's ${lvl(P, 'met_k10'):.2f}$" in p
    for c in ("sel64-met_k10", "sel64-met70_k0.5", "sel64-chat_k10"):
        v, lo, hi = band(P, c)
        assert f"${v:+.2f}$ $[{lo:+.2f}, {hi:+.2f}]$" in p, c
    assert band(P, "sel64-chat_k10")[2] < 0 < band(P, "sel64-met_k10")[1]
    assert f"${lvl(F, 'sel64'):.3f}$ against ${lvl(F, 'sel1'):.3f}$" in p
    for c in ("sel64-sel1", "sel64-met_k10"):
        v, lo, hi = band(F, c)
        assert f"${v:+.3f}$ $[{lo:+.3f}, {hi:+.3f}]$" in p, c
    lo, hi = band(F, "sel64-sel1")[1:]
    assert lo < 0 < hi, "selection now moves precision off its anchor; the paragraph says it does not"
    n = {a: int(h[(F, a)]["n"]) for a in ("sel64", "sel1")}
    assert f"abstains on ${150 - n['sel64']}$ of $150$" in p and f"abstains on ${150 - n['sel1']}$" in p


def test_the_replication_rows_are_the_scored_ones():
    d = {r["quantity"]: r for r in rows("headline_replication.csv")}
    t = body("appendix_selection.tex")
    for run, label in (("R2", "seeds $82$ (all new)"), ("R1", "seeds $82$ (old opp.)")):
        h2h = {r["quantity"][:2]: r for r in rows(f"order_averaged_h2h_{'replic_opp' if run == 'R2' else 'replic'}.csv")}
        v = {q: float(h2h[q]["value"]) for q in ("D1", "D2", "D3")}
        lo, hi = float(h2h["D3"]["lo95"]), float(h2h["D3"]["hi95"])
        row = f"{label} & B & ${v['D1']:+.4f}$ & ${v['D2']:+.4f}$ & ${v['D3']:+.4f}$ $[{lo:+.4f},{hi:+.4f}]$"
        assert row in t, row
        assert d[f"{run} D3 difference of gains, paired"]["reading"] == "REPLICATES"
    g = {r["quantity"][:2]: r for r in rows("order_averaged_h2h_replic_opp_nonempty.csv")}["D3"]
    assert f"raises it to ${float(g['value']):+.4f}$ $[{float(g['lo95']):+.4f}, {float(g['hi95']):+.4f}]$" in t
    assert all(r["reading"] == "PASS" for r in rows("headline_replication.csv") if r["gate"] in ("G0", "G1"))


def test_the_batched_latency_paragraph_and_table_quote_their_csvs():
    p = para("app:batched", "appendix_selection.tex")
    t = body("appendix_selection.tex")
    c = {(r["part"], r["arm"], r["risky"].split("/")[-1], r["W"], r["n"]): float(r["per_request_s"])
         for r in rows("batched_latency.csv")}
    s = lambda arm, W, n, risky="-": c[("single", arm, risky, W, n)]
    R8, R70 = "Meta-Llama-3.1-8B-Instruct", "Meta-Llama-3.1-70B"
    row = (f"selection, $n=1$ / $8$ / $64$ & ${s('SEL','1','1'):.2f}$ / ${s('SEL','1','8'):.2f}$ / "
           f"${s('SEL','1','64'):.2f}$ & ${s('SEL','8','1'):.3f}$ / ${s('SEL','8','8'):.3f}$ / ${s('SEL','8','64'):.2f}$")
    assert row in t, row
    assert f"meter, $8$B pair & ${s('MET','1','1',R8):.2f}$ & ${s('MET','8','1',R8):.3f}$" in t
    m70 = lambda W, arm="MET": c[("70b", arm, R70, W, "1")]
    assert f"meter, $70$B pair & ${m70('1'):.2f}$ & ${m70('8'):.2f}$" in t
    b = {(r["band"], r["W"], r["denominator"].split()[1]): r for r in rows("batched_latency_bands.csv")}
    q = lambda k: float(b[k]["ratio"])
    assert f"${q(('T1', '1', 'Meta-Llama-3.1-70B')):.3f}\\times$ the $70$B meter" in p
    assert b[("T1", "1", "Meta-Llama-3.1-70B")]["reading"] == "CONFIRMED"
    assert f"${q(('T2', '1', 'Meta-Llama-3.1-8B-Instruct')):.3f}\\times$ the meter" in p
    assert b[("T2", "1", "Meta-Llama-3.1-8B-Instruct")]["reading"] == "WITHIN NOISE" and "not read" in p
    assert f"${q(('T3', '1', '-')):.2f}\\times$ a single draw" in p
    f = {(r["W"], r["arm"]): r for r in rows("pareto_frontier.csv")}
    assert {r["arm"] for r in rows("pareto_frontier.csv") if r["pareto"] == "YES"} == {"sel_n1", "sel_n8", "sel_n64", "anchor"}
    for arm in ("met_k10", "met70_k20"):
        r = f[("1", arm)]
        assert r["best_dominator"] == "sel_n64"
        v, lo, hi = (float(r[k]) for k in ("level_margin", "margin_lo95", "margin_hi95"))
        assert f"${v:+.4f}$ $[{lo:+.4f}, {hi:+.4f}]$" in p, arm
