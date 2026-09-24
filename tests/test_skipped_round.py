"""Guards for the round that ran the review points skipped on 2026-09-24 (feat-187..193 and the notes).

Every number the new paragraphs print is re-derived here from the CSV or the arithmetic that produced
it, and each guard reads its own paragraph by label, not the file at large (caution (an)).
"""
import csv
import math
import os
import re

import pytest

from tests.manuscript import body, caption_of

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def h2h_figure_rows():
    """The bands Figure fig:h2h prints beside each row (figures/make_figures_v4.py). v10 moved several
    of this file's prose bands into that figure, so a guard must be able to see them there (caution (al))."""
    import sys
    fig = os.path.join(ROOT, "figures")
    if fig not in sys.path:
        sys.path.insert(0, fig)
    from make_figures_v4 import h2h_forest_rows  # noqa: E402
    return h2h_forest_rows()


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
    # v10 split v9's one paragraph three ways: the sign-class result stays in the repair paragraph, the
    # panel is summarised in Section 4 over six judges (the five-judge panel plus judge C on the
    # committed opponent), and AlpacaEval's repaired band sits in the workload paragraph.
    t = body("appendix_selection.tex")
    i = t.index("\\paragraph{A recording defect, and its repair.}")
    p = t[i:t.index("\\paragraph{", i + 1)]
    assert "all twenty older order-averaged differences keep their sign class" in p
    c = [x for x in d3 if x["pass"] == "ladder opp Llama-3.1-8B" and x["judge"].startswith("C ")]
    six = panel + c
    assert len(c) == 1 and sum(x["reading_deecho"] == "POSITIVE" for x in six) == 5 and len(six) == 6
    sec4 = body("experiments.tex")
    g, lo, hi = _gains("mixtral_deecho")["D3"]
    assert lo < 0 < hi and f"five of six judges exclude zero, and the sixth, \\texttt{{Mixtral-8x7B}}, " \
        f"straddles it at ${g:+.4f}$" in sec4, g
    g, lo, hi = _gains("mixpowk_judgeB_deecho")["D3"]
    assert f"${g:+.4f}$ $[{lo:+.4f}, {hi:+.4f}]$" in para("app:workload", "appendix_selection.tex"), g
    # the three single-arm gains that change class on the repaired text, in the repair paragraph
    for tag, q in (("mtb_conc_bind_deecho", "D1"), ("cotaeval_qa_conc_bind_deecho", "D1"),
                   ("gutenberg_conc_bind_deecho", "D2")):
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
    # v10's Section 4 quotes the reversal as the meter's margin and sends the reader to app:workload,
    # which prints the band; both must be the repaired reading.
    t = body("experiments.tex")
    assert f"reverses on AlpacaEval, where the meter wins by ${-g:.4f}$ (Appendix~\\ref{{app:workload}})" in t, \
        "the main text's AlpacaEval number is not the repaired one"
    assert f"${g:+.4f}$ $[{lo:+.4f}, {hi:+.4f}]$" in para("app:workload", "appendix_selection.tex"), \
        "the band the main text points at is not the repaired one"
    assert "0.0339" not in t, "the echo-carrying AlpacaEval number is back in the main text"


def test_the_threshold_paragraph_quotes_the_recount():
    # v10 sets \label{app:rougethreshold} at the END of the paragraph it names (`An adversarial
    # selector.`), so read back from the label to that paragraph's heading.
    t = body("appendix_selection.tex")
    i = t.index("\\label{app:rougethreshold}")
    p = t[t.rindex("\\paragraph{", 0, i):i]
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
    # v9's prose `the two draws read X and Y` became the table's `repaired text` row beside this one
    h = {r["quantity"][:2]: r for r in rows("order_averaged_h2h_deecho.csv")}
    hv = {q: float(h[q]["value"]) for q in ("D1", "D2", "D3")}
    head = (f"repaired text & B & ${hv['D1']:+.4f}$ & ${hv['D2']:+.4f}$ & $\\mathbf{{{hv['D3']:+.4f}}}$ "
            f"$[{float(h['D3']['lo95']):+.4f},{float(h['D3']['hi95']):+.4f}]$")
    assert head in t, head


def test_the_workload_paragraph_carries_alpacaeval_on_the_repaired_text_too():
    g, lo, hi = _gains("mixpowk_judgeB_deecho")["D3"]
    p = para("app:workload", "appendix_selection.tex")
    # v10 labels the paragraph's differences as recovered-text readings once, before quoting them
    i = p.find("binding-budget differences on the recovered text")
    assert 0 <= i < p.find(f"${g:+.4f}$ $[{lo:+.4f}, {hi:+.4f}]$"), (g, lo, hi)


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
    r2 = {r["quantity"][:2]: r for r in rows("order_averaged_h2h_replic_opp.csv")}["D3"]
    # v10 (app:empties) names the base it raises from, the R2 draw above
    assert (f"raises the fresh-seed headline difference from ${float(r2['value']):+.4f}$ to "
            f"${float(g['value']):+.4f}$ $[{float(g['lo95']):+.4f}, {float(g['hi95']):+.4f}]$") in t
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
    assert f"${q(('T2', '1', 'Meta-Llama-3.1-8B-Instruct')):.3f}\\times$ the $8$B meter's" in p
    assert b[("T2", "1", "Meta-Llama-3.1-8B-Instruct")]["reading"] == "WITHIN NOISE" and "not read" in p
    assert f"${q(('T3', '1', '-')):.2f}\\times$ a single draw" in p
    f = {(r["W"], r["arm"]): r for r in rows("pareto_frontier.csv")}
    assert {r["arm"] for r in rows("pareto_frontier.csv") if r["pareto"] == "YES"} == {"sel_n1", "sel_n8", "sel_n64", "anchor"}
    for arm in ("met_k10", "met70_k20"):
        r = f[("1", arm)]
        assert r["best_dominator"] == "sel_n64"
        v, lo, hi = (float(r[k]) for k in ("level_margin", "margin_lo95", "margin_hi95"))
        assert f"${v:+.4f}$ $[{lo:+.4f}, {hi:+.4f}]$" in p, arm


def test_the_human_and_legal_validation_is_conceded_as_not_done():
    """Two referees asked for human labels and a legal reading; neither was done, and the paper must
    say so rather than let the absence pass unmentioned (caution (ag): concessions go first in a trim).
    v10 keeps both in Appendix I's one sentence and the reason, that infringement is a legal question the
    paper does not answer, in the Ethics Statement."""
    from tests.manuscript import tex
    t = body("appendix_limitations.tex")
    assert "No human rated anything and no lawyer assessed any output" in t
    src = " ".join(open(tex("iclr_2027.tex"), encoding="utf-8").read().split())
    eth = src[src.index("\\section*{Ethics Statement}"):src.index("\\section*{Reproducibility Statement}")]
    assert "we make no claim about legal thresholds for copyright infringement" in eth


def test_the_anchoredbyte_table_and_the_abstract_follow_the_scored_bands():
    import csv as _csv
    from tests.manuscript import tex
    ab = {r["k"]: r for r in rows("anchoredbyte.csv")}
    assert set(ab) == {"0.1", "0.5", "2"} and all(r[g] == "PASS" for r in ab.values() for g in ("G0", "G1", "G2"))
    # v10 folded the AnchoredByte table into Section 4's Table tab:served (columns k, binds, K/S_w,
    # meter, selection - meter); K is no longer a column, and app:anchoredbyte states K = 800k instead.
    e = body("experiments.tex")
    t = e[e.index("\\label{tab:served}"):e.index("\\end{tabular}", e.index("\\label{tab:served}"))]
    assert "so that $K = 800k$ nats" in para("app:anchoredbyte", "appendix_selection.tex")
    for k, r in ab.items():
        h2h = {x["quantity"][:2]: x for x in rows(f"order_averaged_h2h_ab70_k{k}.csv")}
        lo, hi = float(h2h["D3"]["lo95"]), float(h2h["D3"]["hi95"])
        pp = list(_csv.DictReader(open(os.path.join(RES, f"order_averaged_h2h_per_prompt_ab70_k{k}.csv"))))
        lvl = sum(float(x[f"u_metered_k{k}"]) for x in pp) / len(pp)
        assert abs(float(r["K"]) - 800 * float(k)) < 1e-9, (k, r["K"])
        cells = (f"& ${float(k):g}$ & ${100 * float(r['binding_share']):.1f}\\%$ & ${float(r['K_over_Sw']):.2f}$ & "
                 f"${lvl:.4f}$ & ${float(h2h['D3']['value']):+.4f}$ $[{lo:+.4f}, {hi:+.4f}]$")
        assert cells in t, cells
        assert r["D3_reading"] == "REVERSAL CONFIRMED" and lo > 0
    abstract = " ".join(open(tex("iclr_2027.tex"), encoding="utf-8").read().split())
    # registered consequence: every band CONFIRMED -> the abstract names their byte-level decoder
    assert "against their byte-level decoder" in abstract[abstract.index("begin{abstract}"):abstract.index("end{abstract}")]


def test_the_main_text_quotes_the_batched_clock_at_both_pairs():
    """feat-190 fixed that the main text's cost prose and the Conclusion quote the batched per-request
    ratio beside the unbatched 21.8x, at both pairs in the cost prose. v10 moved that prose from Section 2
    (selection.tex) to Section 4's cost subsection (sec:cost); the Conclusion is iclr_closing.tex."""
    b = {(r["band"], r["W"]): float(r["ratio"]) for r in rows("batched_latency_bands.csv")
         if r["band"] in ("T1", "T2")}
    e = body("experiments.tex")
    sel = e[e.index("\\label{sec:cost}"):]
    assert f"${b[('T1', '1')]:.2f}\\times$ the meter's time at the authors' $70$B pair" in sel
    assert f"${b[('T2', '1')]:.2f}\\times$ at the $8$B one" in sel and "$21.8\\times$" in sel
    close = body("iclr_closing.tex")
    assert f"${b[('T1', '1')]:.2f}\\times$ its time per request at the $70$B pair" in close
    assert "$21.8\\times$" in close


def test_the_degeneracy_filter_check_quotes_the_rejudged_breadth():
    r = rows("selection_breadth_rejudged.csv")
    aud = next(x for x in r if "audited" in x["anchor"] and "Phi" in x["judge"])
    t = " ".join(body("appendix_limitations.tex").split())
    assert (f"${float(aud['gain_nonempty']):+.3f}$ $[{float(aud['gain_nonempty_lo95']):+.3f}, "
            f"{float(aud['gain_nonempty_hi95']):+.3f}]$ against ${float(aud['gain']):+.3f}$ on all $500$") in t
    phi = [x for x in r if "Phi" in x["judge"]]
    moveB = max(abs(float(x["gain"]) - float(x["gain_nonempty"])) for x in phi)
    # v10 states the bound in Figure fig:breadth's caption (a judge-B figure) about its open markers,
    # the anchors that fail the empty-draft gate; the bound over them is the bound over every anchor.
    assert moveB == max(abs(float(x["gain"]) - float(x["gain_nonempty"])) for x in phi
                        if x["entry_gate"] == "FAIL")
    assert f"on non-empty prompts their gains move by at most ${moveB:.3f}$" in caption_of("fig:breadth")
    lost = [x for x in r if float(x["gain_lo95"]) > 0 and float(x["gain_nonempty_lo95"]) <= 0]
    assert [(x["anchor"], "Phi" in x["judge"]) for x in lost] == [("Comma-7B (1T tokens)", True)]
    assert "every interval that excluded zero still does but one, Comma-1T's under judge~B" in t
    assert all(float(x["gain_nonempty"]) > float(x["gain"]) for x in r
               if x["anchor"].startswith("Comma-7B") and "Llama" in x["judge"])


def test_the_anchoredbyte_paragraph_quotes_the_he_metrics_of_its_arms():
    h = {(r["metric"], r["arm"] or r["contrast"]): r for r in rows("he_metrics.csv")}
    p = para("app:anchoredbyte", "appendix_selection.tex")
    F = "factscore_precision"
    assert f"${float(h[(F, 'ab_k0.5')]['value']):.3f}$ against selection's ${float(h[(F, 'csel64')]['value']):.4f}$" in p
    assert f"at $k=0.1$ it is ${float(h[(F, 'ab_k0.1')]['value']):.4f}$" in p
    lo, hi = float(h[(F, "csel64-ab_k0.1")]["lo95"]), float(h[(F, "csel64-ab_k0.1")]["hi95"])
    assert lo < 0 < hi, "the k=0.1 meter now separates from selection on precision; 'inseparable' is wrong"
    assert all(float(h[("prometheus_fluency_nonempty", f"csel64-ab_k{k}")]["lo95"]) > 0 for k in ("0.1", "0.5", "2")), \
        "selection is no longer the more fluent at every budget"


def test_the_byte_level_timing_is_the_measured_one():
    c = {(r["part"], r["arm"], r["W"], r["n"]): float(r["per_request_s"]) for r in rows("batched_latency.csv")}
    t = body("appendix_selection.tex")
    ab1, ab8 = c[("ab", "METAB", "1", "1")], c[("ab", "METAB", "8", "1")]
    s64 = c[("ab", "SEL", "1", "64")]
    assert f"AnchoredByte, Comma-7B $+$ $70$B & ${ab1:.2f}$ & ${ab8:.2f}$" in t
    assert (f"selection at Comma-7B, $n=1$ / $8$ / $64$ & ${c[('ab', 'SEL', '1', '1')]:.2f}$ / "
            f"${c[('ab', 'SEL', '1', '8')]:.2f}$ / ${s64:.2f}$ & ${c[('ab', 'SEL', '8', '1')]:.3f}$ / "
            f"${c[('ab', 'SEL', '8', '8')]:.3f}$ / ---") in t
    assert ("ab", "SEL", "8", "64") not in c, "the out-of-memory cell was measured after all; fill the dash"
    p = para("app:batched", "appendix_selection.tex")
    assert f"${ab1:.2f}$ s against ${s64:.2f}$ s" in p and f"${s64 / ab1:.3f}\\times$" in p


def test_the_headline_sentence_carries_its_fresh_seed_replication():
    """feat-188 fixed: REPLICATES -> the main text carries the disjoint-seed draw (R2). v10's Section 4
    states the headline band, then how far every re-draw moves it (selection re-drawn; every sampled arm
    re-drawn, R2; the same against the committed opponent, R1), and Figure fig:h2h prints each band."""
    def d3(name):
        d = {r["quantity"][:2]: r for r in rows(name)}["D3"]
        return float(d["value"]), float(d["lo95"]), float(d["hi95"])
    head = d3("order_averaged_h2h_deecho.csv")
    draws = {f: d3(f) for f in ("order_averaged_h2h_seed52_deecho.csv", "order_averaged_h2h_replic_opp.csv",
                                "order_averaged_h2h_replic.csv")}
    assert all(lo > 0 for _, lo, _ in draws.values()), "a re-draw no longer excludes zero"
    t = " ".join(body("experiments.tex").split())
    assert f"${head[0]:+.4f}$ $[{head[1]:+.4f}, {head[2]:+.4f}]$" in t, "the headline band left Section 4"
    move = max(abs(head[0] - g) for g, _, _ in draws.values())
    assert f"Re-drawing selection, then every sampled arm, moves it by at most ${move:.4f}$" in t, move
    fig = {label: band for _g, label, band in h2h_figure_rows()}
    r2 = fig["every arm re-drawn, seeds 82-84"]
    assert all(abs(a - b) < 1e-9 for a, b in zip(r2, draws["order_averaged_h2h_replic_opp.csv"])), \
        "the replication left Figure fig:h2h"


def test_the_abstracts_batched_claim_holds_at_both_70b_pairs():
    """'batched it serves a request at the authors' 70B pairs in at most X the meter's time': X must
    bound BOTH the token-level pair (feat-190 T1) and the byte-level pair (the ab note), and the
    unbatched 21.8x must stay beside it."""
    from tests.manuscript import tex
    b = [r for r in rows("batched_latency_bands.csv") if r["W"] == "1" and (
        r["band"] == "T1" or (r["band"] == "AB" and r["numerator"].startswith("SEL n=64")))]
    assert len(b) == 2, b
    t1 = next(float(r["ratio"]) for r in b if r["band"] == "T1")
    ab = next(float(r["ratio"]) for r in b if r["band"] == "AB")
    a = " ".join(open(tex("iclr_2027.tex"), encoding="utf-8").read().split())
    a = a[a.index("begin{abstract}"):a.index("end{abstract}")]
    # v10 wording: `it costs 21.8x the meter's decode time unbatched and X and Y batched at the
    # authors' two 70B pairs`
    # the byte-level ratio is printed at the body's precision ($0.061\times$, Table 4), so every number
    # in the abstract is literally in the body (test_abstract_consistency)
    assert f"${t1:.2f}\\times$ and ${ab:.3f}\\times$ batched at the authors' two $70$B pairs" in a
    assert "$21.8\\times$ the meter's decode time unbatched" in a


def test_the_cotaeval_infringement_paragraph_reads_its_csv():
    """feat-193: 'Not run' became the scored split. Every claim in the paragraph, the two that argue
    against us included (the risky model IS closer than the anchor, so the registered 'uninformative'
    wording may not be used; the adversarial oracle beats the risky model, so I3 failed), is re-read
    from results/cotaeval_infringement.csv, and the Limitations clause must name both halves."""
    r = {x["arm"]: x for x in rows("cotaeval_infringement.csv")}
    t = body("appendix_selection.tex")
    assert "Not run:} CoTaEval's infringement half" not in t
    # v10 made the bold lead-in a \paragraph heading; locate it by its words
    i = t.index("CoTaEval's infringement half cannot tell a defence from none")
    p = t[i:t.index("infringement.md})", i)]
    n = int(r["risky"]["n"])
    assert n == 500 and "($500$ of its $1{,}000$ items" in p
    arms = [x for x in r.values() if " - " not in x["arm"]]
    assert len(arms) == 14 and all(x["event_count"] == "0" for x in arms), \
        "an arm reached ROUGE-L >= 0.5; 'no arm reaches the event on any item' is false"
    assert f"($0/{n}$ each, Wilson upper bound ${float(r['risky']['event_hi']):.4f}$)" in p

    def lvl(a):
        return f"${float(r[a]['rouge_l']):.4f}$ $[{float(r[a]['rouge_l_lo']):.4f}, {float(r[a]['rouge_l_hi']):.4f}]$"

    def dif(a):
        return f"${float(r[a]['rouge_l']):+.4f}$ $[{float(r[a]['rouge_l_lo']):+.4f}, {float(r[a]['rouge_l_hi']):+.4f}]$"

    ri, an = r["risky"], r["anchor"]
    assert f"({lvl('risky')} against {lvl('anchor')})" in p
    assert float(ri["rouge_l_lo"]) > float(an["rouge_l_hi"]), "the risky model is no longer closer than the anchor"
    fires = ri["event_count"] == "0" and float(an["rouge_l_lo"]) <= float(ri["rouge_l"]) <= float(an["rouge_l_hi"])
    assert fires == ("uninformative" in p), "the registered wording must appear exactly when its rule fires"
    for arm, sign, words in (("sel64 - anchor", 0, "no closer than the anchor alone"),
                             ("sel64 - risky", -1, "further than the risky model"),
                             ("sel64_ne - risky", -1, "over non-empty draws alone"),
                             ("met_k10 - risky", 0, "the meter at $k=10$ is the risky model"),
                             ("oracle64 - risky", 1, "closer than the risky model's one draw")):
        lo, hi = float(r[arm]["rouge_l_lo"]), float(r[arm]["rouge_l_hi"])
        assert (1 if lo > 0 else -1 if hi < 0 else 0) == sign, (arm, "the reading no longer matches its words")
        j = p.index(dif(arm))
        assert words in p[max(0, j - 120):j + 40], (arm, words)
    assert "We predicted that even an adversarial scorer would stay below the risky model, and it does not" in p
    assert round(float(r["sel64"]["empty_served"]) * n) == 1 and "an empty draft on one item" in p
    close = body("iclr_closing.tex")
    s = close[close.index("F1 on CoTaEval"):]
    s = s[:s.index("}).") + 3]
    assert "infringement event not even the risky model reaches" in s, "the Conclusion names only one half"
