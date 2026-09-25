"""v14 (2026-09-25): the review items first skipped and then run. Every number the paper quotes from them is
read back out of the CSV that produced it, and every claim ABOUT those numbers is checked against their shape."""
import csv
import os
import re

from manuscript import ROOT, body

R = os.path.join(ROOT, "results")


def rows(name):
    return list(csv.DictReader(open(os.path.join(R, name))))


def f(x, nd):
    return f"{float(x):.{nd}f}"


# ---- review 2 Q3: the meter as speculative decoding ------------------------------------------------------------

def spec():
    a = {r["k"]: r for r in rows("speculative_acceptance_8b.csv") if r["quantity"] == "acceptance"}
    b = rows("speculative_acceptance_70b.csv")[0]
    return a, b


def test_the_speculative_paragraph_quotes_its_csv():
    a, b = spec()
    txt = body("appendix_selection.tex")
    i = txt.index("\\label{app:speculative}")
    par = txt[i:txt.index("\\paragraph", i + 10) if "\\paragraph" in txt[i + 10:i + 4000] else i + 4000]
    for want in (f(a["10"]["alpha_mean"], 3), f(a["3"]["alpha_mean"], 3), f(a["0.5"]["alpha_mean"], 3),
                 f(a["10"]["tokens_per_risky_forward"], 2), f(a["10"]["selection_ratio"], 3),
                 f(a["0.5"]["selection_ratio"], 3), f(a["10"]["alpha_star"], 3),
                 f(a["10"]["selection_ratio_committed"], 3), f(b["selection_ratio_at_alpha1"], 3),
                 f(a["10"]["pstar_median_abs_err"], 4)):
        assert f"${want}" in par, want
    assert a["10"]["best_g"] == "1" and "($g=1$)" in par


def test_the_main_text_bound_is_the_csvs_and_does_not_erase_the_advantage():
    a, b = spec()
    m70 = {r["k"]: r for r in rows("speculative_acceptance_70b_measured.csv") if r["quantity"] == "acceptance"}
    txt = body("experiments.tex")
    # "at best": the larger of the two measured budgets that bracket the timed k=10, at each pair
    s70 = max(float(m70[k]["selection_ratio"]) for k in ("20", "3"))
    s8 = max(float(a[k]["selection_ratio"]) for k in ("10", "3"))
    assert f"at best ${s70:.2f}\\times$ and ${s8:.2f}\\times$" in txt
    assert s70 <= float(b["selection_ratio_at_alpha1"])            # measured sits inside the alpha=1 bound
    app = body("appendix_selection.tex")
    for want in (f"${float(m70['20']['alpha_mean']):.3f}$ at $k=20$", f"${float(m70['3']['alpha_mean']):.3f}$ at $k=3$",
                 f"($g={m70['20']['best_g']}$)", f"to ${float(m70['20']['selection_ratio']):.3f}\\times$"):
        assert want in app, want
    # "lift these to at most" is a claim that selection still wins: both bounds must stay below parity,
    # and the 70B bound must hold at PERFECT acceptance, which is why that pair needs no measured alpha
    assert s70 < 1 and s8 < 1
    assert b["alpha_star"] == ""                                   # no acceptance reaches parity at g <= 8
    assert float(a["10"]["alpha_mean"]) < float(a["10"]["alpha_star"])   # measured below the 8B break-even


# ---- review 2 Q6: where the gain lies ---------------------------------------------------------------------------

def strata():
    return {(r["task"], r["stratum"]): r for r in rows("utility_surprisal.csv")}


def test_the_gain_location_paragraph_quotes_its_csv():
    s = strata()
    txt = body("appendix_selection.tex")
    i = txt.index("\\label{app:gainwhere}")
    par = txt[i:i + 2600]
    for task, n_hi, n_lo in (("gsm8k", 439, 61), ("triviaqa", 287, 213)):
        hi = s[(task, "pi_ho >= 1/32, all (S <= log n)")]
        lo = s[(task, "pi_ho = 0 (S > log n)")]
        assert (int(hi["questions"]), int(lo["questions"])) == (n_hi, n_lo)
        band = f"${float(hi['gain']):+.3f}$ $[{float(hi['gain_lo95']):+.3f}, {float(hi['gain_hi95']):+.3f}]$"
        assert band in par, band
        assert f"${n_hi}$" in par and f"${n_lo}$" in par
    j = s[("judged headline (judge B)", "served S > log 64")]
    band = f"${float(j['gain']):+.3f}$ $[{float(j['gain_lo95']):+.3f}, {float(j['gain_hi95']):+.3f}]$"
    assert band in par and f"${int(j['questions'])}$" in par
    assert f"${float(s[('judged headline (judge B)', 'served S, nats')]['s_median']):.1f}$" in par


def test_gains_only_where_log_n_reaches_S_is_true_of_the_numbers():
    """The main text says a vote gains only where log n >= S. That is a claim about a set of numbers
    (caution (ai)): it holds iff the held-out S > log n stratum gains exactly nothing on both tasks."""
    s = strata()
    for task in ("gsm8k", "triviaqa"):
        assert float(s[(task, "pi_ho = 0 (S > log n)")]["gain"]) == 0.0
        assert float(s[(task, "pi_ho >= 1/32, all (S <= log n)")]["share_of_total_gain"]) == 1.0
    assert re.search(r"a vote gains only where \$\\log n \\ge S\$", body("iclr_closing.tex"))
    # and the judged side says the opposite, which the paragraph must not blur: all of it where S > log 64
    assert float(s[("judged headline (judge B)", "served S <= log 64")]["gain"]) == 0.0


# ---- feat-211: CP-k's rejection rule, run -------------------------------------------------------------------------

def pass_(tag):
    return {(r["quantity"], r["arm"]): r for r in rows(f"matched_h2h_{tag}.csv")}


def band4(r):
    return f"${float(r['value']):+.{dp(r['value'])}f}$ $[{float(r['lo95']):+.{dp(r['lo95'])}f}, {float(r['hi95']):+.{dp(r['hi95'])}f}]$"


def dp(x):
    """The decimals the paper prints a CSV value at: its own, with trailing zeros kept to at least 3."""
    s = str(x).split(".")[-1] if "." in str(x) else ""
    return max(3, len(s))


def test_the_related_work_sentence_is_the_rule_as_run():
    a = {r["certificate_nats"]: r for r in rows("cpk_baseline.csv")}
    rw = body("related_work_v4.tex")
    assert float(a["4.1589"]["served_risky_pct"]) == 0.0
    assert f"on ${float(a['83.178']['served_risky_pct']):.1f}\\%$ at $83$ nats" in rw
    assert "serves the risky model on no prompt at $\\log 64$" in rw
    # "beat it at every matched certificate" is a claim about three intervals under two judges (C1-C4)
    assert "beat it at every matched certificate" in rw
    for j in ("cpk_B_hostb", "cpk_G"):
        P = pass_(j)
        for d in ("blk10n64 - cpk_83.18", "blk25n64 - cpk_33.27", "sel_n64 - cpk_4.16"):
            assert P[("difference", d)]["reading"] == "CONFIRMED", (j, d)


def test_the_cpk_appendix_quotes_its_csvs():
    B = pass_("cpk_B_hostb")
    app = body("appendix_onset.tex")
    i = app.index("\\label{app:cpk}")
    par = app[i:app.index("\\label{tab:cpk}", i)]
    for d in ("sel_n64 - cpk_4.16", "blk25n64 - cpk_33.27", "blk10n64 - cpk_83.18", "cpk_83.18 - pw_83.18",
              "cpk_33.27 - pw_33.27"):
        assert band4(B[("difference", d)]) in par, d
    for d in ("cpk_83.18 - pw_83.18", "cpk_33.27 - pw_33.27"):
        assert B[("difference", d)]["reading"] == "UNRESOLVED"          # "it is unresolved"
    leak = {r["certificate_nats"]: r for r in rows("cpk_extraction.csv")}
    assert all(float(leak[c]["nv_recall_mean"]) == 0.0 for c in ("4.1589", "33.271", "83.178", "159.83", "250.0"))
    assert float(leak["400.0"]["nv_recall_mean"]) > 0 and "through $250$ nats" in par   # nothing only THROUGH 250
    assert f"(${float(leak['600.0']['nv_recall_mean']):.4f}$ recall" in par
    assert f"reads ${float(leak['memoriser alone']['nv_recall_mean']):.4f}$" in par
    med = [r for r in rows("cpk_baseline.csv") if r["certificate_nats"].startswith("median R")][0]
    assert f"median $R$ is ${float(med['kappa']):.1f}$ nats" in par
    # C5 failed and the paper says so: accepted draws are as long as all draws
    a = {r["certificate_nats"]: r for r in rows("cpk_baseline.csv")}
    assert float(a["83.178"]["accepted_median_tokens"]) >= float(a["83.178"]["all_draws_median_tokens"])
    assert "They are not" in par


def test_the_cpk_table_is_its_csvs():
    B, G = pass_("cpk_B_hostb"), pass_("cpk_G")
    a = {r["certificate_nats"]: r for r in rows("cpk_baseline.csv")}
    leak = {r["certificate_nats"]: r for r in rows("cpk_extraction.csv")}
    from manuscript import caption_of  # noqa: F401  (the table is found by its label below)
    app = body("appendix_onset.tex")
    tab = app[app.index("\\label{tab:cpk}"):app.index("\\end{tabular}", app.index("\\label{tab:cpk}"))]
    for c, name in (("4.1589", "4.16"), ("33.271", "33.27"), ("83.178", "83.18"), ("159.83", "159.83")):
        g = f"cpk_{name} - anchor_k0"
        b = B[("gain", g)]
        row = (f"${float(c):.2f}$ & ${float(a[c]['kappa']):.2f}$ & ${float(a[c]['served_risky_pct']):.1f}\\%$ & "
               f"${float(b['value']):+.4f}$ $[{float(b['lo95']):+.4f}, {float(b['hi95']):+.4f}]$ & "
               f"${float(G[('gain', g)]['value']):+.4f}$ & ${float(leak[c]['nv_recall_mean']):.4f}$ \\\\")
        assert row in tab, row


# ---- feat-212: scorer family by judge family ---------------------------------------------------------------------

WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def test_the_factorial_table_is_its_csv():
    app = body("appendix_selection.tex")
    i = app.index("\\label{tab:factorial}")
    tab = app[i:app.index("\\end{tabular}", i)]
    for r in rows("scorer_judge_factorial.csv"):
        cells = [f"${float(r[k]):+.4f}$ $[{float(r[k + '_lo']):+.4f}, {float(r[k + '_hi']):+.4f}]$"
                 for k in ("S", "D3_qwen", "D3_gemma")]
        assert f"{r['judge']} & " + " & ".join(cells) + " \\\\" in tab, r["judge"]


def test_the_factorial_counts_and_claims_are_the_csvs():
    fac = {r["judge"]: r for r in rows("scorer_judge_factorial.csv")}
    n_g = sum(r["D3_gemma_reading"] == "CONFIRMED" for r in fac.values())
    n_q = sum(r["D3_qwen_reading"] == "CONFIRMED" for r in fac.values())
    prefer_q = [j for j, r in fac.items() if r["S_reading"] == "CONFIRMED"]
    exp, app = body("experiments.tex"), body("appendix_selection.tex")
    assert f"{WORDS[n_g]} of six judges resolve it with either scorer" in exp
    assert n_q >= n_g                                         # "with either scorer": both counts at least n_g
    assert prefer_q == ["B"] and "only~B prefers the Qwen reward's drafts" in exp
    par = app[app.index("\\label{app:factorial}"):app.index("\\label{tab:factorial}")]
    assert "Only judge~B prefers them" in par
    assert fac["G"]["S_reading"] == "REFUTED" and "the gemma-scored drafts win" in par
    assert all(fac[j]["S_reading"] == "UNRESOLVED" for j in "CDEF")
    assert f"headline resolves under {WORDS[n_g]} of six judges, against {WORDS[n_q]}" in par
    assert fac["B"]["D3_gemma_reading"] != "CONFIRMED"         # "B ... is one of the two that do not resolve it"
    assert sum(r["D3_gemma_reading"] != "CONFIRMED" for r in fac.values()) == 2
    for j in ("D", "F"):
        assert f"${float(fac[j]['D3_gemma']):+.4f}$".replace("0$", "$") in par or \
               f"${float(fac[j]['D3_gemma']):+.4f}$" in par
    # the same-host reproduction claim: every same-host pass equals the committed panel's D3
    for j in "CDEFG":
        assert abs(float(fac[j]["D3_qwen"]) - float(fac[j]["D3_qwen_committed"])) < 1e-9, j
    assert f"reads ${float(fac['B']['D3_qwen']):+.3f}$ here against ${float(fac['B']['D3_qwen_committed']):+.4f}$" in par
