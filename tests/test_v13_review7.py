"""Guards for the seventh review round (v13, 2026-09-25): feat-210's matched-certificate pass, the
windowed meter, and the numbers the review audit re-derived (frontier ratio, window thresholds, CP-k,
per-work audit cost, D3 against direct levels, the 70B opponent, judge validity).

Every expected value is read from the CSV that produced it, never typed here (caution (ag)), and every
claim ABOUT a set of numbers is checked against the numbers' shape (cautions (ai), (ao)).
"""
import csv
import math
import os
import re
import statistics

from manuscript import ROOT, body, caption_of, tex

R = os.path.join(ROOT, "results")


def rows(name):
    with open(os.path.join(R, name), newline="") as f:
        return list(csv.DictReader(f))


def h2h(tag):
    """(quantity, arm) -> row of results/matched_h2h_matched_<tag>.csv"""
    return {(r["quantity"], r["arm"]): r for r in rows(f"matched_h2h_matched_{tag}.csv")}


def f(x):
    return float(x)


def printed(value, text):
    """Does `text` print `value` at SOME precision that rounds from it once (caution (j))?"""
    for m in re.finditer(r"[+-]?\d[\d{},]*\.?\d*", text):
        s = m.group(0).replace("{,}", "").replace(",", "")
        try:
            v = float(s)
        except ValueError:
            continue
        d = len(s.split(".")[1]) if "." in s else 0
        if round(abs(value), d) == abs(v) and (v == 0 or (v > 0) == (value > 0) or not s.startswith(("+", "-"))):
            return True
    return False


def band_str(v, lo, hi, d=4):
    return f"${v:+.{d}f}$ $[{lo:+.{d}f}, {hi:+.{d}f}]$"


def table(label):
    """Rows of the tabular under \\label{<label>} as lists of cells, $ stripped, blank mechanism
    cells inheriting the row above."""
    live = " ".join(body("experiments.tex", "appendix_onset.tex").split())
    i = live.index("\\label{%s}" % label)
    t = live[live.index("\\midrule", i): live.index("\\end{tabular}", i)]
    out, last = [], ""
    for raw in t.replace("\\midrule", "").replace("\\bottomrule", "").split("\\\\"):
        cells = [c.strip().replace("$", "") for c in raw.split("&")]
        if len(cells) < 2:
            continue
        if cells[0]:
            last = cells[0]
        else:
            cells[0] = last
        out.append(cells)
    return out


LOG64 = math.log(64)
# the matched tables' arms, keyed by the gain they print (arm minus its own control)
GAIN = {
    "anchored decoding, k=0.5": "met_k0.5 - anchor_k0",
    "budget up front": "frontpw_4.16 - win_0",
    "best of 64": "sel_n64 - sel_n1",
    "best of 64, never empty": "nonempty - sel_n1",
    "installments, L=25": "blk25n64 - blk200n1",
    "installments, L=10": "blk10n64 - blk200n1",
}


def expected_certificates(mech, span):
    """(span, output) certificates in nats from the mechanism's own definition, T=200, w=50."""
    if mech.startswith("anchored decoding"):
        return None, 0.5 * 200
    if mech in ("budget up front", "best of 64", "best of 64, never empty"):
        return None, LOG64
    if mech.startswith("per-token pathwise"):
        return None, None  # read from the arm's k below
    if mech.startswith("windowed"):
        W = next(float(r["W"]) for r in rows("windowed_arms.csv") if r["arm"] == f"win_{span}")
        return W, math.ceil(200 / 50) * W
    L = int(re.search(r"L=(\d+)", mech).group(1))
    return (math.ceil(50 / L) + 1) * LOG64, math.ceil(200 / L) * LOG64


def arm_for(mech, span, output):
    if mech in GAIN:
        return GAIN[mech]
    if mech.startswith("per-token pathwise"):
        return {"33.27": "pw_33.27 - win_0", "83.18": "pw_83.18 - win_0"}[output]
    if mech.startswith("windowed"):
        return f"win_{span} - win_0"
    raise AssertionError(f"unknown mechanism row {mech!r}")


def leak_expected(mech, span):
    if mech.startswith("windowed"):
        W = float(span)
        for r in rows("windowed_leakage.csv"):
            if r["config"] == "plain" and abs(float(r["W"]) - W) < 0.01:
                return float(r["nv_recall_mean"])
        raise AssertionError(f"no plain leakage row at W={W}")
    if mech == "best of 64":  # the caption: "for selection, the best of its 64 draws"
        return statistics.mean(float(r["oracle_recall_n64"])
                               for r in rows("selection_extraction_feat210_per_passage.csv"))
    return None


def check_matched_table(label, n_rows):
    B, G = h2h("plain_B"), h2h("plain_G")
    got = table(label)
    assert len(got) == n_rows, f"{label}: {len(got)} rows, expected {n_rows}"
    seen = set()
    for mech, span, output, gainB, gainG, leak in got:
        mech_n = re.sub(r"\s+", " ", mech.replace("\\,", " ")).strip()
        arm = arm_for(mech_n, span, output)
        seen.add(arm)
        rb, rg = B[("gain", arm)], G[("gain", arm)]
        assert gainB.replace(" ", "") == band_str(f(rb["value"]), f(rb["lo95"]), f(rb["hi95"])).replace("$", "").replace(" ", ""), (label, arm, gainB)
        assert printed(f(rg["value"]), gainG), (label, arm, gainG, rg["value"])
        s_exp, o_exp = expected_certificates(mech_n, span)
        if mech_n.startswith("per-token pathwise"):
            k = {"33.27": 0.166355, "83.18": 0.415888}[output]
            o_exp = 200 * k
        assert printed(o_exp, output), (label, mech_n, output, o_exp)
        if s_exp is not None:
            assert printed(s_exp, span), (label, mech_n, span, s_exp)
        else:
            assert span == "---", (label, mech_n, span)
        le = leak_expected(mech_n, span)
        if le is None:
            assert leak == "---", (label, mech_n, leak)
        else:
            assert printed(le, leak), (label, mech_n, leak, le)
    return seen


def test_the_full_matched_table_rounds_from_both_judge_passes():
    seen = check_matched_table("tab:matchedfull", 13)
    assert len(seen) == 13


def test_the_compact_matched_table_rounds_from_both_judge_passes():
    seen = check_matched_table("tab:matched", 9)
    # the compact table keeps the arms the text argues from
    for arm in ("met_k0.5 - anchor_k0", "frontpw_4.16 - win_0", "sel_n64 - sel_n1",
                "blk10n64 - blk200n1", "win_4.16 - win_0", "win_125 - win_0"):
        assert arm in seen, arm


def test_the_windowed_table_rounds_from_leakage_and_the_chat_passes():
    lk = {(r["config"], round(float(r["W"]), 2)): float(r["nv_recall_mean"]) for r in rows("windowed_leakage.csv")}
    cB, cG = h2h("chat_B"), h2h("chat_G")
    cap = caption_of("tab:windowed")
    alone = {c: lk[(c, -1.0)] for c in ("plain", "chat")}
    assert f"${alone['plain']:.4f}$" in cap and f"${alone['chat']:.4f}$" in cap
    for W, plain, chat, jB, jG in table("tab:windowed"):
        w = round(float(W), 2)
        assert printed(lk[("plain", w)], plain), (W, plain)
        if chat == "---":
            assert ("chat", w) not in lk, W
        else:
            assert printed(lk[("chat", w)], chat), (W, chat)
        name = {4.16: "winc_4.16", 12.48: "winc_12.48", 24.95: "winc_24.95", 40.0: "winc_40", 125.0: "winc_125"}.get(w)
        key = ("difference", f"sel_n64 - {name}")
        if jB == "---":
            assert key not in cB, W
            continue
        for cell, src in ((jB, cB[key]), (jG, cG[key])):
            assert cell.replace(" ", "") == band_str(f(src["value"]), f(src["lo95"]), f(src["hi95"])).replace("$", "").replace(" ", ""), (W, cell)


def onset(config):
    grid = sorted((float(r["W"]), float(r["nv_recall_mean"])) for r in rows("windowed_leakage.csv")
                  if r["config"] == config and float(r["W"]) > 0)
    return next(W for W, rec in grid if rec >= 0.01)


def test_the_leakage_sentence_quotes_the_windowed_onsets():
    txt = body("experiments.tex")
    s = (f"reaches $0.01$ at ${onset('plain'):g}$ nats a window, ${onset('chat'):g}$ through the chat template")
    assert s in txt, s


def test_every_matched_certificate_difference_backs_the_bold_claim():
    """'Selection beats every meter at every matched certificate' (Section 4) and the abstract's
    'At matched certificates, best-of-64 and its blockwise variant beat every meter we ran' stand only
    while P1, P3 and P4 are all CONFIRMED under the pre-specified judge."""
    B = h2h("plain_B")
    matched = ["sel_n64 - win_4.16", "sel_n64 - frontpw_4.16", "blk10n64 - pw_83.18",
               "blk25n64 - pw_33.27", "blk10n64 - win_24.95", "blk25n64 - win_12.48"]
    assert all(B[("difference", a)]["reading"] == "CONFIRMED" for a in matched)
    assert "Selection beats every meter at every matched certificate" in body("experiments.tex")
    ab = " ".join(open(tex("iclr_2027.tex"), encoding="utf-8").read().split())
    assert "At matched certificates, best-of-$64$ and its blockwise variant beat every meter we ran" in ab


def test_the_failed_prediction_is_stated_where_the_windowed_meter_draws_level():
    """P2 at W=125 was wrong: the outcome rule fixed in advance says the paper states the tie."""
    B = h2h("plain_B")
    tie = B[("difference", "sel_n64 - win_125")]
    assert tie["reading"] == "UNRESOLVED"
    txt = body("experiments.tex")
    assert ("draws level only at $W = 125$ (" + band_str(f(tie["value"]), f(tie["lo95"]), f(tie["hi95"]), 4)
            .replace("$+0.0220$", "$+0.022$") + "; we had predicted selection would still lead)") in txt
    # and 'trails selection up to 40 nats a window' (intro) holds at every W <= 40 and not above
    for W in ("4.16", "24.95", "40"):
        assert B[("difference", f"sel_n64 - win_{W}")]["reading"] == "CONFIRMED", W
    assert "trails selection up to $40$ nats a window" in body("iclr_intro.tex")


def test_judge_G_signs_and_resolutions_match_the_sentence():
    B, G = h2h("plain_B"), h2h("plain_G")
    nine = ["sel_n64 - win_4.16", "sel_n64 - frontpw_4.16", "sel_n64 - win_24.95", "sel_n64 - win_40",
            "sel_n64 - win_125", "blk10n64 - pw_83.18", "blk25n64 - pw_33.27", "blk10n64 - win_24.95",
            "blk25n64 - win_12.48"]
    same = sum((f(B[("difference", a)]["value"]) > 0) == (f(G[("difference", a)]["value"]) > 0) for a in nine)
    unresolved = [a for a in nine if G[("difference", a)]["reading"] != "CONFIRMED"]
    assert same == 9 and len(unresolved) == 1
    assert "Judge~G gives all nine registered differences the same sign and resolves all but one" in body("experiments.tex")


def posthoc():
    out = {}
    for r in rows("windowed_meter.csv"):
        if r["band"] == "post hoc":
            m = re.match(r"(.*) \((\w+)\), empty texts scored as losses", r["quantity"])
            out[(m.group(2), m.group(1))] = r
    return out


def test_empties_as_losses_sentences_match_the_post_hoc_rows():
    P = posthoc()
    plain = [a for (t, a) in P if t == "plain_B"]
    heldG = {a for a in plain if f(P[("plain_G", a)]["lo95"]) > 0}
    heldB = {a for a in plain if f(P[("plain_B", a)]["lo95"]) > 0}
    assert set(plain) - heldG == {"sel_n64 - win_125"}
    assert heldB == {"sel_n64 - win_4.16", "blk10n64 - pw_83.18", "blk25n64 - pw_33.27"}
    exp = body("experiments.tex")
    assert ("counted as losses the differences hold under judge~G at every budget but $W=125$, and under "
            "judge~B against the per-token meters and the windowed meter at $\\log 64$ only") in exp
    app = body("appendix_selection.tex")
    i = app.index("Counted instead as losses (post hoc)")
    s = app[i: i + 420]
    assert f"$+{f(P[('plain_B', 'sel_n64 - win_4.16')]['value']):.3f}$" in s
    assert f"$+{f(P[('plain_B', 'sel_n64 - win_24.95')]['value']):.3f}$ each" in s
    assert f(P[("plain_B", "sel_n64 - win_24.95")]["value"]) == f(P[("plain_B", "sel_n64 - win_40")]["value"])


def test_the_judge_validity_paragraph_rounds_from_the_pass():
    B, G = h2h("plain_B"), h2h("plain_G")
    app = body("appendix_selection.tex")
    i = app.index("\\label{app:judgevalidity}")
    s = app[i: app.index("\\paragraph", i + 30)]
    known = B[("gain", "opp_r1 - anchor_k0")]
    assert band_str(f(known["value"]), f(known["lo95"]), f(known["hi95"]), 3).replace("+0.065$", "+0.065$") in s \
        or f"$+{f(known['value']):.3f}$ $[{f(known['lo95']):+.4f}, {f(known['hi95']):+.3f}]$" in s
    assert f"$+{f(G[('gain', 'opp_r1 - anchor_k0')]['value']):.4f}$ under~G" in s
    cp = B[("difference, consistent pair", "sel_n64 - metered_k10")]
    c4 = B[("difference, consistent four", "sel_n64 - metered_k10")]
    assert f"on the ${int(cp['n'])}$ prompts where both" in s
    assert f"$+{f(cp['value']):.3f}$ $[{f(cp['lo95']):+.3f}, {f(cp['hi95']):+.3f}]$" in s
    assert c4["reading"] == "UNRESOLVED" and f"on the ${int(c4['n'])}$ where all four arms do, too few to resolve" in s
    ne = G[("difference", "nonempty - metered_k10")]
    assert f"$+{f(ne['value']):.4f}$ $[{f(ne['lo95']):+.3f}, {f(ne['hi95']):+.4f}]$" in s
    # the k=0.5 null: MDE and the 90% interval from the scored sensitivity row
    sens = next(r for r in rows("windowed_meter.csv") if r["quantity"].startswith("sensitivity of met_k0.5"))
    mde = float(re.search(r"MDE80 ([\d.]+)", sens["reading"]).group(1))
    assert f"against $+{mde:.4f}$" in s
    assert f"$[{f(sens['lo90']):+.4f}, {f(sens['hi90']):+.3f}]$" in s
    assert "within 0.051: yes" in sens["reading"]
    # the empties: B and G levels of selection's empty picks
    lb = B[("level", "sel_n64")]
    lg = G[("level", "sel_n64")]
    assert f"${int(lb['n_empty'])}$ empty picks at ${f(lb['level_empty']):.3f}$" in s
    assert f"G at ${f(lg['level_empty']):.3f}$, above the arm's own ${f(lg['value']):.3f}$" in s


def test_the_consistency_ranges_are_the_measured_ones():
    cons = {}
    for tag, j in (("plain_B", "B"), ("plain_G", "G")):
        vals = [f(r["consistency"]) for (q, a), r in h2h(tag).items()
                if q == "level" and a in ("sel_n64", "sel_n1", "metered_k10", "anchor_k0")]
        cons[j] = (min(vals), max(vals))
    exp = body("experiments.tex")
    assert f"agrees with itself on ${round(100 * cons['G'][0])}$ to ${round(100 * cons['G'][1])}\\%$" in exp
    app = body("appendix_selection.tex")
    assert f"${cons['B'][0]:.3f}$ to ${cons['B'][1]:.3f}$ over the four arms for B" in app
    assert f"${cons['G'][0]:.3f}$ to ${cons['G'][1]:.3f}$ for G" in app


def test_the_frontier_ratio_reads_off_the_order_averaged_instrument():
    fr = {r["group"]: r for r in rows("frontier_ratio.csv")}
    a = fr["all"]
    app = body("appendix_onset.tex")
    i = app.index("\\label{app:frontierratio}")
    s = app[i: app.index("\\paragraph", i + 30)]
    pi = f(a["pi_wins_both_orders"])
    assert f"${100 * pi:.1f}\\%$ of prompts" in s
    assert f"$\\log(1/\\pi)$ is ${f(a['log_inv_pi']):.2f}$ pooled" in s
    assert f"at least ${f(a['ratio_at_u_max']):.1f}\\times$" in s
    assert f"${f(fr['neutral']['ratio_at_u_max']):.1f}\\times$ and ${f(fr['factual']['ratio_at_u_max']):.1f}\\times$ by class" in s
    assert fr["creative"]["log_inv_pi"] == "inf" and "infinite for stories" in s
    ratios = [f(a[f"ratio_k{k}"]) for k in ("1", "3", "10")]
    assert ratios == sorted(ratios, reverse=True)
    assert "ratio of $" + "$, $".join(f"{round(x):,}".replace(",", "{,}") for x in ratios[:2]) + "$ and $" \
        + f"{round(ratios[2]):,}".replace(",", "{,}") + "$" in s
    assert a["ratio_k0.5"] == "" and "at $k=0.5$ the meter gains nothing and the ratio is unbounded" in s
    d = {r["arm"]: r for r in rows("frontier_distance_h2h_deecho.csv")}
    assert f"${f(d['selection, n=64']['over_frontier']):.1f}\\times$ the frontier" in s
    assert f"{round(f(d['metered decoder, k=10']['over_frontier'])):,}".replace(",", "{,}") in s
    fro = body("frontier.tex")
    # v16 (2026-09-26, review 8 W1): the ratio restates the 171.3-nat spend in units of the utility's price, and
    # Proposition 4's second clause follows from its first; the body now gives the price and points here
    assert "costs $0.023$ nats by Lemma~\\ref{thm:nfl} (Appendix~\\ref{app:frontierratio})" in fro
    assert "the proposition's second clause follows from its first" in s
    prf = body("appendix_proofs.tex")
    assert f"spends ${f(a['ratio_at_u_max']):.1f}$ times" in prf and f"${100 * pi:.1f}\\%$ of prompts" in prf


def test_figure_two_quotes_the_decoded_length_thresholds():
    w = rows("onset_window_threshold.csv")
    tgt = [f(r["median_passage_threshold_over_s"]) for r in w]
    win = [f(r["window_threshold_over_s"]) for r in w]
    for r in w:  # the two thresholds are n/T and 50/T of the same decoded length
        assert abs(f(r["window_threshold_over_s"]) - 50 / f(r["T_decoded"])) < 1e-3
        assert abs(f(r["median_passage_threshold_over_s"]) - f(r["n_target_median"]) / f(r["T_decoded"])) < 1e-3
    s = f"${min(tgt):.2f}$ to ${max(tgt):.2f}$ of $s(x)$"
    t = f"${min(win):.2f}$ to ${max(win):.2f}$ of it"
    fro = body("frontier.tex")
    assert s in fro and t in fro
    tiny = next(r for r in w if r["pair"].startswith("TinyComma"))
    assert f"${int(tiny['T_decoded'])}$" in caption_of("fig:horns")
    # 'it never excluded what was extracted': every onset sits far above a window's void point
    assert all(f(r["onset_over_s"]) > 4 * f(r["window_threshold_over_s"]) for r in w)
    # and the appendix counts the pairs whose onset clears the median target's own void point
    below = sum(f(r["median_passage_threshold_over_s"]) < f(r["onset_over_s"]) for r in w)
    words = {7: "seven", 8: "eight", 9: "nine"}
    gaps = [f(r["onset_over_s"]) - f(r["median_passage_threshold_over_s"]) for r in w]
    above = [-g for g in gaps if g < 0]
    app = body("appendix_onset.tex")
    assert (f"below the onset at {words[below]} of {words[len(w)]} pairs (by ${min(g for g in gaps if g > 0):.3f}$ "
            f"to ${max(gaps):.3f}$)") in app
    assert all(f"${x:.3f}$ above" in app for x in above)


def test_the_cpk_sentence_rounds_from_its_csv():
    c = {r["quantity"]: r for r in rows("cpk_certificate.csv")}
    half = f(c["certificate at acceptance 0.5"]["value"])
    one = f(c["certificate at acceptance 0.01"]["value"])
    # v14 (feat-211): the related-work sentence now reports the rule as run, as its registration fixed, and
    # this certificate-only arithmetic moved with it into Appendix app:cpk (caution (al))
    rw = body("appendix_onset.tex")
    assert f"${half:.2f}$ nats at half acceptance, above a window's $159.8$" in rw and half > 159.8
    assert f"${one:.2f}$ at one output in a hundred" in rw


def test_the_per_work_audit_price_rounds_from_its_csv():
    a = {(int(r["n"]), float(r["eps"]), int(r["works"])): r for r in rows("audit_cost.csv")}
    one = a[(64, 0.01, 1)]
    cat64, cat8 = a[(64, 0.01, 100000)], a[(8, 0.01, 100000)]
    assert int(one["draws_per_work"]) == math.ceil(3 * 64 / 0.01)
    sel = body("selection.tex")
    assert "per work, $" + f"{int(one['draws_per_work']):,}".replace(",", "{,}") + "$ draws" in sel
    assert f"about ${f(one['gpu_hours']):.2f}$ A100-hours" in sel
    assert f"${round(f(cat64['gpu_hours'])):,}$ for $10^5$ works".replace(",", "{,}") in sel
    app = body("appendix_selection.tex")
    assert (f"a catalogue of $10^5$ works costs ${round(f(cat64['gpu_hours'])):,}$ and "
            f"${round(f(cat8['gpu_hours'])):,}$ GPU-hours").replace(",", "{,}") in app


def test_the_d3_paragraph_counts_its_own_rows():
    d = rows("d3_vs_direct.csv")
    same = sum(r["same_reading"] == "True" for r in d)
    assert all(f(r["controls_differ"]) >= 0 for r in d)  # D3 never kinder to selection
    differ = [r for r in d if r["same_reading"] != "True"]
    app = body("appendix_selection.tex")
    assert f"over all ${len(d)}$ rows of the table the two give the same reading on ${same}$" in app
    for r in differ:  # each named exception is a direct win that is a tie on gains
        assert r["direct_reading"] == "CONFIRMED" and r["d3_reading"] == "UNRESOLVED"
        assert f"$[{f(r['d3_lo95']):+.4f}, {f(r['d3_hi95']):+.4f}]$" in app
    pos = [f(r["controls_differ"]) for r in d if f(r["controls_differ"]) > 0]
    assert f"${min(pos):.3f}$ to ${max(pos):.4f}$ above the meter's anchor-alone" in app


def test_the_70B_opponent_paragraph_rounds_from_its_csv():
    o = {r["arm"]: r for r in rows("opponent_degeneracy.csv")}
    b, i, a = o["70B base, k=-1"], o["8B-Instruct, k=-1"], o["anchor alone, k=0"]
    app = body("appendix_selection.tex")
    j = app.index("The $70$B base as an opponent.")
    s = app[j: j + 900]
    assert f"${f(b['median_words']):.0f}$ words against the $8$B-Instruct's ${f(i['median_words']):.0f}$ and the anchor's ${f(a['median_words']):.0f}$" in s
    assert f"empty on ${f(b['empty_pct']):.1f}\\%$ (the anchor ${f(a['empty_pct']):.1f}\\%$)" in s
    assert f"${f(b['looping_pct']):.1f}\\%$ of responses against the anchor's ${f(a['looping_pct']):.1f}\\%$" in s


def test_the_abstract_follows_the_registered_outcome_rule_for_the_chat_regime():
    """feat-210's outcome rule: the abstract may name the windowed meter as the mechanism for the chat
    regime only if it wins there (P5) WITHOUT leaking at W=125. It leaks, so the abstract must say so."""
    lk = {(r["config"], float(r["W"])): float(r["nv_recall_mean"]) for r in rows("windowed_leakage.csv")}
    leaks = lk[("chat", 125.0)] >= 0.01
    ab = " ".join(open(tex("iclr_2027.tex"), encoding="utf-8").read().split())
    ab = ab[ab.index("\\begin{abstract}"): ab.index("\\end{abstract}")]
    sent = next(x for x in ab.split(". ") if "windowed meter wins" in x)
    if leaks:
        assert "leaks near-verbatim text" in sent and "$125$" in sent and "both judges prefer it" in sent
    cG = h2h("chat_G")[("difference", "sel_n64 - winc_125")]
    cB = h2h("chat_B")[("difference", "sel_n64 - winc_125")]
    assert cB["reading"] == cG["reading"] == "REFUTED"  # 'where both judges prefer it'


def test_the_query_caps_round_from_their_csv():
    """Review 1 Q7 / review 3 W9(b): the cap floor(S / log n) for the shortest protected events."""
    q = {(r["event"].split(",")[0], int(r["n"])): r for r in rows("query_caps.csv")}
    for (ev, n), r in q.items():
        if "window" in ev:
            assert int(r["cap_median"]) == math.floor(float(r["S_nats"]) / math.log(n)), (ev, n)
    s = " ".join(body("appendix_proofs.tex").split())
    i = s.index(r"For an untrusted scorer the cap is")
    sent = s[i: s.index("query\\_\\allowbreak caps.csv", i)]
    w10, w50, tq = ("10-token window", "50-token window", "TriviaQA answer")
    assert (f"${int(q[(w10, 8)]['cap_median'])}$ and ${int(q[(w10, 64)]['cap_median'])}$ for a $10$-token window of median "
            f"surprisal ${float(q[(w10, 8)]['S_nats']):.2f}$ nats") in sent
    assert f"${int(q[(w50, 8)]['cap_median'])}$ and ${int(q[(w50, 64)]['cap_median'])}$ for a $50$-token one" in sent
    assert (f"median ${float(q[(tq, 8)]['S_nats']):.2f}$ nats, ${int(float(q[(tq, 8)]['cap_median']))}$ and "
            f"${int(float(q[(tq, 64)]['cap_median']))}$, with ${100 * float(q[(tq, 8)]['frac_no_query']):.1f}\\%$ and "
            f"${100 * float(q[(tq, 64)]['frac_no_query']):.1f}\\%$ of answers allowing none") in sent
    # the n=64 window cap is Table 1's query horizon
    ct = {r["param"]: r for r in rows("certificate_table.csv")}
    assert int(ct["n=64"]["queries_before_vacuous"]) == int(q[(w50, 64)]["cap_median"])
