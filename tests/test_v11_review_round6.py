"""Guards for the v11 revision (2026-09-24, sixth review round).

Every number this round added to the paper rounds from a committed CSV, once (caution (j)), and each
claim ABOUT a set of numbers is checked against the set (caution (ai)):

  Table 2's "active" column      results/served_activity.csv   (analysis/served_activity.py)
  the windowed meter, Sec. 3     results/window_logratio.csv   (analysis/window_logratio.py)
  empty answers, Sec. 4 + App.   results/empty_as_loss.csv, results/nonempty_rule.csv
  prefix extension, Sec. 2 + App results/prefix_extension.csv  (feat-199, registered)
"""
import csv
import math
import os
import re

from manuscript import ROOT, body, caption_of, carries_band, tex

RES = os.path.join(ROOT, "results")


def _rows(name):
    return list(csv.DictReader(open(os.path.join(RES, name), encoding="utf-8")))


def _rounds(printed, value):
    d = len(printed.split(".")[1]) if "." in printed else 0
    return f"{value:.{d}f}" == f"{float(printed):.{d}f}"


# ------------------------------------------------------------------ Table 2, the "active" column

def test_table2_active_column_is_the_judged_trajectories_strict_blend_share():
    """A referee found Table 2's binding share disagreeing with Appendix G's beta; the column is now
    one definition on one population (strict blends, on exactly the trajectories the judge scored),
    and the caption must say so."""
    want = {(r["block"], float(r["k"])): 100 * float(r["active_share"]) for r in _rows("served_activity.csv")}
    assert len(want) == 18, want
    t = body("tab_served.tex")          # v13 (2026-09-25): the table is its own file, in Appendix C
    i = t.index(r"\label{tab:served}")
    tab = t[t.index(r"\midrule", i): t.index(r"\bottomrule", i)]
    label, seen, t07 = None, 0, False
    names = {"$8$B-Instruct, continuing text": "8B-Instruct, continuing text",
             "$70$B base (the authors' pair)": "70B base",
             "$8$B-Instruct, chat template": "8B-Instruct, chat template"}
    for line in tab.replace(r"\midrule", "").split(r"\\"):
        cells = [c.strip() for c in line.split("&")]
        if cells[0].startswith(r"\multicolumn"):
            t07 = "the authors' $0.7$ and $1.1$" in line        # the feat-195 block
            continue
        if len(cells) < 6:
            continue
        if cells[0]:
            label = names.get(cells[0])
            if label and t07:
                label += ", T=0.7"
        if label is None:
            continue                    # AnchoredByte rows: guarded by tests/test_skipped_round.py
        k = float(cells[1].strip("$").replace("^\\dagger", ""))
        printed = re.fullmatch(r"\$([\d.]+)\\%\$", cells[2]).group(1)
        assert _rounds(printed, want[(label, k)]), (label, k, printed, want[(label, k)])
        seen += 1
    assert seen == 18, seen
    assert "strict blend of the two models" in caption_of("tab:served")


def test_the_chat_block_reports_every_budget_and_the_crossover_it_shows():
    """feat-196's registered consequence: the chat block reports every budget on the grid, and the
    text states the smallest budget at which the meter's interval lies above selection's, with its
    K/S_w. Rows round from results/served_opponent.csv; the crossover is derived, not typed."""
    t1 = {r["quantity"]: r for r in _rows("served_opponent.csv") if r["band"] == "T1"}
    rows = {}
    for q, r in t1.items():
        m = re.fullmatch(r"chat template(, host B)? k=([\d.]+): selection n=64 minus meter", q)
        if m:
            k = m.group(2)
            lvl = t1[q.replace("selection n=64 minus meter", "meter level")]
            rows[float(k)] = (bool(m.group(1)), r, lvl)
    assert sorted(rows) == [0.5, 1.0, 2.0, 3.0, 5.0, 10.0], sorted(rows)
    t = body("experiments.tex")
    tt = body("tab_served.tex")         # v13 (2026-09-25): the table is its own file, in Appendix C
    tab = tt[tt.index(r"\label{tab:served}"): tt.index(r"\bottomrule", tt.index(r"\label{tab:served}"))]
    block = tab[tab.index("via its chat template"):]
    for k, (hostb, d, lvl) in rows.items():
        kk = f"{k:g}" + ("^\\dagger" if hostb else "")
        cell = f"& ${kk}$ & "
        i = block.index(cell)
        line = block[i: block.index("\\\\", i)]
        v, lo, hi = float(d["value"]), float(d["lo95"]), float(d["hi95"])
        assert f"${v:+.4f}$" in line and carries_band(v, lo, hi, "experiments.tex", "tab_served.tex"), (k, line)
        # round the CSV's DECIMAL string half-up, as a reader would: 0.2475 is 0.24749999... in binary
        from decimal import ROUND_HALF_UP, Decimal
        want = Decimal(lvl["value"]).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        assert f"${want}$" in line, (k, line, lvl["value"])
    # the crossover sentence: the smallest k whose interval (selection minus meter) lies below zero
    first = min(k for k, (_, d, _l) in rows.items() if float(d["hi95"]) < 0)
    ks = 200 * first / 159.83
    assert f"first lies above selection's at $k={first:g}$" in t, first
    assert f"$K/S_w = {ks:.2f}$" in t, ks
    tie = max(k for k, (_, d, _l) in rows.items() if k < first)
    d = rows[tie][1]
    assert float(d["lo95"]) <= 0 <= float(d["hi95"]) and f"after a tie at $k={tie:g}$" in t


# ------------------------------------------------------------------ Section 3, the windowed meter

def test_the_windowed_meter_numbers_round_from_the_log():
    w = {r["quantity"]: r for r in _rows("window_logratio.csv")}
    real, top = w["realised log-ratio per window"], w["largest realised window per trajectory"]
    sec3 = body("frontier.tex")
    i = sec3.index(r"\label{sec:uncovered}")
    para = sec3[i: sec3.index(r"\section", i) if r"\section" in sec3[i:] else None]
    # v13 (2026-09-25, feat-210): the windowed meter is BUILT and measured (Table~\ref{tab:matched},
    # Appendix~\ref{app:windowed}), so Section 3 no longer argues from this log what the meter would
    # do on 1,500 trajectories; it quotes the log's median as the imitation cost per window and points
    # at the appendix, which keeps the whole distribution (checked below).
    assert f"has median ${float(real['median']):.1f}$ nats" in para and r"\ref{app:windowlr}" in para
    assert float(top["p99"]) < float(real["S_w"]), "the windowed meter would bind on the protected window"
    # the sampling floor a near-verbatim variant faces at 40 nats is quoted where the meter's chat win is
    assert abs(3 * math.exp(40) / 1e17 - 7.06) < 0.01
    assert "only with more than $3e^{40}$ clean draws" in body("experiments.tex")
    # the introduction's "about $40$ nats for a $50$-token window" is the same median, rounded
    assert (f"about ${round(float(real['median']))}$ nats for a $50$-token window"
            in body("iclr_intro.tex")), "the imitation cost per window left the introduction"
    app = body("appendix_onset.tex")
    for q in ("median", "p90", "p99", "max"):
        assert f"${float(real[q]):.1f}$" in app, (q, real[q])
    assert f"exceeds ${float(top['p99']):.1f}$ on $1\\%$ of trajectories" in app


# ------------------------------------------------------------------ Section 4, empty answers

def test_the_empty_answer_readings_round_from_their_csvs_and_the_heading_follows_them():
    e = {(r["rule"], r["quantity"]): r for r in _rows("empty_as_loss.csv")}
    n = {(r["rule"], r["quantity"]): r for r in _rows("nonempty_rule.csv")}
    loss = e[("empty is a loss", "D3 difference")]
    keep = n[("empty is a loss", "D3 difference")]
    for r in (loss, keep):
        assert carries_band(float(r["value"]), float(r["lo95"]), float(r["hi95"]), "experiments.tex"), r
        assert carries_band(float(r["value"]), float(r["lo95"]), float(r["hi95"]), "appendix_selection.tex"), r
    served = int(e[("empty served texts", "sel_n64")]["value"])
    assert int(e[("empty served texts", "metered_k10")]["value"]) == 0
    exp = body("experiments.tex")
    assert f"serves ${served}$ empty texts and the meter none" in exp
    # the heading lists what the difference does NOT survive; empty-as-loss belongs there exactly
    # while that interval covers zero, and the non-empty rule's reading must clear zero to be quoted
    # as keeping the difference
    head = exp[exp.index(r"\textbf{At $k=10$ the difference survives"):][:260]
    covers = float(loss["lo95"]) <= 0 <= float(loss["hi95"])
    assert ("empty answers scored as losses" in head) == covers, (covers, head)
    assert float(keep["lo95"]) > 0 and "keeps $+" in exp
    # the CPU control is disclosed with its count
    ctl = n[("control", "committed picks re-judged on CPU, level reproduced")]
    assert f"reproduces their committed levels on ${int(ctl['value'])}$" in body("appendix_selection.tex")
    assert int(ctl["n"]) == served


# ------------------------------------------------------------------ Section 2, prefix extension

def test_the_prefix_extension_arm_is_reported_as_registered():
    """feat-199's P1 FAILED: the attack reconstructs no window although the composed certificate is
    vacuous for most. The registered consequence: report it as the realised side of the m log n
    horizon, call the certificate loose for this attack, and do NOT call the scorer safe."""
    rows = {r["arm"]: r for r in _rows("prefix_extension.csv")}
    r64 = rows["n=64"]
    assert int(r64["exact_windows"]) == 0 and int(r64["passages_certificate_vacuous"]) > 50
    sec2 = body("selection.tex")
    para = sec2[sec2.index(r"\label{sec:compose}"):]
    assert f"reconstructs none of ${r64['n_passages']}$ windows" in para
    assert f"certificate (${float(r64['certificate_nats']):.1f}$ nats)" in para
    assert f"vacuous for ${r64['passages_certificate_vacuous']}$" in para
    assert f"${100 * float(r64['true_token_in_pool']):.1f}\\%$ of queries" in para
    assert "looseness, not safety" in para and r"$S/\log n$" in para
    # the pre-launch sentence the arm refuted must be gone everywhere
    for f in ("selection.tex", "experiments.tex", "iclr_intro.tex", "iclr_closing.tex",
              "appendix_selection.tex"):
        assert "reconstructs text a single response never does" not in body(f), f
    # and Table tab:prefixext rounds from the CSV, row by row
    t = body("appendix_selection.tex")
    i = t.index(r"\label{tab:prefixext}")
    lines = t[t.index(r"\midrule", i): t.index(r"\bottomrule", i)].replace(r"\midrule", "").split(r"\\")
    got = [[c.strip().strip("$") for c in ln.split("&")] for ln in lines if ln.strip()]
    order = ["n=1", "n=8", "n=64", "memoriser alone, greedy"]
    assert len(got) == 4, got
    for cells, arm in zip(got, order):
        r = rows[arm]
        assert _rounds(cells[3], float(r["token_accuracy_mean"])), (arm, cells)
        assert cells[4] == r["exact_windows"], (arm, cells)
        assert _rounds(cells[5], float(r["nv_recall_mean"])), (arm, cells)
        if arm != "memoriser alone, greedy":
            assert _rounds(cells[1], float(r["certificate_nats"])), (arm, cells)
            assert cells[2] == r["passages_certificate_vacuous"], (arm, cells)
            assert _rounds(cells[6], float(r["true_token_in_pool"])), (arm, cells)


# ------------------------------------------------------------------ feat-198, a scorer from another family

def test_the_scorer_family_arm_is_reported_as_registered():
    """feat-198 (results/onset_prediction_scorer_family.md) read UNRESOLVED: the headline depends on the
    scorer. Registered: Section 4's paragraph reports the arm and its heading adds the scorer to what
    the difference does not survive, whatever it reads; the qualifier then travels to every sentence
    that states the continuing-text win. The heading rule is conditioned on the CSV, not on a phrase."""
    h = {r["quantity"][:2]: r for r in _rows("order_averaged_h2h_scorer_gemma27b.csv")}
    d1, d3 = h["D1"], h["D3"]
    for r in (d1, d3):
        assert carries_band(float(r["value"]), float(r["lo95"]), float(r["hi95"]), "experiments.tex"), r
    confirmed = float(d3["lo95"]) > 0
    exp = body("experiments.tex")
    head = exp[exp.index(r"\textbf{At $k=10$ the difference survives"):][:320]
    assert ("a scorer from another family" in head) == (not confirmed), (confirmed, head)
    sf = {(r["scorer"], r["quantity"]): r["value"] for r in _rows("scorer_family.csv")}
    assert int(sf[("both", "prompts whose served draw changes")]) >= 50, "G1: the arm cannot distinguish scorers"
    g_w, q_w = sf[("gemma-2-27b-it", "median words served")], sf[("Qwen2.5-7B-Instruct", "median words served")]
    g_e, q_e = sf[("gemma-2-27b-it", "empty served texts")], sf[("Qwen2.5-7B-Instruct", "empty served texts")]
    assert f"(median ${g_w}$ words against ${q_w}$)" in exp and f"(${g_e}$ against ${q_e}$)" in exp
    if not confirmed:
        abstract = " ".join(open(tex("iclr_2027.tex"), encoding="utf-8").read().split())
        abstract = abstract[abstract.index("begin{abstract}"):abstract.index("end{abstract}")]
        # The qualifier travels with every sentence that states the continuing-text win. v13
        # (2026-09-25, Review 3 A2) moved that win out of the abstract and the introduction into
        # Section 4; wherever either states it again, the qualifier comes back with it.
        if "continuing text" in abstract.lower() or "$k=10$" in abstract:
            assert "though not with a scorer from another family" in abstract
        intro = body("iclr_intro.tex")
        if "continuing text" in intro.lower():
            assert "with its Qwen scorer though not with a gemma one" in intro
        close = body("iclr_closing.tex")
        assert "with a scorer from another family" in close
        assert "the headline depends on the scorer" in exp
        # v14 (feat-212): the same comparison judged by all six judges resolves under four of them, so the
        # concession is judge~B's and must say so wherever it is stated -- conditioned on the CSV, not a phrase
        fac = {r["judge"]: r for r in _rows("scorer_judge_factorial.csv")}
        if any(fac[j]["D3_gemma_reading"] == "CONFIRMED" for j in fac if j != "B"):
            assert "under the pre-specified judge, with a scorer from another family" in close
            assert "under~B the headline depends on the scorer" in exp
            assert "or, under judge~B, a scorer from another family" in exp


# ------------------------------------------------------------------ feat-195, the authors' decoding settings

def test_the_temperature_07_block_and_its_registered_sentences():
    """feat-195 (results/onset_prediction_he_decoding.md). Table 2's block rounds from
    results/served_opponent.csv and results/he_decoding.csv (K/S_w against the WARPED anchor, as
    registered), and H1's REFUTED carries its registered consequence: the continuing-text claim is
    scoped to temperature 1.0 with the 0.7 result in the same sentence, in the abstract and Section 4."""
    from decimal import ROUND_HALF_UP, Decimal
    t1 = {r["quantity"]: r for r in _rows("served_opponent.csv") if r["band"] == "T1"}
    he = {(r["band"], r["quantity"]): r for r in _rows("he_decoding.csv")}
    sw = float(he[("S_w", "50-token window, warped anchor (0.7, 1.1), median over passages")]["value"])
    t = body("experiments.tex")
    tt = body("tab_served.tex")         # v13 (2026-09-25): the table is its own file, in Appendix C
    i = tt.index(r"\label{tab:served}")
    tab = tt[i: tt.index(r"\bottomrule", i)]
    blk = tab[tab.index("the authors' $0.7$ and $1.1$"):]
    blk = blk[: blk.index(r"\midrule")]
    for pair, ks in (("8B", ("0.5", "1", "10")), ("70B", ("0.5", "1", "20"))):
        for k in ks:
            d = t1[f"temperature 0.7, {pair} k={k}: selection n=64 minus meter"]
            lvl = Decimal(t1[f"temperature 0.7, {pair} k={k}: meter level"]["value"]).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP)
            kk = he[("K/S_w", f"k={k}, K = 200k over the warped anchor's S_w")]["value"]
            rows = [ln for ln in blk.split(r"\\") if f"& ${k}$ &" in ln]
            row = rows[0] if pair == "8B" else rows[-1]
            assert f"${lvl}$" in row, (pair, k, row, lvl)
            assert f"${float(d['value']):+.4f}$" in row, (pair, k, row)
            assert any(_rounds(x, float(kk)) for x in re.findall(r"\$([\d.]+)\$", row)), (pair, k, kk, row)
    assert f"$S_w = {sw:.1f}$ nats" in caption_of("tab:served")
    h1 = he[("H1", "J1 8B k=10: D3 difference of gains, paired")]
    h2 = he[("H2", "J1 8B k=0.5: D5 met8b_k0.5 minus selection, paired")]
    h5 = he[("H5", "J3 70B k=0.5: D5 met70b_k0.5 minus selection, paired")]
    h3 = he[("H3", "J3 70B k=20: D3 difference of gains, paired")]
    # H2 and H5 are reported in the sentence about the informative budget, whatever they read
    assert carries_band(float(h2["value"]), float(h2["lo95"]), float(h2["hi95"]), "experiments.tex")
    assert f"selection leads by ${-float(h5['value']):.3f}$" in t
    if h1["reading"] == "REVERSAL REFUTED":
        assert f"the $8$B wins instead, ${float(h1['value']):+.3f}$ $[{float(h1['lo95']):+.3f}, {float(h1['hi95']):+.3f}]$" in t
        assert carries_band(float(h3["value"]), float(h3["lo95"]), float(h3["hi95"]), "experiments.tex")
        assert ("Continuing text at temperature $1.0$, that model loses" in t
                or "Continuing text at temperature $1.0$, the risky model loses" in t)
        abstract = " ".join(open(tex("iclr_2027.tex"), encoding="utf-8").read().split())
        abstract = abstract[abstract.index("begin{abstract}"):abstract.index("end{abstract}")]
        # the registered consequence: the 0.7 loss is in the abstract, and a continuing-text WIN, if the
        # abstract states one (v13 moved it to Section 4, Review 3 A2), is scoped to temperature 1.0
        assert "loses to the $8$B at their temperature $0.7$" in abstract
        if "continuing text" in abstract.lower():
            assert "continuing text at temperature $1.0$" in abstract.lower()
    close = body("iclr_closing.tex")
    assert "where its authors use $0.7$ and $1.1$" not in close, "Limitations still says the meter ran only at 1.0"
    assert "at the authors' temperature $0.7$ and penalty $1.1$" in close
