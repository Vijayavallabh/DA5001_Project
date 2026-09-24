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
    assert len(want) == 12, want
    t = body("experiments.tex")
    i = t.index(r"\label{tab:served}")
    tab = t[t.index(r"\midrule", i): t.index(r"\bottomrule", i)]
    label, seen = None, 0
    names = {"$8$B-Instruct, continuing text": "8B-Instruct, continuing text",
             "$70$B base (the authors' pair)": "70B base",
             "$8$B-Instruct, chat template": "8B-Instruct, chat template"}
    for line in tab.replace(r"\midrule", "").split(r"\\"):
        cells = [c.strip() for c in line.split("&")]
        if len(cells) < 6 or cells[0].startswith(r"\multicolumn"):
            continue
        if cells[0]:
            label = names.get(cells[0])
        if label is None:
            continue                    # AnchoredByte rows: guarded by tests/test_skipped_round.py
        k = float(cells[1].strip("$").replace("^\\dagger", ""))
        printed = re.fullmatch(r"\$([\d.]+)\\%\$", cells[2]).group(1)
        assert _rounds(printed, want[(label, k)]), (label, k, printed, want[(label, k)])
        seen += 1
    assert seen == 12, seen
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
    tab = t[t.index(r"\label{tab:served}"): t.index(r"\bottomrule", t.index(r"\label{tab:served}"))]
    block = tab[tab.index("via its chat template"):]
    for k, (hostb, d, lvl) in rows.items():
        kk = f"{k:g}" + ("^\\dagger" if hostb else "")
        cell = f"& ${kk}$ & "
        i = block.index(cell)
        line = block[i: block.index("\\\\", i)]
        v, lo, hi = float(d["value"]), float(d["lo95"]), float(d["hi95"])
        assert f"${v:+.4f}$" in line and carries_band(v, lo, hi, "experiments.tex"), (k, line)
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
    assert f"on ${int(top['n']):,}$".replace(",", "{,}") in para
    assert f"has median ${float(real['median']):.1f}$ nats" in para
    # "exceeds $125$ nats on only $1\%$ of them": the p99 of the per-trajectory maximum
    assert f"exceeds ${int(float(top['p99']))}$ nats on only $1\\%$" in para
    assert float(top["p99"]) < float(real["S_w"]), "the windowed meter would bind on the protected window"
    assert f"below the protected window's ${float(real['S_w']):.1f}$" in para
    # the sampling floor quoted for a 40-nat certificate
    assert abs(3 * math.exp(40) / 1e17 - 7.06) < 0.01 and r"3e^{40} \approx 7\times10^{17}" in para
    # the abstract's "a median $40$ nats per $50$-token window" is the same median, rounded
    abstract = " ".join(open(tex("iclr_2027.tex"), encoding="utf-8").read().split())
    assert f"a median ${round(float(real['median']))}$ nats per $50$-token window" in abstract
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
        assert "though not with a scorer from another family" in abstract
        assert "with its Qwen scorer though not with a gemma one" in body("iclr_intro.tex")
        close = body("iclr_closing.tex")
        assert "or with a scorer from another family" in close and "which scorer selects" in close
        assert "the headline depends on the scorer" in exp
