"""The numbers Sections 5 and 6 rest on, checked against the CSVs that produced them.

The selection sections are new in v6 and carry the paper's constructive claim, so every load-bearing
literal in them is pinned here rather than read by eye. The odometer arithmetic is included because
it is the one place the paper does a division in prose."""
import csv
import math
import re

from tests.manuscript import tex

SEL = tex("sections/selection.tex")
EXP = tex("sections/experiments.tex")
APP = tex("sections/appendix_selection.tex")


def _rows(path):
    return list(csv.DictReader(open(path)))


def test_the_median_protected_target_is_quoted_from_the_odometer_csv():
    """849 is S_total_median in results/odometer.csv. It was quoted as 850 in four places until
    2026-09-11, which is a second rounding of an integer the CSV already stores exactly."""
    vals = {r["S_total_median"] for r in _rows("results/odometer.csv")}
    assert len(vals) == 1, vals
    s_tot = float(vals.pop())
    body = "".join(open(f, encoding="utf-8").read() for f in (SEL, EXP, APP,
                                                              tex("iclr_2027.tex"),
                                                              tex("sections/frontier.tex"),
                                                              tex("sections/iclr_closing.tex")))
    assert f"${s_tot:.0f}$" in body, f"the paper no longer quotes S(x) = {s_tot:.0f}"
    assert "$850$" not in body, "the rounded 850 is back"


def test_the_selection_budget_arithmetic_is_exact():
    """log n - (n-1)/n at n = 8, and log 8 for the pathwise budget. Both are quoted to 2 or 3 dp."""
    kl8 = math.log(8) - 7 / 8
    assert f"${kl8:.3f}$" == "$1.204$"
    body = open(SEL, encoding="utf-8").read() + open(EXP, encoding="utf-8").read()
    assert "$1.204$" in body or "$1.20$" in body
    assert f"$\\log 8 = {math.log(8):.2f}$" in open(SEL, encoding="utf-8").read()


def test_the_composition_count_divides_out():
    """332 queries at n = 8 and 2 at k = 3, both from a 400-nat odometer over the measured spend."""
    spend = {r["k"]: float(r["mean_spend_nats"]) for r in _rows("results/utility_price.csv")}
    kl8 = math.log(8) - 7 / 8
    assert int(400 / kl8) == 332, 400 / kl8
    assert int(400 / spend["3.0"]) == 2, 400 / spend["3.0"]
    body = open(SEL, encoding="utf-8").read().replace("\n", " ")
    assert "$332$ queries at $n=8$ against $2$" in body, body[body.find("composes"):][:220]
    m = re.search(r"spends a measured \$([\d.]+)\$ nats", body)
    assert m and float(m.group(1)) == spend["3.0"], (m.group(1) if m else None, spend["3.0"])


def test_the_cross_judge_gain_and_its_interval_come_from_the_csv():
    rows = _rows("results/selection_crossjudge.csv")
    g = {r["gain"] for r in rows}
    lo = {r["gain_lo95"] for r in rows}
    hi = {r["gain_hi95"] for r in rows}
    assert len(g) == len(lo) == len(hi) == 1
    body = "".join(open(f, encoding="utf-8").read() for f in (EXP, tex("sections/iclr_closing.tex")))
    assert f"$+{float(g.pop()):.3f}$" in body
    assert f"$[+{float(lo.pop()):.3f}, +{float(hi.pop()):.3f}]$" in body


def test_extraction_is_zero_at_every_n_in_the_table():
    """The safety claim. Every n up to 64 must read 0.0000, and the risky baseline must not."""
    rows = {r["n"]: r for r in _rows("results/selection_extraction.csv")}
    for n in ("1", "2", "4", "8", "16", "32", "64"):
        assert float(rows[n]["nv_recall_mean"]) == 0.0, n
        assert float(rows[n]["nv_recall_max"]) == 0.0, n
    # The threshold is "substantially non-zero", not a specific value: it was 0.4 until the
    # tokenizer fix of 2026-09-12 moved the baseline from 0.4338 (the memoriser fed the ANCHOR's
    # token ids) to 0.3925 (its own), which is the number every anchor's arm now agrees on.
    assert float(rows["-1"]["nv_recall_mean"]) > 0.3
    body = open(EXP, encoding="utf-8").read()
    assert f"${float(rows['-1']['nv_recall_mean']):.4f}$" in body
    assert f"${float(rows['-1']['nv_recall_max']):.4f}$" in body


def test_the_n_sweep_rows_in_the_table_round_from_selection_scaling_csv():
    """Table 1 reports GAINS over each arm's own control, not levels: the judge-consistency arm
    showed an absolute level is largely a statement about slot order. Checked mechanically against
    the CSV, not by eye (caution (j))."""
    import csv as _csv
    from tests.manuscript import tex as _tex
    rows = list(_csv.DictReader(open("results/selection_scaling.csv")))
    body = open(_tex("sections/experiments.tex"), encoding="utf-8").read()
    want = [("Phi-3.5-mini-instruct", 8, "judge B"), ("Phi-3.5-mini-instruct", 64, "judge B"),
            ("Meta-Llama-3.1-8B-Instruct", 64, "judge C")]
    for judge, n, label in want:
        r = next(x for x in rows if judge in x["judge"] and int(float(x["n"])) == n)
        cell = ("pointwise reward            & {} & ${}$ & ${:.3f}$ & ${:+.3f}$ & "
                "$[{:+.3f}, {:+.3f}]$").format(label, n, float(r["kl_nats"]), float(r["gain"]),
                                               float(r["gain_lo95"]), float(r["gain_hi95"]))
        assert cell in body, cell


def test_no_absolute_judged_level_is_quoted_as_a_comparison():
    """The instrument check (results/onset_prediction_judge_consistency.md) found the same text
    wins 261/500 shown second and 24/500 shown first. Levels from different judged passes are
    therefore not comparable, and the paper must not put two of them side by side again."""
    from tests.manuscript import tex as _tex
    body = open(_tex("sections/experiments.tex"), encoding="utf-8").read().replace("\n", " ")
    assert "reaches $0.522$" not in body and "reaches $0.577$" not in body, \
        "a level-vs-level comparison is back in Section 6"
    assert "gains over each arm's own" in body or "gain over control" in body


def test_the_reversal_claim_is_true_of_the_csvs_it_cites():
    """The paper says the comparison is a reversal. That is only allowed while selection's GAIN at
    n=64 exceeds the metered decoder's gain over its own control, at a far smaller budget."""
    import csv as _csv
    import re as _re
    from tests.manuscript import tex as _tex
    sel = next(r for r in _csv.DictReader(open("results/selection_scaling.csv"))
               if "Phi-3.5" in r["judge"] and int(float(r["n"])) == 64)
    j2 = list(_csv.DictReader(open("results/judge_separation_v6_judge2.csv")))
    anchor = next(x for x in j2 if x["decoder"].startswith("anchor"))
    k10 = next(x for x in j2 if x["decoder"] == "KL" and float(x["k"]) == 10.0)
    dec_gain = float(k10["utility"]) - float(anchor["utility"])
    dec = next(r for r in _csv.DictReader(open("results/selection_crossjudge.csv"))
               if "metered" in r["selector"])
    assert float(sel["gain"]) > dec_gain, (sel["gain"], dec_gain)
    ratio = float(dec["kl_nats"]) / float(sel["kl_nats"])
    body = open(_tex("sections/experiments.tex"), encoding="utf-8").read().replace("\n", " ")
    m = _re.search(r"metered\s*decoder gains \$\+([\d.]+)\$ for \$([\d.]+)\$ nats and selection\s*"
                   r"\$\+([\d.]+)\$ for \$([\d.]+)\$", body)
    assert m, "the reversal sentence has moved"
    assert abs(float(m.group(1)) - dec_gain) < 0.001, (m.group(1), dec_gain)
    assert float(m.group(3)) == round(float(sel["gain"]), 3), (m.group(3), sel["gain"])
    assert abs(float(m.group(4)) - float(sel["kl_nats"])) < 0.005
    assert "fifty-fourth" in body and 53.0 < ratio < 55.0, ratio


def test_the_sweep_is_monotone_in_log_n_on_both_judges():
    """O1 read SCALES. If a rerun ever made it non-monotone the paragraph would be wrong, and the
    Spearman the paper quotes is the thing to check."""
    import csv as _csv
    rows = list(_csv.DictReader(open("results/selection_scaling.csv")))
    for judge in {r["judge"] for r in rows}:
        arms = sorted((int(float(r["n"])), float(r["u"])) for r in rows if r["judge"] == judge)
        assert arms[-1][1] > arms[0][1], (judge, arms)
        assert float(next(r for r in rows if r["judge"] == judge)["spearman_u_logn"]) > 0.95, judge
    b = {int(float(r["n"])): r for r in rows if "Phi-3.5" in r["judge"]}
    gain8 = float(b[8]["u"]) - float(b[1]["u"])
    gain64 = float(b[64]["u"]) - float(b[1]["u"])
    assert gain64 >= gain8 + 0.05, (gain8, gain64)      # the committed SCALES band


def test_the_position_bias_numbers_in_section_6_come_from_the_per_prompt_file():
    """The judge-consistency paragraph is the paper's strongest methodological claim and every
    figure in it is derivable from results/judge_consistency{,_per_prompt}.csv."""
    import csv as _csv
    import collections
    import re as _re
    from tests.manuscript import tex as _tex
    rows = list(_csv.DictReader(open("results/judge_consistency_per_prompt.csv")))
    fw = collections.Counter(r["u_n1_fwd"] for r in rows)
    rv = collections.Counter(r["u_n1_rev"] for r in rows)
    first_wins, second_wins = fw["1.0"], rv["1.0"]
    assert second_wins > 5 * first_wins, (first_wins, second_wins)
    crit = {r["criterion"]: r for r in _csv.DictReader(open("results/judge_consistency.csv"))}
    c1 = float(crit["C1 order consistency"]["value"])
    c2 = float(crit["C2 first-slot win rate"]["value"])
    body = open(_tex("sections/experiments.tex"), encoding="utf-8").read().replace("\n", " ")
    m = _re.search(r"win \$(\d+)\$ of \$500\$ shown second and \$(\d+)\$ shown first", body)
    assert m and (int(m.group(1)), int(m.group(2))) == (second_wins, first_wins), \
        (m.groups() if m else None, second_wins, first_wins)
    m = _re.search(r"only \$([\d.]+)\\%\$ of items get a mutually\s*consistent verdict", body)
    assert m and abs(float(m.group(1)) - 100 * c1) < 0.05, (m.group(1) if m else None, c1)
    m = _re.search(r"first slot wins \$([\d.]+)\\%\$", body)
    assert m and abs(float(m.group(1)) - 100 * c2) < 0.05, (m.group(1) if m else None, c2)
    # and the order-averaged gain the paragraph leans on
    c3 = crit["C3 gain, order-averaged"]
    m = _re.search(r"leaves \$\+([\d.]+)\$\s*\$\[\+([\d.]+), \+([\d.]+)\]\$", body)
    assert m, "the order-averaged gain has moved"
    assert abs(float(m.group(1)) - float(c3["value"])) < 5e-4, (m.group(1), c3["value"])


def test_the_memoriser_baseline_is_identical_at_every_anchor():
    """Same model, same 100 passages, same seeds, and since 2026-09-12 its own tokenizer, so the
    k=-1 arm must not depend on which anchor it was measured beside. It used to: feeding the safe
    model's token ids to the memoriser moved this by 0.04, which is a tenth of the quantity."""
    import glob as _glob
    import os as _os
    vals = set()
    for path in _glob.glob("results/selection_extraction*.csv"):
        if path.endswith("_per_passage.csv"):
            continue
        r = {x["n"]: x for x in _rows(path)}
        if "-1" not in r:
            continue
        vals.add((round(float(r["-1"]["nv_recall_mean"]), 4),
                  round(float(r["-1"]["nv_recall_max"]), 4),
                  round(float(r["-1"]["ge_0p01_pct"]), 1)))
        assert _os.path.basename(path)
    assert len(vals) == 1, vals
    assert vals == {(0.3925, 0.8154, 78.0)}, vals


def test_every_anchor_reports_zero_recall_at_every_n():
    """Proposition 4 says n multiplies the ANCHOR's own rate, and no anchor saw the work. Four
    anchors now, and Section 6 says 'at all four anchors', so all four have to be on disk."""
    import glob as _glob
    arms = [p for p in _glob.glob("results/selection_extraction*.csv")
            if not p.endswith("_per_passage.csv")]
    assert len(arms) == 4, arms
    for path in arms:
        for r in _rows(path):
            if r["n"] == "-1":
                continue
            assert float(r["nv_recall_mean"]) == 0.0, (path, r["n"])
            assert float(r["nv_recall_max"]) == 0.0, (path, r["n"])
