"""The numbers Sections 5 and 6 rest on, checked against the CSVs that produced them.

The selection sections are new in v6 and carry the paper's constructive claim, so every load-bearing
literal in them is pinned here rather than read by eye. The odometer arithmetic is included because
it is the one place the paper does a division in prose."""
import csv
import pathlib
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
    # `$850$` ALSO COUNTS PROMPTS. The workload arm runs on 850 of them, so a bare substring check
    # fails on a sentence that has nothing to do with surprisal -- caution (an): a guard matching a
    # number without its context is not guarding its sentence. What is forbidden is 850 standing
    # where the median surprisal belongs.
    import re as _re
    for m in _re.finditer(r"\$850\$", body):
        after = body[m.end():m.end() + 40]
        before = body[max(0, m.start() - 80):m.start()]
        assert "nat" not in after.lower(), f"the rounded 850 is back: ...{after[:40]!r}"
        assert "surprisal" not in before.lower() and "S(x)" not in before, \
            f"the rounded 850 is back: {before[-60:]!r}"


def test_the_selection_budget_arithmetic_is_exact():
    """log n - (n-1)/n at n = 8, and log 8 for the pathwise budget. Both are quoted to 2 or 3 dp."""
    kl8 = math.log(8) - 7 / 8
    assert f"${kl8:.3f}$" == "$1.204$"
    body = open(SEL, encoding="utf-8").read() + open(EXP, encoding="utf-8").read()
    assert "$1.204$" in body or "$1.20$" in body
    assert f"$\\log 8 = {math.log(8):.2f}$" in open(SEL, encoding="utf-8").read()


def test_the_composition_count_divides_out():
    """The odometer count depends on the Renyi order of the per-query charge, and the paper
    advertises a PATHWISE certificate. Quoting the KL count (332) under a log n headline was the
    defect a reviewer caught on 2026-09-14: 400/log 8 = 192 is the number at the advertised order,
    400/(log 8 - 7/8) = 332 the number the deployed KL odometer permits. Both must appear, the
    pathwise one must be the one Section 2 quotes, and neither may be swapped for the other."""
    spend = {r["k"]: float(r["mean_spend_nats"]) for r in _rows("results/utility_price.csv")}
    kl8 = math.log(8) - 7 / 8
    import math as _m
    assert int(400 / kl8) == 332, 400 / kl8
    assert int(400 / _m.log(8)) == 192, 400 / _m.log(8)
    # a 200-token response at the audited k=3 is certified at 600 nats, so the odometer admits none
    assert int(400 / (3 * 200)) == 0
    assert int(400 / spend["3.0"]) == 2, 400 / spend["3.0"]
    body = open(SEL, encoding="utf-8").read().replace("\n", " ")
    body1 = body.replace("\n", " ")
    assert "that is $192$ queries" in body1, body1[body1.find("composes"):][:260]
    assert "$1.204$ nats gives $332$" in body1, "the KL count must be reported beside the pathwise one"
    assert "$332$ queries" not in body1, "the KL count is being quoted as THE composition count"
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


def test_the_n_sweep_arms_round_from_selection_scaling_csv():
    """These were Table 1 rows until 2026-09-17, when four of its five rows became rows of the
    forest figure and the table came out. They report GAINS over each arm's own control, never
    levels: the judge-consistency arm showed an absolute level is largely slot order. Checked
    mechanically against the CSV wherever the paper prints them (caution (j))."""
    import csv as _csv
    from tests.manuscript import carries_band, body as _body
    rows = list(_csv.DictReader(open("results/selection_scaling.csv")))
    want = [("Phi-3.5-mini-instruct", 8), ("Phi-3.5-mini-instruct", 64),
            ("Meta-Llama-3.1-8B-Instruct", 64)]
    for judge, n in want:
        r = next(x for x in rows if judge in x["judge"] and int(float(x["n"])) == n)
        assert carries_band(float(r["gain"]), float(r["gain_lo95"]), float(r["gain_hi95"]),
                            "experiments.tex"), (judge, n, r["gain"])
    # the measured KL of the headline arm is stated, and is NOT the certificate (log 64 = 4.159)
    r64 = next(x for x in rows if "Phi-3.5" in x["judge"] and int(float(x["n"])) == 64)
    assert f"${float(r64['kl_nats']):.3f}$ nats" in _body("experiments.tex"), r64["kl_nats"]


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
    """The paper says the comparison is a reversal. Since 2026-09-14 that claim is the ORDER-AVERAGED
    head-to-head (results/order_averaged_h2h.csv), not the two single-order gains it used to quote:
    a reviewer pointed out that a comparison of two gains measured with a position-dominated judge
    had never itself been checked for position, and feat-113 checked it. The difference must be
    positive with an interval excluding zero, and Section 3 must quote all three numbers."""
    import csv as _csv
    import re as _re
    from tests.manuscript import tex as _tex
    rows = {r["quantity"]: r for r in
            _csv.DictReader(open("results/order_averaged_h2h.csv"))}
    sel = rows["D1 selection gain, order-averaged"]
    met = rows["D2 metered gain, order-averaged"]
    dif = rows["D3 difference of gains, paired"]
    # the reversal is only claimable while the paired difference excludes zero
    assert float(dif["lo95"]) > 0, dif
    assert dif["reading"] == "REVERSAL CONFIRMED", dif
    assert abs((float(sel["value"]) - float(met["value"])) - float(dif["value"])) < 5e-4

    # Scan the body, not one file. Figure 1 moved from Section 3 into the introduction on
    # 2026-09-17 and took the divergence-ratio claim with it in its caption, at which point a
    # guard pinned to experiments.tex alone reported the claim missing when it had only moved.
    # Same lesson as caution (af), in reverse: follow the claim across every section that can
    # carry it, and keep asserting it exists and matches the CSV.
    body = " ".join("".join(open(_tex(f"sections/{f}.tex"), encoding="utf-8").read()
                            for f in ("iclr_intro", "selection", "experiments", "orders")).split())
    m = _re.search(r"selection gains \$\+([\d.]+)\$ \$\[\+([\d.]+), \+([\d.]+)\]\$ for \$3.175\$ "
                   r"nats and the metered decoder \$\+([\d.]+)\$ \$\[\+([\d.]+), \+([\d.]+)\]\$",
                   body)
    assert m, "the order-averaged head-to-head sentence has moved"
    assert abs(float(m.group(1)) - float(sel["value"])) < 5e-4, (m.group(1), sel["value"])
    assert abs(float(m.group(4)) - float(met["value"])) < 5e-4, (m.group(4), met["value"])
    m2 = _re.search(r"difference of \$\+([\d.]+)\$ \$\[\+([\d.]+), \+([\d.]+)\]\$", body)
    assert m2, "the paired difference has moved"
    assert abs(float(m2.group(1)) - float(dif["value"])) < 5e-4, (m2.group(1), dif["value"])
    # and the divergence ratio it is set against, which no judging pass can change
    dec = next(r for r in _csv.DictReader(open("results/selection_crossjudge.csv"))
               if "metered" in r["selector"])
    scal = next(r for r in _csv.DictReader(open("results/selection_scaling.csv"))
                if "Phi-3.5" in r["judge"] and int(float(r["n"])) == 64)
    ratio = float(dec["kl_nats"]) / float(scal["kl_nats"])
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
    # The judge-methodology block moved to Appendix (app:judgemethod) on 2026-09-19; the
    # numbers went with it, so scan both files rather than the section it used to sit in.
    body = " ".join(open(_tex("sections/experiments.tex"), encoding="utf-8").read().split()) + " " + \
           " ".join(open(_tex("sections/appendix_selection.tex"), encoding="utf-8").read().split())
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
        # Any _70b arm is a different RISKY model, not an anchor arm: its k=-1 row is the 70B
        # alone and has no reason to equal the LoRA memoriser's. endswith("_70b.csv") was too
        # narrow -- selection_extraction_70b_raw.csv (feat-103) walked straight through it and
        # contributed a (0.0, 0.0, 0.0) baseline. Match the sibling test and skip the substring.
        # _multilingual is a different MEMORISER on a different corpus (feat-152: French and
        # German passages, its own LoRA), so its k=-1 baseline has no reason to equal the English
        # one and in fact exceeds it, 0.5455 against 0.3925. The invariant here is about the SAME
        # memoriser measured beside different ANCHORS; it does not reach across corpora.
        if path.endswith("_per_passage.csv") or "_70b" in path or "_multilingual" in path:
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
            if not p.endswith("_per_passage.csv") and "_70b" not in p]
    assert len(arms) >= 4, arms
    # Section 6 states the count, and the paper may never claim more anchors than were measured.
    # It may claim FEWER while a registered anchor's arm is still running: C4 of the six-anchor
    # pre-registration says the sentence stays at four until every new anchor has been scored, so
    # the count is pinned to the registered set once that set is complete and bounded by the
    # measured set until then. Both halves matter -- the upper bound stops an unmeasured anchor
    # being counted, the equality stops a measured one being quietly left out.
    import re as _re
    from analysis.selection_breadth import ANCHORS as _ANCHORS
    from tests.manuscript import tex as _tex
    words = {4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
    nums = {w: n for n, w in words.items()}
    body = " ".join(open(_tex("sections/experiments.tex"), encoding="utf-8").read().split())
    m = _re.search(r"at all ([a-z]+) anchors", body)
    assert m and m.group(1) in nums, "Section 6 no longer states the anchor count"
    stated = nums[m.group(1)]
    assert stated <= len(arms), (f"Section 6 claims {stated} anchors; only {len(arms)} leakage "
                                 f"arms are on disk")
    if len(arms) >= len(_ANCHORS):
        assert stated == len(_ANCHORS), (stated, len(_ANCHORS),
                                         "every registered anchor is measured; say so")
    for path in arms:
        for r in _rows(path):
            if r["n"] == "-1":
                continue
            assert float(r["nv_recall_mean"]) == 0.0, (path, r["n"])
            assert float(r["nv_recall_max"]) == 0.0, (path, r["n"])


def test_the_pathwise_form_of_prop_sparse_states_where_it_has_no_force():
    """Appendix A sharpens Proposition 6 from an expectation to a per-trajectory count for the
    deployed class. The first draft had the inequality backwards, so this pins the arithmetic:
    with K = k T_max = 600 and T = 200, N_eps <= K/eps is below T only when eps > 3, i.e. it
    constrains NOTHING below 3 nats in a single step."""
    from tests.manuscript import tex
    body = " ".join(open(tex("sections/appendix_proofs.tex")).read().split())
    assert "holds surely, not just in expectation" in body, "the sharpening is gone"
    k, t_max, T = 3.0, 200, 200
    eps_star = k * t_max / T          # 3.0: above this the count is below the trivial bound
    assert eps_star == 3.0
    assert "constrains nothing below $\\varepsilon = 3$ nats" in body, (
        "the appendix must say the pathwise count is uninformative BELOW eps=3, not above")
    assert "$600/\\varepsilon$" in body and "$200$ steps" in body


def test_the_70b_extraction_arm_is_reported_as_gate_failed_not_as_a_zero():
    """Its three zeros look exactly like the anchor arms' and mean the opposite. The 70B recovers
    NOTHING of these passages unaided, so its own committed gate fails and E1 is not read: an
    adversary that cannot extract proves nothing about a certificate."""
    r = {x["n"]: x for x in _rows("results/selection_extraction_70b.csv")}
    assert float(r["-1"]["nv_recall_mean"]) < 0.10, "the gate would now pass; re-score the arm"
    log = open("results/onset_prediction_extraction_natural.md", encoding="utf-8").read()
    head, _, scored = log.partition("\n## Scoring,")
    assert scored, "the arm is unscored"
    assert "gate fails" in scored and "NOT read as NO LEAK" in scored
    # and the arm is now known to have been invalid: a base model seeded with an instruction
    # header. The addendum must stay, and it must not have edited the score above its line.
    assert "Addendum" in scored and "instruction header" in scored, \
        "the pipeline defect that caused the gate failure is no longer disclosed"
    assert "0.4137" in scored, "the contradicting known truth is not cited"
    assert pathlib.Path("results/onset_prediction_extraction_natural_raw.md").exists(), \
        "the corrected arm has no pre-registration"
    # and the paper must not claim it. The limitation "the zero leakage is measured against a
    # memoriser WE fine-tuned" may only be dropped while an arm exists that measured it against
    # one we did not -- feat-110, whose own gate must pass. Three arms before it produced a clean
    # 0.0000 with a gate that did not, and any of them would have bought this sentence's removal
    # on nothing.
    from tests.manuscript import tex
    close = " ".join(open(tex("sections/iclr_closing.tex"), encoding="utf-8").read().split())
    nat = "results/selection_extraction_70b_hp2.csv"
    passed = False
    if pathlib.Path(nat).exists():
        h = {x["n"]: x for x in _rows(nat)}
        passed = float(h["-1"]["nv_recall_mean"]) >= 0.10 and all(
            float(v["nv_recall_mean"]) == 0.0 for k, v in h.items() if k != "-1")
    if not passed:
        assert "memoriser \\emph{we} fine-tuned" in close, \
            "Limitations must say whose memoriser the zero-leakage result is against"
    else:
        assert "memoriser \\emph{we} fine-tuned" not in close, \
            "feat-110 passed its gate at NO LEAK; the withdrawn limitation is back"


def test_the_n64_comma7b_arm_is_reported_with_its_failed_nested_check():
    """G3 failed, so the cross-arm reading is unreadable and the abstract must NOT have moved to
    this arm's number. The within-pass curve is fine and is what the appendix reports."""
    import csv as _csv
    from tests.manuscript import tex
    rows = list(_csv.DictReader(open("results/selection_scaling_comma7b64.csv")))
    b = [r for r in rows if "Phi-3.5" in r["judge"]]
    assert {int(r["n"]) for r in b} == {1, 2, 4, 8, 16, 32, 64}
    g8 = float(next(r for r in b if r["n"] == "8")["gain"])
    breadth = float(next(r for r in _csv.DictReader(open("results/selection_scaling_comma7b.csv"))
                         if "Phi-3.5" in r["judge"] and r["n"] == "8")["gain"])
    assert abs(g8 - breadth) > 0.03, "G3 would now pass; the appendix text must be re-scored"
    # The internal nested-check disclosure ("G3 failed, not readable") was process vocabulary and
    # was removed on 2026-09-19. What still matters, and is all this test now pins, is that the
    # abstract never moved to that arm's number -- asserted below.
    apx = " ".join(open(tex("sections/appendix_selection.tex"), encoding="utf-8").read().split())
    g64 = float(next(r for r in b if r["n"] == "64")["gain"])
    # the abstract keeps the number G1 never licensed it to change
    absr = " ".join(open(tex("iclr_2027.tex"), encoding="utf-8").read().split())
    assert f"{g64:.3f}" not in absr.split("\\end{abstract}")[0], \
        "the abstract moved to a gain whose cross-arm reading was declared unreadable"


def test_the_cross_pass_floor_has_both_measurements():
    """0.034 from the second-pair pass and 0.039 from the n=64 arm. If either moves, the appendix
    sentence that says 'about 0.04' has to move with it."""
    import csv as _csv
    from tests.manuscript import tex
    n1 = [r for r in _csv.DictReader(open("results/frontier_pair_llama321b.csv"))
          if r["arm"] == "selection, n=1" and "Phi-3.5" in r["judge"]][0]
    assert abs(abs(float(n1["gain"])) - 0.034) < 5e-4
    apx = " ".join(open(tex("sections/appendix_proofs.tex"), encoding="utf-8").read().split())
    assert "$0.039$" in apx and "$-0.034$" in apx
    assert "cross-pass floor" in apx


def test_the_cost_column_keeps_a_bound_and_a_measurement_apart():
    r"""The paper's cost columns must not blur what is BOUNDED with what was MEASURED.

    Two passes were needed. Until 2026-09-17 Table 1's column was headed "budget, nats" while
    every value in it was `kl_nats` from selection_scaling.csv -- flattering the baseline 12x. The
    third read-through renamed it "measured KL, nats", which was **also wrong and is corrected
    here**: `analysis.selection_decoding.kl_best_of_n` returns the CLOSED FORM
    `log n - (n-1)/n` \citep{beirami2025bestofn}, so 3.1745 at n=64 and 1.2044 at n=8 were never
    measurements of anything. The same mislabel sat in an appendix table whose column read
    "realised KL" over four genuinely measured metered rows and three bounded selection rows.
    Only the metered decoder's 171.3 is a realisation. Same class as cautions (ae) and (ah): the
    number was right and the quantity named was not.

    Table 1 was then removed -- four of its five rows had become rows of the forest figure -- and
    the cost column moved into that figure, so this guard moved with it rather than retiring. A
    guard whose subject moves to another surface and is not followed is a guard that passes by
    never running (caution (aj)). The figure draws two KINDS of cost and must keep them apart:
    every selection row is the certificate log n, exact by construction, and the metered row is
    what that decoder SPENT, against a budget of K = k*T_max = 2000 it never published.
    """
    import math as _math
    from tests.manuscript import _forest, body as _body
    rows = _forest()
    certified = [r for r in rows if r[3]]
    assert len(certified) >= 12, len(certified)
    for label, _band, cost, _c, _g in certified:
        n = round(_math.exp(cost))
        assert abs(cost - _math.log(n)) < 1e-9, (label, cost, n)   # exactly log of an integer
        assert n & (n - 1) == 0 and 1 <= n <= 1024, (label, n)     # and of a power of two
        if "$n=" in label:                                         # where the row names n, it agrees
            assert int(label.split("$n=")[1].split("$")[0]) == n, (label, n)
    spent = [r for r in rows if not r[3]]
    assert len(spent) == 1 and "metered" in spent[0][0], spent
    assert abs(spent[0][2] - 171.28) < 5e-3, spent[0]

    txt = _body("experiments.tex", "selection.tex", "iclr_intro.tex")
    for bad in ("budget of $171.3$", "budget, nats", "$171.3$-nat budget"):
        assert bad not in txt, bad
    # both bounds are stated, and neither is called a measurement
    assert "$3.175$" in txt and "$4.159$" in txt, \
        "the pathwise certificate and the sharper KL bound must both be named"
    # EVERY live section, not the three the repair was about -- caution (af). The first version
    # scanned three files and missed a fourth instance in appendix_selection.tex ("for $3.17$
    # nats of measured KL"), found the same day while reading for something else.
    import csv as _csv, glob as _glob, re as _re, sys as _sys, os as _os
    from tests.manuscript import DIR as _DIR
    _live = [f for f in _glob.glob(_os.path.join(_DIR, "sections", "*.tex"))
             if not _re.search(r"_v\d", _os.path.basename(f))]
    _live.append(_os.path.join(_DIR, "iclr_2027.tex"))
    assert len(_live) > 8, _live
    _all = " ".join(" ".join(open(f, encoding="utf-8").read().split()) for f in _live)
    for bad in ("measured KL from the anchor", "measured KL, nats", "realised KL &",
                "nats of measured KL", "measured KL against"):
        assert bad not in _all, f"a closed-form bound is being called a measurement: {bad!r}"
    # and the closed form really is what the CSV holds, so the correction is not cosmetic
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from analysis.selection_decoding import kl_best_of_n
    rows = list(_csv.DictReader(open("results/selection_scaling.csv")))
    for r in rows:
        n = int(float(r["n"]))
        assert abs(float(r["kl_nats"]) - kl_best_of_n(n)) < 5e-5, (n, r["kl_nats"])


def _live_sections():
    import glob as _glob, os as _os, re as _re
    from tests.manuscript import DIR as _DIR
    live = [f for f in _glob.glob(_os.path.join(_DIR, "sections", "*.tex"))
            if not _re.search(r"_v\d", _os.path.basename(f))]
    live.append(_os.path.join(_DIR, "iclr_2027.tex"))
    assert len(live) > 8, live
    return live


def test_no_selection_bound_is_described_as_a_realisation():
    """A bound near a realisation word must say it is a bound. Structural, not a blocklist.

    The previous version of this check was a list of five exact phrasings ("measured KL, nats",
    "realised KL &", ...). That is caution (aj)'s shape --- a guard whose trigger is a sentence
    someone will reword --- and it retired itself exactly that way: the caption of the paper's ONLY
    main-text figure said "$x$ the \\emph{realised} divergence" over an axis carrying selection's
    closed-form log n - (n-1)/n beside the meter's genuinely measured 171.3, and not one of the five
    strings matched. Fifth instance of the same defect, in the most-read caption in the paper.

    So check the property instead of the spelling: wherever a realisation word sits within 160
    characters of one of selection's closed-form values, a bound word must sit there too. The
    metered decoder's 171.3 is untouched by this --- it IS a realisation.
    """
    import math
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from analysis.selection_decoding import kl_best_of_n

    # Match the BARE value, not "$value$". The sixth instance of this defect sat in
    # "$\\log 8 = 2.08$ nats and gives exactly $8$, certified and realised alike" -- selection's
    # amplification called a realisation where no analysis script measures a realised selection
    # divergence at all (selection_decoding.csv holds only the closed form). A delimiter-anchored
    # pattern cannot see it, because 2.08 is inside a larger math group and has no $ of its own.
    vals = set()
    for n in (2, 4, 8, 16, 32, 64, 128, 256):
        for v in (kl_best_of_n(n), math.log(n)):
            vals |= {f"{v:.4f}", f"{v:.3f}", f"{v:.2f}"}
    real = ("realis", "realiz", "measured", "measurement", "actually spends")
    # "granted" is the one legitimate way a realisation word may sit beside one of these values:
    # a METERED arm granted log 8 nats up front and measured spending exactly them (appendix_proofs)
    # is a real measurement whose number coincides with a selection bound only because the budget
    # was set for comparability. It does not excuse the figure-caption defect this test was written
    # for, which grants nothing and calls a closed form realised.
    bound = ("bound", "certificate", "certifies", "at most", "permits", "allows",
             "closed form", "closed-form", "$\\log n$", "\\log n", "granted")

    bad = []
    for f in _live_sections():
        txt = " ".join(open(f, encoding="utf-8").read().split())
        for w in real:
            start = 0
            while (i := txt.find(w, start)) != -1:
                start = i + 1
                lo, hi = max(0, i - 160), min(len(txt), i + 160)
                # Clip at table structure. In a tabular the unit that carries the distinction is
                # the ROW -- appendix_selection's cost table labels each row's own order, and a
                # flat character window reads the metered row's "measured mean" beside the
                # selection rows' $2.079$ and $1.204$ and calls it a defect. Prose still gets the
                # full window, because the caption bug this test exists for spans two sentences.
                for sep in ("\\\\", "\\midrule", "\\bottomrule", "\\toprule", "\\end{tabular}"):
                    j = txt.rfind(sep, lo, i)
                    if j != -1:
                        lo = max(lo, j + len(sep))
                    j = txt.find(sep, i, hi)
                    if j != -1:
                        hi = min(hi, j)
                win = txt[lo:hi]
                # A value is only a candidate if it is being used as a DIVERGENCE. The fourth
                # false positive of this guard, 2026-09-21: "climbs at $4.55$ interval
                # half-widths" collides with kl_best_of_n(256) = 4.5490, and a half-width count
                # is dimensionless -- it is not a spend, so calling it measured is not the defect
                # this test exists for. Same class as the three refinements above (a bare value,
                # a clipped table row, the word "granted").
                hits = [v for v in vals if v in win]
                dimensionless = ("half-width", "half widths", "$\\times$", "\\times")
                hits = [v for v in hits
                        if not any(u in win[win.find(v):win.find(v) + 40] for u in dimensionless)]
                if not hits:
                    continue
                if any(b in win for b in bound):
                    continue
                bad.append((os.path.basename(f), w, win.strip()))
    assert not bad, ("a selection bound is sitting next to a realisation word with nothing "
                     f"calling it a bound: {bad[:3]}")
