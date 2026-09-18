"""feat-116: the compute-matched arm, checked against the CSVs and the pre-registration.

The paper's compute concession is now a measured frontier rather than one worst-case ratio, so the
cost column and the judged column must agree with the two files that produced them, and every band
committed before the run must have been scored afterwards.
"""
import csv
import math

PREREG = "results/onset_prediction_compute_matched.md"
GRID = (2, 4, 8, 16, 32, 64)


def _rows(path):
    return list(csv.DictReader(open(path)))


def _head():
    """Everything above the scoring rule. Split on the NEWLINE-prefixed heading: every
    pre-registration quotes `## Scoring log` in backticks on line 3, so partitioning on the bare
    string cuts at line 3 and leaves a "head" of two sentences of boilerplate (caution (r))."""
    return open(PREREG, encoding="utf-8").read().partition("\n## Scoring log")[0]


def test_the_cost_column_is_the_serving_cost_model_and_not_a_second_arithmetic():
    """compute_matched.csv prices each arm by importing serving_cost.py's constants. If the two
    files ever disagree the paper has two cost models, which is how a reviewer finds one wrong."""
    sc = {(r["scorer"], r["n"]): float(r["ratio_vs_metered"])
          for r in _rows("results/serving_cost.csv") if r["n"]}
    for r in _rows("results/compute_matched.csv"):
        if r["scorer"] == "--":
            assert float(r["cost_vs_metered"]) == 1.0, "the metered decoder is the unit"
            continue
        ref = sc[(r["scorer"], r["n"])]
        assert abs(float(r["cost_vs_metered"]) - ref) < 0.011, (r["arm"], r["cost_vs_metered"], ref)


def test_both_scorers_cover_the_committed_grid_on_the_same_prompts():
    rows = {r["arm"]: r for r in _rows("results/compute_matched.csv")}
    for s in ("05b", "7b"):
        for n in GRID:
            assert f"sel{s}_n{n}" in rows, f"sel{s}_n{n} is missing from the frontier"
    ns = {r["n_prompts"] for r in rows.values()}
    assert len(ns) == 1, f"the arms are not on one prompt set: {ns}"


def test_the_certified_nats_column_is_log_n_and_not_a_measured_divergence():
    """log n is the pathwise certificate and is exact. Quoting a measured KL here instead would
    compare selection's realised spend against the metered decoder's certified budget."""
    for r in _rows("results/compute_matched.csv"):
        if r["n"]:
            assert abs(float(r["nats_certified"]) - math.log(int(r["n"]))) < 5e-4, r["arm"]


def test_every_committed_band_was_scored():
    head, bands = _head(), _rows("results/compute_matched_bands.csv")
    scored = {b["band"].split()[0] for b in bands}
    for label in ("F1", "F2", "F3", "F4", "F5"):
        assert f"**{label} " in head, f"{label} is not committed in the pre-registration"
        assert label in scored, f"{label} was committed and never scored"
    for b in bands:
        assert b["reading"], f"{b['band']} has no reading"


def test_the_replication_gate_is_what_gates_the_others():
    """F5 pins this pass's sel7b_n64 and metered_k10 to feat-113 within the judge's own cross-pass
    floor. A pass that cannot reproduce its own reference may not price a new one, and the
    pre-registration says so before any band is quoted."""
    head = _head()
    assert "no band above is quoted" in head
    f5 = [b for b in _rows("results/compute_matched_bands.csv") if b["band"].startswith("F5")]
    assert len(f5) == 1 and f5[0]["reading"] == "REPLICATES", f5


def test_the_small_reward_cache_is_the_whole_candidate_pool():
    rows = _rows("results/selection_rewards64_qwen05b.csv")
    by = {}
    for r in rows:
        by.setdefault(r["prompt_id"], set()).add(int(r["rank"]))
    assert len(by) == 500, f"{len(by)} prompts scored, not 500"
    assert all(v == set(range(64)) for v in by.values()), "a prompt is missing candidate ranks"
    vals = [float(r["reward"]) for r in rows]
    assert len(set(round(v, 3) for v in vals)) > 100, "the 0.5B reward is near-constant"


def _tex(name):
    from tests.manuscript import tex
    return " ".join(open(tex(name), encoding="utf-8").read().split())


def test_the_appendix_frontier_table_rounds_from_the_csv():
    """Caution (j): a paper number rounds from the CSV once, and a table is checked mechanically
    rather than read. Twelve gains and twelve cost ratios."""
    apx = _tex("sections/appendix_selection.tex")
    checked = 0
    for r in _rows("results/compute_matched.csv"):
        if r["scorer"] == "--":
            continue
        assert f"${float(r['gain']):+.4f}$" in apx or \
               f"$\\mathbf{{{float(r['gain']):+.4f}}}$" in apx, (r["arm"], r["gain"])
        assert f"${float(r['cost_vs_metered']):.2f}\\times$" in apx, (r["arm"], r["cost_vs_metered"])
        checked += 1
    assert checked == 12, checked


def test_limitations_carries_the_committed_consequence_of_f1_and_f3():
    """F1 FAILS and F3 NO CROSSING both fired, and the pre-registration committed the paper to say
    that the gain is the scorer's capability and does not survive shrinking it. That half stands.
    The OTHER half of the sentence -- that the cost is intrinsic at the scales tested -- was
    retracted by feat-117, which measured saturation by 1.5B, so this test now forbids it: a
    conclusion a later arm refuted may not sit in Limitations because an earlier one committed it."""
    close = _tex("sections/iclr_closing.tex")
    assert "scorer's" in close and "capability" in close
    assert "never reaches the meter" in close
    assert "where that bar sits is open" in close, \
        "the surviving open question -- where the capability bar sits -- is not stated"
    assert "intrinsic at the scales we tested" not in close, \
        "feat-117 refuted this; saturation by 1.5B means the cost is the scorer's, not the mechanism's"
    # feat-118 then refuted the REPLACEMENT as a general claim: on GSM8K exact match a 1.5B scorer
    # buys nothing and the largest is still the best. The 1.5B figure may therefore appear only
    # attributed to the workload it was measured on, and the disagreement must appear with it --
    # this sentence is a deployer-facing recommendation and the two axes we can measure disagree.
    assert "saturates by $1.5$B" not in close, \
        "feat-118 refuted saturation as a general claim; it may not be stated unqualified"
    assert "on the judged workload" in close, "the 1.5B figure is not attributed to its workload"
    assert "nothing on GSM8K" in close, "the judge-free disagreement is not stated beside it"


def test_the_conceded_compute_ratios_were_not_softened_by_the_failed_rescue():
    """The committed consequence of NO CROSSING is that the n=64 concession stands EXACTLY as
    written, in the introduction, in Section 2 and in Limitations: a failed rescue may not be spent
    as a discount. The literal is read from the CSV rather than hardcoded, because it moved once
    already -- 57.5x was computed from the models' NAMES and is 61.3x from their parameter counts."""
    worst = max(float(r["cost_vs_metered"]) for r in _rows("results/compute_matched.csv"))
    for f in ("sections/iclr_intro.tex", "sections/selection.tex", "sections/iclr_closing.tex"):
        assert f"${worst:.1f}\\times$" in _tex(f), (f, worst)


def test_the_cost_model_uses_measured_parameter_counts_and_not_model_names():
    """Qwen2.5-7B-Instruct holds 7.6156B parameters, not 7.0, and pricing it at its name understated
    selection's serving cost by 8.8% everywhere it appeared. Every constant here must be a count
    someone took off a checkpoint; a value equal to the round number in the model's name is the bug
    this pins. Reproduce with sum(p.numel() for p in from_pretrained(<id>).parameters())."""
    from analysis.serving_cost import P_ANCHOR, P_RISKY, P_SCORER, P_SMALL
    measured = {"anchor": (P_ANCHOR, 1.7586), "risky": (P_RISKY, 8.0303),
                "scorer": (P_SCORER, 7.6156), "small": (P_SMALL, 0.4940)}
    for name, (got, want) in measured.items():
        assert abs(got - want) < 5e-4, (name, got, want)
    for name, label in (("anchor", 1.8), ("risky", 8.0), ("scorer", 7.0), ("small", 0.5)):
        if abs(measured[name][1] - label) > 5e-4:
            assert abs(measured[name][0] - label) > 5e-4, f"{name} is priced at its name again"


def test_no_single_order_gain_is_quoted_in_limitations():
    """The estimand mix this arm found: Limitations read '+0.054 against its +0.040', a SINGLE-order
    selection gain against an ORDER-AVERAGED metered gain. 0.054 is selection_scaling.csv's
    single-order n=8 value and must never reappear in a comparison; every judged gain in the closing
    must be a value some order-averaged CSV actually holds."""
    import re
    close = _tex("sections/iclr_closing.tex")
    assert "$+0.054$" not in close, "the single-order n=8 gain is back beside an order-averaged one"
    ok = {f"{float(r['gain']):+.4f}" for f in ("results/compute_matched.csv",
                                               "results/scorer_scale.csv")
          for r in _rows(f)} | {"+0.1045", "+0.0645", "+0.0400"}
    for lit in re.findall(r"\$([+-]0\.\d{3,4})\$", close):
        v = f"{float(lit):+.4f}"
        assert v in ok, f"{lit} in Limitations is in no order-averaged CSV"


def test_the_turnover_claim_is_retracted_and_not_merely_deleted():
    """feat-116 read the 0.5B curve as turning over and drew a deployer-facing warning from it.
    feat-117 registered the test and it came back FLAT, so the claim had to go. A retraction is not
    a deletion: the appendix must still say that we made the reading, that we tested it, and what
    the interval was, or the paper silently loses a negative result about itself."""
    apx = _tex("sections/appendix_selection.tex")
    assert "did not\nsurvive" in apx or "did not survive" in apx, "the retraction is not stated"
    assert "$-0.0160\\,[-0.0340,+0.0010]$" in apx, "the interval that retracted it is not quoted"
    assert "\\textsc{flat}" in apx, "the reading is not named"
    assert "$\\log n$ is not a free knob" in apx, "the withdrawn claim is not named as withdrawn"
    assert "peaks at $n=16$ and falls" not in apx, "the withdrawn claim is stated as fact again"


def test_the_saturation_finding_rounds_from_the_scorer_scale_csv():
    """The replacement claim: 1.5B over 0.5B separates and the two steps above it do not, so the
    61.3x concession is the scorer's price and not the mechanism's. Checked against the CSV."""
    apx = _tex("sections/appendix_selection.tex")
    R = {r["arm"]: r for r in _rows("results/scorer_scale.csv")}
    for tag in ("05b", "15b", "3b", "7b"):
        r = R[f"sel{tag}_n64"]
        assert f"${float(r['gain']):+.4f}$" in apx, (tag, r["gain"])
        assert f"${float(r['cost_vs_metered']):.2f}\\times$" in apx or \
               f"$\\mathbf{{{float(r['cost_vs_metered']):.2f}\\times}}$" in apx, (tag, r["cost_vs_metered"])
    frac = float(R["sel15b_n64"]["gain"]) / float(R["sel7b_n64"]["gain"])
    cost = float(R["sel15b_n64"]["cost_vs_metered"]) / float(R["sel7b_n64"]["cost_vs_metered"])
    assert f"${frac * 100:.1f}\\%$" in apx, round(frac * 100, 1)
    assert f"${cost * 100:.1f}\\%$" in apx, round(cost * 100, 1)


def test_the_matched_compute_cost_is_the_one_the_csv_computed():
    """The F4 band's label was a hardcoded '0.94x' while the same run's compute_matched.csv gave
    cost_vs_metered = 0.92 for sel05b_n4, and the main text picked up the label rather than the
    column on 2026-09-17. Caution (j): a paper number rounds from the CSV, once. Every place that
    quotes this cost must now agree with the column."""
    cost = next(float(r["cost_vs_metered"]) for r in _rows("results/compute_matched.csv")
                if r["arm"] == "sel05b_n4")
    assert cost < 1.0, "F4 is only 'matched compute' if the arm costs no more than the meter"
    bands = open("results/compute_matched_bands.csv", encoding="utf-8").read()
    assert f"({cost:.2f}x)" in bands, f"the bands label disagrees with the column ({cost})"
    body = _tex("sections/selection.tex")          # where a stale label does real damage
    assert f"${cost:.2f}\\times$ the cost" in body, \
        f"Section 2 must quote the computed matched-compute cost, {cost:.2f}x"
