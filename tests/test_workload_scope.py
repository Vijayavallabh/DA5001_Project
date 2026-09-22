"""feat-168: the headline reversal is scoped to workloads inside the anchor's support.

This is a concession -- it names a benchmark where the paper's own headline fails -- and
caution (ag) says a length edit deletes those first, while caution (aq) found six of eight
surviving deletion. So every number is derived from the CSV that measured it and the shape claims
are checked against the data rather than against their phrasing.
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import manuscript as M  # noqa: E402


def _d(name, prefix):
    path = os.path.join(ROOT, "results", f"order_averaged_h2h__{name}.csv")
    rows = [r for r in csv.reader(open(path, encoding="utf-8")) if r and r[0].startswith(prefix)]
    assert len(rows) == 1, f"{path}: {len(rows)} rows for {prefix}"
    return float(rows[0][2]), float(rows[0][3]), float(rows[0][4]), rows[0][7].strip()


def test_the_appendix_states_the_workload_where_the_reversal_fails():
    txt = M.body("appendix_selection.tex")
    got, lo, hi, reading = _d("mixpowk_judgeB", "D3")
    assert hi < 0, "D3 no longer excludes zero on the losing side; the concession is stale"
    assert reading == "REVERSAL REFUTED", reading
    # This appendix prints bands at 4 dp, as its neighbours do; carries_band matches the 3 dp
    # form the forest figure uses, so the literal is built here from the CSV instead.
    def band(g, lo_, hi_):
        return f"${g:+.4f}$ $[{lo_:+.4f}, {hi_:+.4f}]$"
    assert band(got, lo, hi) in txt, \
        f"the paired difference {band(got, lo, hi)} left the appendix"
    for prefix, arm in (("D1", "selection"), ("D2", "metered")):
        v, vlo, vhi, _ = _d("mixpowk_judgeB", prefix)
        assert band(v, vlo, vhi) in txt, \
            f"the {arm} gain on that workload ({band(v, vlo, vhi)}) left the appendix"
    assert "AlpacaEval" in txt, "the benchmark is no longer named"
    assert "support ceiling" in txt, "the mechanism of the loss is no longer named"


def test_the_body_scopes_the_claim_and_points_at_the_evidence():
    txt = M.body("experiments.tex")
    assert "not across opponents or workloads" in txt, \
        "the body's replication claim dropped the workload scope"
    assert "moving off the anchor's support" in txt, \
        "the body no longer says what kind of workload breaks it"


def test_the_discarded_first_pass_is_reported_not_buried():
    """The k=10 pass was discarded and saying so is what makes the k=1.0 choice auditable. Its two
    damning numbers -- the activity and the byte-identity rate -- must stay in the appendix."""
    txt = M.body("appendix_selection.tex")
    assert "794" in txt and "805" in txt, "the byte-identity count of the discarded pass was cut"
    assert "0.016" in txt.replace("\\%", "%"), "the discarded pass's activity rate was cut"
    assert "8.008" in txt.replace("\\%", "%"), "the chosen budget's activity rate was cut"


def test_the_calibration_curve_matches_its_csv_and_the_argmin_is_what_the_paper_says():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "mixpow_kcal.csv"),
                                    encoding="utf-8")))
    assert len(rows) == 5, len(rows)
    target = 261239 / 3118893
    best = min(rows, key=lambda r: abs(float(r["activity"]) - target))
    assert float(best["k"]) == 1.0, f"argmin moved to k={best['k']}; the appendix says 1.0"
    above = [r for r in rows if float(r["activity"]) > target]
    below = [r for r in rows if float(r["activity"]) < target]
    assert above and below, "the grid no longer brackets the target, so the choice is an endpoint"
    txt = M.body("appendix_selection.tex")
    for r in rows:  # every plotted activity is quoted to 3 or 4 dp somewhere in the paragraph
        a = float(r["activity"])
        assert f"{a:.3f}" in txt or f"{a:.4f}" in txt, \
            f"the activity at k={r['k']} ({a:.4f}) is not in the appendix"


def test_the_committed_arms_activity_is_derived_not_typed():
    """The calibration target was typed as 0.08376 from a scan of a whole k-SWEEP directory --
    pooling k=0.5, which binds on 48% of steps, and the protected classes the judge never reads.
    The arm the paper judges is k=10 over the three ordinary classes and reads 0.000080: wrong by
    a factor of 1046, and a false comparison built on it reached the compiled PDF.

    So this asserts the paper quotes the DERIVED number and never the pooled one, and it derives
    it here the same way `analysis/budget_calibration.py` does rather than trusting either.
    """
    import sys
    sys.path.insert(0, ROOT)
    from analysis.budget_calibration import committed_activity

    rate, act, tot, n = committed_activity()
    assert n == 1500, f"the reference arm no longer has 1500 trajectories ({n})"
    assert rate < 0.001, f"the committed k=10 arm now binds at {rate:.6f}; the paragraph is stale"

    txt = M.body("appendix_selection.tex")
    count = f"${act}$ of ${tot:,}$".replace(",", "{,}")
    assert count in txt, \
        f"the appendix does not quote the committed arm's own activity ({act} of {tot})"
    # The rate must sit WITH its count. A bare "0.008" check passed when the rate was deleted,
    # because "0.008" is a substring of the chosen budget's "8.008" further down the paragraph --
    # caution (an), a guard satisfied by a different occurrence of its own digits.
    i = txt.index(count)
    near = txt[i:i + 60]
    assert f"{100 * rate:.3f}" in near, \
        f"the derived rate ({100 * rate:.3f}%) is not printed beside its count: {near!r}"

    # The pooled figure must never come back, in either spelling.
    for bad in ("8.376", "0.08376"):
        assert bad not in txt, \
            f"the appendix has reacquired the pooled sweep figure {bad}; it is not an arm's rate"


def test_the_paper_does_not_claim_the_budgets_were_matched():
    """The k=1.0 arm binds about 1000x harder than the committed arm, not 'as hard as'. The
    corrected paragraph must say which, because the whole point of quoting a rate is the reader
    comparing it to the paper's own."""
    txt = M.body("appendix_selection.tex")
    assert "calibrated to bind as hard as it does" not in txt, \
        "the withdrawn 'matched budget' claim is back in the appendix"
    assert "thousand times" in txt, \
        "the appendix no longer says how much harder the chosen budget binds than the paper's own"
    body_txt = M.body("experiments.tex")
    assert "calibrated to bind as hard as" not in body_txt, \
        "the withdrawn 'matched budget' claim is in the body"


def test_the_control_that_makes_the_scoping_claim_defensible_is_in_the_paper():
    """Without the same-pipeline control on our own workload, app:workload is one benchmark
    measured once and the obvious reviewer question -- workload or pipeline? -- has no answer.
    feat-170 Arm A and Arm C supply it, and Arm B replicates the AlpacaEval reading. All four
    bands are rebuilt from their CSVs here, and the SIGNS are asserted per workload, because the
    whole claim is that the sign is constant within a workload and opposite between them."""
    txt = M.body("appendix_selection.tex")

    def d3(tag):
        rows = [r for r in csv.reader(open(os.path.join(
            ROOT, "results", f"order_averaged_h2h__{tag}.csv"), encoding="utf-8"))
            if r and r[0].startswith("D3")]
        assert len(rows) == 1, tag
        return float(rows[0][2]), float(rows[0][3]), float(rows[0][4]), rows[0][7].strip()

    ours = {"wscope_a": "k=10", "wscope_c": "k=0.9"}
    alpaca = {"mixpowk_judgeB": "k=1.0", "wscope_b": "k=1.0 re-drawn"}

    for tag, label in ours.items():
        g, lo, hi, reading = d3(tag)
        assert lo > 0, f"our workload at {label} no longer favours selection ({g:+.4f})"
        assert reading == "REVERSAL CONFIRMED", f"{tag}: {reading!r}"
    for tag, label in alpaca.items():
        g, lo, hi, reading = d3(tag)
        assert hi < 0, f"AlpacaEval at {label} no longer favours the meter ({g:+.4f})"
        assert reading == "REVERSAL REFUTED", f"{tag}: {reading!r}"

    # The two bands the paragraph prints for the control, and the replication.
    for tag in ("wscope_a", "wscope_c", "wscope_b"):
        g, lo, hi, _ = d3(tag)
        assert f"${g:+.4f}$ $[{lo:+.4f}, {hi:+.4f}]$" in txt, \
            f"{tag}'s band left the appendix; the control or the replication was trimmed"
    assert "split is the workload, not the budget, the batch size, the pipeline or the seed" in txt, \
        "the appendix no longer states what the control establishes"
    assert "opposite} directions" in txt, \
        "the observation that the two workloads respond oppositely was trimmed"


def test_no_judge_crosses_between_the_two_workloads():
    """The strongest form of the workload claim: three judges, two workloads, and every point
    estimate positive on one and negative on the other. If any judge ever crosses, the split is
    partly a judge effect and the appendix must say so -- so the SIGNS are asserted from the CSVs,
    not the prose."""
    def d3(tag):
        rows = [r for r in csv.reader(open(os.path.join(
            ROOT, "results", f"order_averaged_h2h__{tag}.csv"), encoding="utf-8"))
            if r and r[0].startswith("D3")]
        assert len(rows) == 1, tag
        return float(rows[0][2]), float(rows[0][3]), float(rows[0][4])

    pairs = (("wscope_c", "mixpowk_judgeB"), ("armc_judgeC", "mixpowk_judgeC"),
             ("armc_mixtral", "mixpowk_mixtral"))
    txt = M.body("appendix_selection.tex")
    for ours, alpaca in pairs:
        go, _lo, _hi = d3(ours)
        ga, _la, hia = d3(alpaca)
        assert go > 0, f"{ours}: our workload no longer favours selection ({go:+.4f})"
        assert ga < 0, f"{alpaca}: AlpacaEval no longer favours the meter ({ga:+.4f})"
        assert hia < 0, f"{alpaca}: no longer clears zero ({hia:+.4f})"
    # Every one of the six bands must be in the paper -- this is the claim's whole evidence.
    for tag in (t for pair in pairs for t in pair):
        g, lo, hi = d3(tag)
        assert f"${g:+.4f}$ $[{lo:+.4f}, {hi:+.4f}]$" in txt, f"{tag}'s band left the appendix"
    assert "no judge crosses" in txt, "the appendix no longer states the strongest form"


def test_the_split_is_two_judge_on_both_sides():
    """One judge on each side would leave the workload claim confounded with the judge. Judge C
    reads the same sign as judge B on our workload and on AlpacaEval, and both readings are in the
    paper. Signs come from the CSVs so a re-run that flipped one fails here, not in review."""
    txt = M.body("appendix_selection.tex")

    def d3(tag):
        rows = [r for r in csv.reader(open(os.path.join(
            ROOT, "results", f"order_averaged_h2h__{tag}.csv"), encoding="utf-8"))
            if r and r[0].startswith("D3")]
        assert len(rows) == 1, tag
        return float(rows[0][2]), float(rows[0][3]), float(rows[0][4])

    ours_b, ours_c = d3("wscope_c"), d3("armc_judgeC")
    alp_b, alp_c = d3("mixpowk_judgeB"), d3("mixpowk_judgeC")
    for label, (g, lo, _hi) in (("ours/judgeB", ours_b), ("ours/judgeC", ours_c)):
        assert lo > 0, f"{label} no longer favours selection ({g:+.4f})"
    for label, (g, _lo, hi) in (("alpaca/judgeB", alp_b), ("alpaca/judgeC", alp_c)):
        assert hi < 0, f"{label} no longer favours the meter ({g:+.4f})"
    for tag, (g, lo, hi) in (("armc_judgeC", ours_c), ("mixpowk_judgeC", alp_c)):
        assert f"${g:+.4f}$ $[{lo:+.4f}, {hi:+.4f}]$" in txt, \
            f"judge C's {tag} reading left the appendix; the split reverts to one judge"
    assert "not one judge's" in txt, "the appendix no longer says the split survives a second judge"


def test_the_paper_does_not_assert_a_mechanism_its_own_data_refutes():
    """The support-ceiling sentence was measurably wrong: nearly three quarters of the swing is the
    METER gaining, not selection losing, and the three classes of our own corpus span twice the
    competence gap with no sign change. Guard the retraction, and the two numbers that force it."""
    txt = M.body("appendix_selection.tex")
    assert "cannot identify the cause" in txt, \
        "the appendix reasserts a mechanism; its own decomposition does not support one"
    assert "$0.038$" in txt and "$0.092$" in txt, \
        "the decomposition that shows the meter moves, not selection, was trimmed"
    assert "$0.137$" in txt and "$0.065$" in txt, \
        "the within-corpus competence span that kills the predictor was trimmed"
    # And the per-class D3s must still all share a sign, or the retraction is stale.
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "workload_predictor.csv"),
                                    encoding="utf-8")))
    within = [r for r in rows if r["workload"].startswith("ours: ")]
    assert len(within) == 3, within
    assert len({r["winner"] for r in within}) == 1, \
        "the three classes no longer agree; revisit the retraction, the predictor may live"


def test_every_d3_verdict_agrees_with_its_own_interval():
    """`order_averaged_h2h.py` called ANY negative point estimate REVERSAL REFUTED without
    consulting its interval, while the mirror image -- positive, straddling zero -- was correctly
    UNRESOLVED. So -0.0065 [-0.0385, +0.0255], indistinguishable from zero, got the most definite
    word available. feat-124's log caught it in prose for that arm and the script was never fixed,
    so it mislabelled feat-173's MT-Bench cell the same way. A verdict a human must correct every
    time is one the code should not emit (caution (av))."""
    import glob
    bad = []
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "order_averaged_h2h__*.csv"))):
        for r in csv.reader(open(path, encoding="utf-8")):
            if not r or not r[0].startswith("D3"):
                continue
            g, lo, hi, got = float(r[2]), float(r[3]), float(r[4]), r[7].strip()
            want = ("REVERSAL UNRESOLVED" if lo <= 0 <= hi else
                    "REVERSAL CONFIRMED" if g > 0 else "REVERSAL REFUTED")
            if got != want:
                bad.append(f"{os.path.basename(path)}: {g:+.4f} [{lo:+.4f}, {hi:+.4f}] "
                           f"says {got!r}, arithmetic says {want!r}")
    assert not bad, "verdicts disagree with their intervals:\n  " + "\n  ".join(bad)


def test_the_mtbench_gcal_row_rounds_from_mtbenchs_own_grid():
    """The G-cal row once printed AlpacaEval's activities under MT-Bench's heading, because
    budget_calibration.py wrote both workloads to one filename and MT-Bench's grid was committed
    nowhere (caution (ax)). Read the row against the CSV mechanically -- caution (j), reading it is
    not enough -- and pin the two grids apart so the same swap cannot happen silently again."""
    import csv as _csv
    import os as _os
    import re as _re
    here = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

    def grid(path):
        return [float(r["activity"]) for r in _csv.DictReader(open(
            _os.path.join(here, "results", path), encoding="utf-8"))]

    mtb, alp = grid("mtb_kcal.csv"), grid("mixpow_kcal.csv")
    assert len(mtb) == len(alp) == 5
    # The two workloads must not be confusable at the precision the document prints.
    assert sum(1 for a, b in zip(mtb, alp) if round(a, 3) == round(b, 3)) < 3, \
        "the two grids agree at 3 dp on most points; the swap this guards against is undetectable"

    doc = open(_os.path.join(here, "results", "onset_prediction_third_workload.md"),
               encoding="utf-8").read()
    row = [ln for ln in doc.splitlines() if ln.startswith("| G-cal |")]
    assert len(row) == 1, "expected exactly one G-cal row in the scored table"
    quoted = [float(x) for x in _re.findall(r"`([0-9]*\.[0-9]+)`", row[0])]
    # the target is quoted as a percentage with a % sign and so is not picked up here
    assert len(quoted) == 5, f"expected five activities in the G-cal row, got {quoted}"
    for q in quoted:
        assert any(abs(q - m) < 5e-4 for m in mtb), \
            f"{q} is not an MT-Bench activity; mtb_kcal.csv holds {mtb}"
        assert not (any(abs(q - a) < 5e-4 for a in alp)
                    and not any(abs(q - m) < 5e-4 for m in mtb)), q


def test_the_two_calibration_files_belong_to_different_roots():
    """One file per workload is the repair; assert the trajectory counts differ, which is the
    cheapest thing that cannot be true if one overwrote the other."""
    import csv as _csv
    import os as _os
    here = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    def n(path):
        return {int(r["n_trajectories"]) for r in _csv.DictReader(open(
            _os.path.join(here, "results", path), encoding="utf-8"))}
    assert n("mtb_kcal.csv") == {80}, "MT-Bench is 80 prompts"
    assert n("mixpow_kcal.csv") == {200}, "the AlpacaEval calibration is 200 prompts"


def test_every_degeneracy_number_in_the_appendix_rounds_from_its_own_csv():
    """Three numbers carried the vacuity argument across three workloads with NO producing script:
    `794 of 805`, `4.3%`, `8.008%`, and the committed arm's `24 of 299,843`. Each was computed once
    by hand into a scoring log -- caution (ai) crossed with caution (j). They now come from
    results/workload_degeneracy.csv and this reads the paragraph against it mechanically.
    """
    import csv as _csv
    import os as _os
    from tests.manuscript import ROOT, body
    rows = list(_csv.DictReader(open(
        _os.path.join(ROOT, "results", "workload_degeneracy.csv"), encoding="utf-8")))
    by = {(r["workload"], r["budget"]): r for r in rows}
    txt = body("appendix_selection.tex")

    # The committed arm: the rate AND its count, which is what makes the rate checkable.
    c = by[("ours (committed)", "vacuous")]
    assert c["steps_active"] == "24" and c["steps_total"] == "299843"
    assert f"{float(c['activity']):.3%}".rstrip("%") in ("0.008",), c["activity"]

    # AlpacaEval at the paper's own budget: the byte-identity count the appendix leads with.
    a = by[("AlpacaEval", "vacuous")]
    assert a["n_byte_identical"] == "794" and a["n_prompts"] == "805"
    assert a["steps_active"] == "26" and a["steps_total"] == "160227"
    assert "794 of 805" in txt or ("$794$" in txt and "$805$" in txt), \
        "the byte-identity count moved out of the paragraph without this guard moving with it"

    # AlpacaEval at the binding budget: the chosen rate and the share it matched on.
    b = by[("AlpacaEval", "binding")]
    assert round(float(b["activity"]) * 100, 3) == 8.008, b["activity"]
    assert round(float(b["byte_ident"]) * 100, 1) == 4.3, b["byte_ident"]

    # Our own corpus at its binding budget.
    o = by[("ours", "binding")]
    assert round(float(o["activity"]) * 100, 1) == 8.9, o["activity"]
    assert round(float(o["byte_ident"]) * 100, 1) == 2.0, o["byte_ident"]


def test_the_one_uninterpretable_byte_identity_row_is_flagged_as_such():
    """The committed pass samples its metered arm and its opponent independently, so at k=10 it
    reads 2.8% byte-identical while wscope's k=10 arm -- same workload, same budget -- reads 99.5%.
    Exactly one row may be flagged, and it must be that one: if the flag ever spreads, the
    signature it is derived from has stopped meaning what it means."""
    import csv as _csv
    import os as _os
    from tests.manuscript import ROOT
    rows = list(_csv.DictReader(open(
        _os.path.join(ROOT, "results", "workload_degeneracy.csv"), encoding="utf-8")))
    flagged = [r for r in rows if r["byte_ident_interpretable"] != "yes"]
    assert [r["workload"] for r in flagged] == ["ours (committed)"], \
        f"expected exactly the committed pass to be flagged, got {[r['workload'] for r in flagged]}"
    # and the two rows that make the point must disagree as sharply as the appendix says
    k10 = {r["workload"]: float(r["byte_ident"]) for r in rows if r["budget"] == "vacuous"}
    assert k10["ours (committed)"] < 0.10 < 0.90 < k10["ours"], k10


def test_the_third_workload_paragraph_carries_its_bands_and_its_shape_claim():
    """MT-Bench entered the appendix on 2026-09-22. Caution (af): a claim added without a guard is
    the first thing a length edit deletes, and caution (ai): the CLAIM ABOUT a set of numbers is
    what nothing checks. Both bands and the decomposition's shape are rebuilt from their CSVs."""
    import csv as _csv
    import os as _os
    from tests.manuscript import ROOT, body, carries_band

    def gains(tag):
        d = {r["quantity"][:2]: r for r in _csv.DictReader(open(_os.path.join(
            ROOT, "results", f"order_averaged_h2h__{tag}.csv"), encoding="utf-8"))}
        return {k: (float(v["value"]), float(v["lo95"]), float(v["hi95"]))
                for k, v in d.items() if k in ("D1", "D2", "D3")}

    sec = "appendix_selection.tex"
    # the three judges on MT-Bench's binding cell, all unresolved
    for tag in ("mtb_conc_bind", "mtb_conc_bind_judgeC", "mtb_conc_bind_mixtral"):
        g, lo, hi = gains(tag)["D3"]
        assert lo < 0 < hi, f"{tag} no longer straddles zero; the paragraph says all three do"
        assert carries_band(g, lo, hi, sec), f"the paragraph dropped {tag}'s band {g} [{lo}, {hi}]"
    # and the k=10 cell, which does resolve
    g, lo, hi = gains("mtb_conc_k10")["D3"]
    assert hi < 0, "the MT-Bench k=10 cell no longer resolves against the meter"
    assert carries_band(g, lo, hi, sec), "the k=10 band was cut"

    # THE SHAPE CLAIM: the meter's gain spans more than selection's, and it is the meter that
    # crosses -- below selection on our corpus, above it on both external ones.
    ours, alp, mtb = gains("wscope_c"), gains("mixpowk_judgeB"), gains("mtb_conc_bind")
    sel = [x["D1"][0] for x in (ours, alp, mtb)]
    met = [x["D2"][0] for x in (ours, alp, mtb)]
    assert all(v > 0 for v in sel), "the paragraph says selection gains on all three"
    span_s, span_m = max(sel) - min(sel), max(met) - min(met)
    assert round(span_s, 3) == 0.056 and round(span_m, 3) == 0.092, (span_s, span_m)
    assert met[0] < sel[0] and met[1] > sel[1] and met[2] > sel[2], \
        "the meter no longer crosses the way the paragraph says it does"
    # SCOPE THE CHECK TO ITS OWN SENTENCE (caution (an)). `"$0.092$" in txt` passed a mutation
    # that changed this sentence's span to 0.091, because the OLDER two-workload decomposition
    # 10 lines below prints $0.092$ as well -- a guard satisfied by a different occurrence of its
    # own number is not guarding its sentence.
    txt = body(sec)
    i = txt.find("the decomposition holds a third time")
    assert i > 0, "the third-workload decomposition sentence is gone"
    claim = txt[i:i + 420]
    assert f"${span_s:.3f}$" in claim and f"${span_m:.3f}$" in claim, \
        f"the two spans no longer round from their CSVs in: {claim[:200]}"
    assert "gains on all three" in claim, \
        "the sentence no longer says selection gains on all three, which its own CSVs do"


def test_the_prompt_template_is_excluded_by_two_header_free_arms():
    """A reviewer's obvious objection -- your workload gets a different prompt -- is answered by
    our own corpus's factual class, which goes through the SAME slot AlpacaEval does. The claim is
    a shape over four arms, so it is rebuilt from the CSV rather than read (caution (ai)), and the
    header fractions come from what the models were SERVED, not from a metadata field."""
    import csv as _csv
    import os as _os
    from tests.manuscript import ROOT, body
    rows = {r["arm"]: r for r in _csv.DictReader(open(
        _os.path.join(ROOT, "results", "prompt_header_audit.csv"), encoding="utf-8"))}
    assert set(rows) == {"ours: neutral", "ours: creative", "ours: factual", "AlpacaEval"}

    free = [r for r in rows.values() if float(r["header"]) == 0.0]
    assert {r["arm"] for r in free} == {"ours: factual", "AlpacaEval"}, \
        "the two header-free arms are what excludes the template; they moved"
    assert {r["winner"] for r in free} == {"selection", "meter"}, \
        "the two header-free arms no longer disagree, so the exclusion no longer holds"
    for r in free:                       # and each interval must clear zero, in its own direction
        lo, hi = float(r["lo95"]), float(r["hi95"])
        assert lo * hi > 0, f"{r['arm']} now straddles zero: [{lo}, {hi}]"

    # the header-carrying classes must agree with the header-free one, or the paragraph's second
    # half is false even though its first half survives
    carried = [r for r in rows.values() if float(r["header"]) == 1.0]
    assert len(carried) == 2 and all(float(r["d3"]) > 0 for r in carried)

    claim = body("appendix_selection.tex")
    i = claim.find("Nor is it the prompt")
    assert i > 0, "the template-exclusion sentence was cut"
    claim = claim[i:i + 1000]
    # THE TWO HEADER-FREE ARMS CARRY THE ARGUMENT, so they must appear with their INTERVALS --
    # a bare value is not enough here, and not only in principle: `$+0.0990$` occurs twice in this
    # very sentence (once as the band, once in the comparison), so perturbing the band alone left
    # a guard on the bare value passing. Caution (an), inside a guard written for caution (ai).
    from tests.manuscript import carries_band
    for r in free:
        g, lo, hi = float(r["d3"]), float(r["lo95"]), float(r["hi95"])
        assert carries_band(g, lo, hi, "appendix_selection.tex"), \
            f"{r['arm']}'s band {g} [{lo}, {hi}] left the paragraph"
        assert f"${g:+.4f}$ $[{lo:+.4f}, {hi:+.4f}]$" in claim, \
            f"{r['arm']}'s band is printed somewhere else, not in this sentence"
    for r in carried:                    # the agreeing classes need only their values
        assert f"${float(r['d3']):+.4f}$" in claim, f"{r['arm']}'s gain left the paragraph"
    assert "$500$" in claim and "$850$" in claim, "the class sizes that make the point were cut"


def test_the_support_ceiling_direction_is_flagged_where_it_is_stated():
    """Caution (ao): the same file said the opposite budget response is 'what a support ceiling
    predicts' and, two paragraphs later, that the support account is measurably wrong. Both
    sentences were true of their own quantity and the pair read as a contradiction. The flag has to
    live WITH the observation, not only in the refutation, or a reader meets them in the wrong
    order."""
    from tests.manuscript import body
    txt = body("appendix_selection.tex")
    i = txt.find("respond to a binding budget in \\emph{opposite} directions")
    assert i > 0, "the opposite-direction observation was cut"
    near = txt[i:i + 700]
    assert "refutes" in near or "refuted" in near, \
        "the observation no longer says the decomposition below refutes the account it matches"
    j = txt.find("across five passes")
    assert j > 0, "the five-passes claim was cut"
    assert "these two" in txt[j:j + 160], \
        "the five-passes claim is no longer scoped to the two workloads it is about; MT-Bench and " \
        "any later workload are not among those five"
