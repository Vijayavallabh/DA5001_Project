"""feat-162: the measured compute-matched cell.

The arm's whole point is that a cell chosen by parameter counts is not the cell a clock chooses,
so these guard the three places that choice can go wrong: the fit that removes the model loader,
the gate order (nothing below a failed gate may be read), and the argmin rule that picks the cell
before any judged number is looked at.
"""
import csv
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.cost_grid import GRID, fit_line, parse, report  # noqa: E402

SEVEN, HALF = "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"


def _gen_dir(path, n_prompts=40, tokens=200):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "t.jsonl"), "w", encoding="utf-8") as fh:
        for p in range(n_prompts):
            fh.write(json.dumps({"metadata": {"prompt_id": f"p{p}"},
                                 "aggregate": {"generation_length_tokens": tokens}}) + "\n")
    return path


def _log(tmp, a=10.0, b=7.0, met1=26.0, met2=38.0, rew7=(84.0 / 64), rew05=(5.5 / 64),
         met_tokens=200):
    """A synthetic box: draws(n) = a + b n, metered one and two completions, reward linear in n."""
    lines = []
    for rep in (1, 2):
        for n in GRID:
            d = _gen_dir(os.path.join(tmp, f"n{n}_r{rep}"), tokens=200)
            lines.append(f"[cost] DRAWS rep={rep} n={n} seconds={a + b * n:.3f} dir={d}")
        for tpp, s in ((1, met1), (2, met2)):
            d = _gen_dir(os.path.join(tmp, f"met{tpp}_r{rep}"), tokens=met_tokens)
            lines.append(f"[cost] MET rep={rep} tpp={tpp} seconds={s:.3f} dir={d}")
        for model, per in ((SEVEN, rew7), (HALF, rew05)):
            for n in GRID:
                lines.append(f"[cost] REWARD rep={rep} n={n} model={model} "
                             f"load_s=9.100 score_s={per * n:.3f} items={n * 40} prompts=40")
    p = os.path.join(tmp, "cost_grid.log")
    open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n[cost] DONE 12:00:00\n")
    return p


def _fake_per_prompt(path, n=40):
    """A stand-in for results/compute_matched_per_prompt.csv: the bootstrap replay is exercised
    against the real file by test_the_paired_replay_lands_on_the_committed_f4, and these cases are
    about the gates and the cell choice, which must not pay for 15 bootstraps over 500 prompts."""
    arms = [f"sel{s}_n{n_}" for s in ("05b", "7b") for n_ in (2, 4, 8, 16, 32, 64)]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["prompt_id"] + [f"gain_{a}" for a in arms + ["metered_k10"]])
        for i in range(n):
            w.writerow([f"p{i}"] + [round(0.01 * (j + 1) + 0.001 * i, 4)
                                    for j in range(len(arms) + 1)])
    return path


def _bands(out):
    return {r["band"].split()[0]: r for r in
            csv.DictReader(open(os.path.join(out, "cost_grid_bands.csv"), encoding="utf-8"))}


def test_the_fit_recovers_a_line_and_reports_its_own_quality():
    a, b, r2 = fit_line([1, 2, 4, 8, 16, 64], [10 + 7 * x for x in (1, 2, 4, 8, 16, 64)])
    assert abs(a - 10) < 1e-9 and abs(b - 7) < 1e-9 and r2 > 0.999999
    # a fit through a flat line has no slope to find, and R^2 must say so rather than read 1.0
    _, b2, r22 = fit_line([1, 2, 4], [5.0, 5.0, 5.0])
    assert abs(b2) < 1e-12 and r22 == 0.0
    with pytest.raises(AssertionError):
        fit_line([1, 1, 1], [1.0, 2.0, 3.0])


def test_parse_reads_the_three_cell_kinds_and_nothing_else(tmp_path):
    log = _log(str(tmp_path))
    open(log, "a", encoding="utf-8").write(
        "[cost] REWARD rep=9 n=99 model=nope load_s=NA score_s=2\n"
        "some transformers warning about generation_length_tokens seconds=99999\n")
    draws, met, rew = parse(log)
    assert len(draws) == 2 * len(GRID) and len(met) == 4
    assert len(rew) == 2 * 2 * len(GRID), "a malformed reward line must not enter the set"
    assert max(m[2] for m in met) < 100, "a warning line was parsed as a timing"


def test_the_loader_is_removed_and_the_marginal_ratio_is_not_the_raw_one(tmp_path):
    out = str(tmp_path / "out")
    report(_log(str(tmp_path)), out, _fake_per_prompt(str(tmp_path / 'pp.csv')))
    rows = list(csv.DictReader(open(os.path.join(out, "cost_grid.csv"), encoding="utf-8")))
    n2 = [r for r in rows if r["n"] == "2" and r["scorer_b"] == "0.494"][0]
    # draws(2) = 10 + 14 = 24 s raw against a marginal 14 s; b_met = 12 s. The two ratios must
    # differ, or the decomposition this arm rests on has done nothing.
    assert abs(float(n2["ratio_marginal"]) - (2 * 7 + 2 * 5.5 / 64) / 12.0) < 1e-3
    assert float(n2["ratio_raw"]) > float(n2["ratio_marginal"])
    fit = _bands(out)["G-fit:"]
    assert "b=7.00s" in fit["value"] and "b_met=12.00s" in fit["value"]


def test_b1_picks_the_cell_nearest_one_and_b2_prices_the_paper_s_cell(tmp_path):
    out = str(tmp_path / "out")
    report(_log(str(tmp_path)), out, _fake_per_prompt(str(tmp_path / 'pp.csv')))
    b = _bands(out)
    assert b["G0"]["reading"] == "PASS" and b["G1"]["reading"] == "PASS"
    assert b["G3"]["reading"] == "PASS"
    # b = 7 s per completion against b_met = 12 s, so n=2 is 1.17x and n=1 is 0.58x: the window
    # [0.70, 1.45] admits only n=2, which is also the argmin.
    assert b["B1"]["value"] == "Qwen2.5-0.5B-Instruct n=2", b["B1"]["reading"]
    assert "MATCHED" in b["B1"]["reading"]
    # the manuscript prints 0.92x for n=4, which here measures 2.34x
    assert b["B2"]["reading"].startswith("MISPRICED"), b["B2"]["reading"]
    assert abs(float(b["B2"]["value"]) - (4 * 7 + 4 * 5.5 / 64) / 12.0) < 1e-3
    # the FACTOR is what the repair will quote, and a verdict string cannot guard a number
    # (caution (av)): 2.362 / 0.92 = 2.57
    assert "a factor of 2.57" in b["B2"]["reading"], b["B2"]["reading"]
    assert "against 0.92x printed" in b["B2"]["reading"]


def test_a_failed_gate_stops_every_band_below_it(tmp_path):
    """caution (as): a gate that fails must stop the reading, not merely annotate it."""
    out = str(tmp_path / "out")
    # make the reward pass dominate the clock: G1 fails, so B1 and B2 must not be computed
    report(_log(str(tmp_path), rew7=10.0), out, _fake_per_prompt(str(tmp_path / 'pp.csv')))
    b = _bands(out)
    assert b["G1"]["reading"] == "FAIL"
    assert "NOT SCORED" in b["B1"]["reading"]
    assert "B2" not in b, "a band was computed under a failed gate"


def test_g2_catches_two_cells_that_did_not_serve_the_same_work(tmp_path):
    out = str(tmp_path / "out")
    report(_log(str(tmp_path), met_tokens=120), out, _fake_per_prompt(str(tmp_path / 'pp.csv')))
    assert _bands(out)["G2"]["reading"] == "FAIL"


def test_the_registration_is_committed_and_states_the_withdrawal_branch():
    """B3 has to name the branch where the concession is withdrawn, or finding it later reads as
    a rescue rather than as a result (caution (ag): concessions are what a revision deletes)."""
    t = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "onset_prediction_cost_matched_measured.md"),
             encoding="utf-8").read()
    head = t.partition("\n## Scoring log")[0]
    assert "CONCESSION WITHDRAWN" in head and "CONCESSION STANDS" in head
    assert "argmin |ratio - 1|" in head, "B1's rule must be fixed before the data"
    assert "does not replace it" in head, "the committed 35.4x is a local measurement"


def test_the_proxy_ratio_is_the_one_the_manuscript_actually_prints():
    """PROXY_N4 is a number copied out of the paper, and a copied number goes stale silently.
    If Section 5 ever reprints its compute-matched cost, this fails and says to re-derive the
    factor rather than letting B2 compare against a ratio nobody prints any more."""
    from tests.manuscript import body
    txt = body("selection.tex")
    from analysis.cost_grid import PROXY_N4
    assert f"the forward-pass count had put it at $n=4$" in txt and f"${PROXY_N4}" not in txt \
        or f"${PROXY_N4}\\times$" in txt, (
        "selection.tex no longer prints the ratio cost_grid.py measures against. It fired once "
        "already, when feat-162 rewrote the sentence, which is what it is for -- re-derive the "
        "factor B2 reports rather than relaxing this.")


def test_g0_fails_when_the_two_paths_are_implausibly_close(tmp_path):
    """The instrument gate is what catches a cell that is not the arm it claims to be -- a
    selection path measuring about the meter's own cost means the draws never happened. It has to
    fail loudly there, and take every band below it down (caution (as))."""
    out = str(tmp_path / "out")
    report(_log(str(tmp_path), b=0.5, rew7=0.01), out, _fake_per_prompt(str(tmp_path / 'pp.csv')))          # ratio(64) ~ 2.7x, far below 15
    b = _bands(out)
    assert b["G0"]["reading"] == "FAIL", b["G0"]
    assert "NOT SCORED" in b["B1"]["reading"] and "B2" not in b


def test_the_paired_replay_lands_on_the_committed_f4():
    """B3 substitutes a CPU re-pairing of committed utilities for the registration's GPU re-judge.
    What licenses the substitution is that replaying compute_matched.py's bootstrap stream
    reproduces its F4 band EXACTLY -- point estimate and both ends. If it ever stops doing so the
    stream is not that script's and nothing drawn from it may be quoted."""
    from analysis.cost_grid import F4, paired_at
    m, lo, hi, replicates = paired_at("sel05b_n4")
    assert replicates, "the replay no longer reproduces the committed F4"
    assert abs(m - F4[0]) < 1e-9, "the point estimate is a mean and does not depend on the draw"
    m1, lo1, hi1, _ = paired_at("sel05b_n1")
    assert lo1 is None and abs(m1 + 0.0400) < 1e-9, "n=1 IS the control: its gain is 0, not judged"


def test_the_proxy_column_is_compute_matched_s_own_arithmetic_and_not_retyped():
    """cost_grid.py prints the FLOP proxy beside the measured ratio so the two can be compared, and
    a proxy retyped from memory is not the proxy the paper printed -- caution (ax). Reproduce
    compute_matched.csv's own cost_vs_metered column, every cell."""
    import csv as _csv
    import os as _os
    from analysis.serving_cost import P_ANCHOR, P_RISKY
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    rows = list(_csv.DictReader(open(_os.path.join(root, "results", "compute_matched.csv"),
                                     encoding="utf-8")))
    seen = 0
    for r in rows:
        if not r["n"]:
            continue
        pb = {"Qwen2.5-0.5B": 0.4940, "Qwen2.5-7B": 7.6156}[r["scorer"]]
        got = round(int(r["n"]) * (P_ANCHOR + pb) / (P_ANCHOR + P_RISKY), 3)
        assert abs(got - float(r["cost_vs_metered"])) < 1e-9, (r["arm"], got, r["cost_vs_metered"])
        seen += 1
    assert seen == 12, f"expected the 0.5B and 7B grids, saw {seen} cells"


def _grid():
    import csv as _csv
    import os as _os
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    rows = list(_csv.DictReader(open(_os.path.join(root, "results", "cost_grid.csv"),
                                     encoding="utf-8")))
    return {(r["scorer"], int(r["n"])): r for r in rows}


def test_every_cost_number_the_paper_prints_rounds_from_the_measured_csv():
    """caution (j): a paper number rounds from its CSV, once. These are the cells Section 5 and
    Appendix I now print, and each is checked against the column it came out of rather than
    against a plausible story about it."""
    from tests.manuscript import body
    g = _grid()
    txt = body("selection.tex", "appendix_selection.tex", "appendix_related.tex",
               "iclr_closing.tex")
    for scorer, n, col, printed in (
            ("Qwen2.5-0.5B-Instruct", 1, "ratio_marginal", "1.07"),
            ("Qwen2.5-0.5B-Instruct", 4, "ratio_marginal", "4.11"),
            ("Qwen2.5-7B-Instruct", 64, "ratio_marginal", "67.24"),
            ("Qwen2.5-7B-Instruct", 64, "ratio_raw", "26.93"),
            ("Qwen2.5-0.5B-Instruct", 64, "draws_s", "458.02"),
            ("Qwen2.5-0.5B-Instruct", 4, "draws_s", "38.66")):
        val = float(g[(scorer, n)][col])
        assert f"{val:.2f}" == printed, (scorer, n, col, val, printed)
        assert printed in txt, f"the paper stopped printing {printed} ({scorer} n={n} {col})"
    # Section 5 rounds the same cells to one and two decimals; both must round from the column
    assert f"{float(g[('Qwen2.5-7B-Instruct', 64)]['ratio_marginal']):.1f}" == "67.2"
    assert f"{float(g[('Qwen2.5-0.5B-Instruct', 4)]['ratio_marginal']):.1f}" == "4.1"


def test_the_abstract_s_convention_factor_is_the_measured_one():
    """The abstract says the price is 2.5x more per request. That factor is the ratio of the two
    conventions at the headline cell, and it is what a deployer reads as a price -- caution (aq),
    where exactly this sentence was the mutation that passed."""
    import os as _os
    g = _grid()
    r = g[("Qwen2.5-7B-Instruct", 64)]
    factor = float(r["ratio_marginal"]) / float(r["ratio_raw"])
    assert f"{factor:.1f}" == "2.5", factor
    tex = _os.environ.get("SATML_DIR") or _os.path.normpath(
        _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                      _os.pardir, "sub", "satml"))
    abstract = " ".join(open(_os.path.join(tex, "iclr_2027.tex"), encoding="utf-8").read().split())
    assert r"$21.8\times$ for a server that runs only the anchor" in abstract, (
        "the abstract dropped the deployable price, which is the number a deployer reads")


def test_the_batching_explanation_is_withdrawn_and_not_merely_deleted():
    """caution (af): a withdrawn claim must be chased, and a withdrawal that simply deletes the
    sentence leaves a reader of the earlier version with no correction. The appendix keeps the
    claim and says it was falsified, so this asserts the two travel together."""
    from tests.manuscript import body
    txt = body("appendix_selection.tex")
    i = txt.find("do not cost\n$64$ sequential ones".replace("\n", " "))
    assert i > 0, "the batching sentence is gone entirely; it should be kept and marked falsified"
    window = txt[max(0, i - 400):i + 400]
    assert "falsifies it" in window, "the batching claim is stated without its withdrawal"
    assert "linear in $n$" in txt and "0.99990" in txt, (
        "the evidence for the withdrawal -- that draws are linear in n -- is not printed")


def test_the_compute_matched_concession_survives_with_both_prices():
    """The concession is the most damaging sentence in the paper and caution (ag) says a length
    edit deletes those first. It must keep its measured loss AND now carry both prices of the cell
    it is measured at, so neither the number nor its correction can go missing alone."""
    from tests.manuscript import body
    txt = body("selection.tex")
    assert "$-0.0395$ $[-0.0720, -0.0065]$" in txt, "the compute-matched loss was dropped"
    assert r"where the forward-pass count had put it at $n=4$ and $-0.0395$" in txt, (
        "the superseded cell and its loss no longer travel with the measured one")
    assert r"the matched cell is $n=2$ at $0.68\times$" in txt, (
        "the deployable matched cell was dropped")
    assert r"$-0.0330$ $[-0.0625, -0.0025]$" in txt, "its paired difference was dropped"


# ---------------------------------------------------------------------------------------------
# feat-163: does batching move the price RATIO, or only the bill?
# ---------------------------------------------------------------------------------------------

def _wlog(tmp, ca, cm, load=10.0, tokens=200, served=None):
    """A synthetic box: at width W one anchor batch costs ca(W) and one metered batch cm(W).
    tpp=1 is load + one batch, tpp=2 is load + two, so differencing them recovers the batch."""
    from analysis.cost_grid import WIDTHS
    lines = []
    for rep in (1, 2):
        for w in WIDTHS:
            for path, c in (("ANCHOR", ca), ("MET", cm)):
                for tpp in (1, 2):
                    d = _gen_dir(os.path.join(tmp, f"{path}{w}_{tpp}_{rep}"),
                                 n_prompts=(served or w), tokens=tokens)
                    lines.append(f"[width] {path} rep={rep} W={w} tpp={tpp} "
                                 f"seconds={load + tpp * c(w):.3f} dir={d}")
    p = os.path.join(tmp, "batch_width.log")
    open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n[width] DONE\n")
    return p


def _wbands(out):
    return {r["band"].split()[0]: r for r in
            csv.DictReader(open(os.path.join(out, "batch_width_bands.csv"), encoding="utf-8"))}


def test_the_batch_cost_is_differenced_not_fitted_and_the_loader_never_enters(tmp_path):
    from analysis.cost_grid import report_width
    out = str(tmp_path / "out")
    # anchor flat in W (overhead-bound), meter growing with W (weight-bound): the ratio must fall
    report_width(_wlog(str(tmp_path), lambda w: 7.0, lambda w: 7.0 + 0.05 * w, load=999.0), out)
    rows = {int(r["width"]): r for r in
            csv.DictReader(open(os.path.join(out, "batch_width.csv"), encoding="utf-8"))}
    assert abs(float(rows[64]["anchor_batch_s"]) - 7.0) < 1e-6, "a 999s loader reached the cost"
    assert abs(float(rows[64]["metered_batch_s"]) - 10.2) < 1e-6
    assert abs(float(rows[8]["c_a_over_c_m"]) - 7.0 / 7.4) < 1e-4


def test_b1_reads_three_ways_and_b2_scales_the_ratio_by_sixty_four(tmp_path):
    from analysis.cost_grid import report_width
    out = str(tmp_path / "out")
    report_width(_wlog(str(tmp_path), lambda w: 7.0, lambda w: 7.0 + 0.05 * w), out)
    b = _wbands(out)
    assert b["G0"]["reading"] == "PASS" and b["G1"]["reading"] == "PASS"
    assert b["B1"]["reading"] == "FALLS", b["B1"]
    assert abs(float(b["B2"]["value"]) - 64 * (7.0 / 17.0)) < 0.02, b["B2"]
    assert "NOT A PRICE" in b["B2"]["reading"], "B2 must carry its own disclaimer"
    # both paths scaling together leaves the ratio flat -- the point of the arm
    out2 = str(tmp_path / "out2")
    # both grow identically; c_a(64) = 6.972, inside G0, so the arm is scored and reads FLAT
    report_width(_wlog(str(tmp_path / "b"), lambda w: 5.5 + 0.023 * w,
                       lambda w: 5.5 + 0.023 * w), out2)
    assert _wbands(out2)["B1"]["reading"] == "FLAT"


def test_g0_catches_a_width_arm_that_disagrees_with_the_arm_it_extends(tmp_path):
    from analysis.cost_grid import report_width
    out = str(tmp_path / "out")
    report_width(_wlog(str(tmp_path), lambda w: 2.0, lambda w: 7.0), out)   # c_a(64)=2.0 vs 6.99
    b = _wbands(out)
    assert b["G0"]["reading"] == "FAIL"
    assert "NOT SCORED" in b["B1"]["reading"] and "B2" not in b


def test_g1_catches_a_batch_that_was_split(tmp_path):
    """--cap-neutral W with --batch-size W is one batch per seed group only if the run really
    served W requests. If it served fewer the width is not W and the whole axis is wrong."""
    from analysis.cost_grid import report_width
    out = str(tmp_path / "out")
    report_width(_wlog(str(tmp_path), lambda w: 7.0, lambda w: 7.0 + 0.05 * w, served=8), out)
    assert _wbands(out)["G1"]["reading"].startswith("FAIL")


# ---------------------------------------------------------------------------------------------
# feat-164: the harness forwards BOTH models on every path, so what does a selection server pay?
# ---------------------------------------------------------------------------------------------

def _aolog(tmp, b, c, load=9.0, seqs=None):
    lines = []
    for rep in (1, 2):
        for w in (64, 200):
            for tpp in (1, 2):
                d = _gen_dir(os.path.join(tmp, f"B{w}_{tpp}_{rep}"), n_prompts=w)
                lines.append(f"[ao] B rep={rep} W={w} tpp={tpp} seconds={load + tpp * b(w):.3f} "
                             f"dir={d}")
            lines.append(f"[anchor] rep={rep} W={w} load_s=3.000 gen_s={c(w):.3f} "
                         f"seqs={seqs or w} new_tokens=200")
    p = os.path.join(tmp, "anchor_only.log")
    open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n[ao] DONE\n")
    return p


def _aobands(out):
    return {r["band"].split()[0]: r for r in
            csv.DictReader(open(os.path.join(out, "anchor_only_cost.csv").replace(
                "anchor_only_cost.csv", "anchor_only_cost_bands.csv"), encoding="utf-8"))}


def test_the_discarded_model_share_and_the_deployable_cost_are_subtractions_not_stories(tmp_path):
    """A is the harness path that every published draw cost came from; B is the same loop with the
    8B swapped for the anchor; C is a plain generate with the anchor alone. (A-B)/A is the cost of
    the model the k=0 path forwards and discards, and C/A is what that overstatement is worth."""
    from analysis.cost_grid import report_anchor
    out = str(tmp_path / "out")
    wl = _wlog(str(tmp_path / "w"), lambda w: 7.0, lambda w: 7.1)      # A = 7.0, D = 7.1
    report_anchor(_aolog(str(tmp_path / "a"), lambda w: 2.1, lambda w: 1.4), wl, out)
    rows = {int(r["width"]): r for r in
            csv.DictReader(open(os.path.join(out, "anchor_only_cost.csv"), encoding="utf-8"))}
    assert abs(float(rows[200]["discarded_model_share"]) - (7.0 - 2.1) / 7.0) < 1e-4
    assert abs(float(rows[200]["C_over_A"]) - 1.4 / 7.0) < 1e-4
    b = _aobands(out)
    assert b["B1"]["reading"] == "DOMINANT", b["B1"]
    assert "overstates a deployment by 5.00x" in b["B2"]["reading"], b["B2"]


def test_b3_rederives_the_matched_cell_on_the_deployable_cost(tmp_path):
    """C/D = 0.197, so n=4 lands at 0.79 and n=8 at 1.58: the argmin is n=4, which is the cell the
    manuscript originally claimed. The band must say so rather than quietly agreeing with it."""
    from analysis.cost_grid import report_anchor
    out = str(tmp_path / "out")
    wl = _wlog(str(tmp_path / "w"), lambda w: 7.0, lambda w: 7.1)
    report_anchor(_aolog(str(tmp_path / "a"), lambda w: 2.1, lambda w: 1.4), wl, out)
    b = _aobands(out)
    assert b["B3"]["value"] == "4", b["B3"]
    assert "matched at n=4" in b["B3"]["reading"], b["B3"]
    assert "-0.0" in b["B3"]["reading"], "the judged gain at the re-derived cell is not reported"


def test_g0b_catches_a_plain_generate_that_did_not_serve_the_width(tmp_path):
    from analysis.cost_grid import report_anchor
    out = str(tmp_path / "out")
    wl = _wlog(str(tmp_path / "w"), lambda w: 7.0, lambda w: 7.1)
    report_anchor(_aolog(str(tmp_path / "a"), lambda w: 2.1, lambda w: 1.4, seqs=7), wl, out)
    b = _aobands(out)
    assert b["G0b"]["reading"] == "FAIL"
    assert "NOT SCORED" in b["B1"]["reading"]


def _ao():
    import csv as _csv
    import os as _os
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    return {int(r["width"]): r for r in _csv.DictReader(
        open(_os.path.join(root, "results", "anchor_only_cost.csv"), encoding="utf-8"))}


def test_the_deployable_price_the_paper_prints_rounds_from_its_csv():
    """feat-164. Every serving number in this paper prices a harness that forwards both models on
    every path; the deployable one is C/A of it. These are the cells the abstract, Section 5 and
    Appendix I now print."""
    from tests.manuscript import body
    r = _ao()[200]
    txt = body("selection.tex", "appendix_selection.tex", "appendix_related.tex",
               "iclr_closing.tex")
    assert f"{float(r['C_over_A']):.2f}" == "0.35"
    assert "a draw costs $0.35$ of that" in txt, "the deployable draw factor is not printed"
    # the TABLE row, not just "the number appears somewhere": caution (an), and the mutation that
    # found it -- 4.463 occurs twice in the appendix, so retyping the table cell left the prose
    # copy satisfying a bare membership test.
    row = " & ".join(f"${float(r[c]):.3f}$s" for c in
                     ("A_harness_anchor_plus_risky_s", "B_harness_anchor_paired_s")) 
    assert row in txt, f"the anchor-only table lost its A/B cells: {row}"
    assert f"$\\mathbf{{{float(r['C_plain_anchor_alone_s']):.3f}}}$s & "
    assert (f"$\\mathbf{{{float(r['C_plain_anchor_alone_s']):.3f}}}$s & "
            f"${float(r['D_metered_s']):.3f}$s") in txt, "the C/D cells of the table moved"
    assert txt.count(f"{float(r['C_plain_anchor_alone_s']):.3f}") == 2, (
        "the anchor-alone cost is printed in the table and in the prose beneath it; if that "
        "changes, re-scope this guard rather than relaxing it")
    # 64 * C/D, the price a server that runs only the anchor pays at the headline n
    assert f"{64 * float(r['C_over_D']):.1f}" == "21.8"
    assert r"$21.8\times$" in txt, "the deployable price at n=64 is not printed"


def test_the_harness_forwards_both_models_and_the_paper_says_so():
    """The explanation feat-162 wrote into the appendix -- that neither path is weight-bound -- was
    wrong, and a withdrawal that deletes it leaves a reader of the earlier version uncorrected. The
    appendix must carry the real mechanism, which is a property of a_patch/factory.py that anyone
    can check."""
    from tests.manuscript import body
    txt = body("appendix_selection.tex")
    assert "forwards \\emph{both} models at every step" in txt, (
        "the appendix no longer states why the two paths measured equal")
    assert "discards it" in txt and "the same work" in txt
    assert "wrong thing to time" in txt, "the design's correctness for an audit is not stated"
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "a_patch", "factory.py"), encoding="utf-8").read()
    assert src.count("self.forward_direct(self.safe_model") >= 1
    assert src.count("self.forward_direct(self.risky_model") >= 1, (
        "the claim the appendix makes about the decoder is no longer true of the decoder")


def test_the_concession_survives_all_three_prices():
    """The compute-matched loss is the paper's most damaging sentence and three different cost
    models now put the matched cell in three different places. It is stated as strongly as it is
    only because the loss holds at all three, so the appendix must say which three."""
    from tests.manuscript import body
    txt = body("appendix_selection.tex")
    assert "survives\nall three prices".replace("\n", " ") in txt, "the three-price claim is gone"
    for token in ("$n=4$", "$n=1$", "$n=2$"):
        assert token in txt, f"the appendix no longer names the {token} cell"
