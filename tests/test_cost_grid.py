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
    report(_log(str(tmp_path)), out)
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
    report(_log(str(tmp_path)), out)
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
    report(_log(str(tmp_path), rew7=10.0), out)
    b = _bands(out)
    assert b["G1"]["reading"] == "FAIL"
    assert "NOT SCORED" in b["B1"]["reading"]
    assert "B2" not in b, "a band was computed under a failed gate"


def test_g2_catches_two_cells_that_did_not_serve_the_same_work(tmp_path):
    out = str(tmp_path / "out")
    report(_log(str(tmp_path), met_tokens=120), out)
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
    assert f"${PROXY_N4}\\times$ the cost" in txt, (
        "selection.tex no longer prints the ratio cost_grid.py measures against")


def test_g0_fails_when_the_two_paths_are_implausibly_close(tmp_path):
    """The instrument gate is what catches a cell that is not the arm it claims to be -- a
    selection path measuring about the meter's own cost means the draws never happened. It has to
    fail loudly there, and take every band below it down (caution (as))."""
    out = str(tmp_path / "out")
    report(_log(str(tmp_path), b=0.5, rew7=0.01), out)          # ratio(64) ~ 2.7x, far below 15
    b = _bands(out)
    assert b["G0"]["reading"] == "FAIL", b["G0"]
    assert "NOT SCORED" in b["B1"]["reading"] and "B2" not in b
