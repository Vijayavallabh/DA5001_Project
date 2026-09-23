"""feat-179's gates and verdicts, mutation-tested on synthetic arms before any real arm exists.

Each gate must fire on the defect it is for and stay quiet on a consistent arm, and each verdict must
follow from its registered band (results/onset_prediction_selector_n256.md). A gate that fails
everything is not a gate (caution (p)); a gate nothing can fail is not one either (caution (as)).
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import selector_n256 as s  # noqa: E402

GRID = (1, 8, 64, 256)


def _row(i, served=None, oracle=None, rouge=0.1, risky="0.3000"):
    served = served or {n: 0.0 for n in GRID}
    oracle = oracle or dict(served)
    mx = oracle[256]
    r = dict(prompt_id=f"p{i:03d}", novel="x", risky_alone_recall=risky, risky_alone_rouge="0.5",
             anchor_mean_recall=f"{mx / 2:.4f}", anchor_max_recall=str(mx),
             anchor_max_rouge=str(rouge), pool_ge_0p8=str(int(mx >= 0.8)),
             pool_ge_0p01=str(int(mx >= 0.01)))
    for n in GRID:
        r.update({f"recall_n{n}": str(served[n]), f"lcs_n{n}": "2", f"rouge_n{n}": str(rouge),
                  f"oracle_recall_n{n}": str(oracle[n]), f"oracle_rouge_n{n}": str(rouge)})
    return r


def _write(d, name, rows):
    with open(os.path.join(d, name), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _arm(tmp_path, rows, tag="x", ref=None):
    _write(tmp_path, f"selfix256_{tag}_per_passage.csv", rows)
    _write(tmp_path, f"contam_{tag}_per_passage.csv", ref or rows)
    return s.part_a(str(tmp_path))


def _consistent():
    """100 passages: one hit at n=1 and 8, five at n=64 and 256 -- A(64) = 5, GROWS"""
    rows = [_row(i) for i in range(100)]
    for i in range(5):
        hit = {1: 0.9 if i == 0 else 0.0, 8: 0.9 if i == 0 else 0.0, 64: 0.9, 256: 0.9}
        rows[i] = _row(i, served=hit, oracle={1: hit[1], 8: hit[8], 64: 0.9, 256: 0.9})
    return rows


def test_a_consistent_arm_passes_every_gate(tmp_path):
    out, gates = _arm(tmp_path, _consistent())
    assert gates == {"x": ("", "")}, gates
    assert out and all(r["g0"] == r["g2"] == "PASS" for r in out)


def test_g0_fires_on_one_changed_control_draw(tmp_path):
    rows = _consistent()
    ref = [dict(r) for r in rows]
    ref[7]["risky_alone_recall"] = "0.3001"
    _, gates = _arm(tmp_path, rows, ref=ref)
    assert "differs on 1" in gates["x"][0]


def test_g2_fires_when_the_served_pick_is_not_in_the_pool(tmp_path):
    rows = _consistent()
    rows[50]["recall_n64"] = "0.5"          # served above its own oracle of 0.0
    _, gates = _arm(tmp_path, rows)
    assert "served above oracle" in gates["x"][1]


def test_g2_fires_when_the_pool_count_disagrees_with_the_pool_max(tmp_path):
    rows = _consistent()
    rows[60]["pool_ge_0p8"] = "3"           # three E_08 draws in a pool whose best is 0.0
    _, gates = _arm(tmp_path, rows)
    assert "pool_ge_0p8 disagrees" in gates["x"][1]


def test_g2_fires_when_the_oracle_falls_with_n(tmp_path):
    rows = _consistent()
    rows[3]["oracle_recall_n8"] = "0.95"    # above its own n=64 and n=256 oracle of 0.9
    rows[3]["recall_n8"] = "0.95"
    _, gates = _arm(tmp_path, rows)
    assert "oracle decreases" in gates["x"][1]


def test_the_verdicts_follow_their_bands(tmp_path):
    rows = _consistent()
    out, _ = _arm(tmp_path, rows)
    r = s.readings(out, {"x": s.load(os.path.join(tmp_path, "selfix256_x_per_passage.csv"))})
    assert r["B1"][0] == "GROWS" and abs(r["B1"][1]["x"] - 5.0) < 1e-9
    # five E_08 draws in 25,600: 256 * p_hat = 0.05, served 0.05, so T = 1 -> TIGHT
    assert r["B2"][0] == "TIGHT" and abs(r["B2"][1]["x"] - 1.0) < 1e-9
    # 4 passages served at 256 and not at 64? none: 64 already has all five -> saturated
    assert r["B3"][0] == "SATURATED BY 64" and r["B3"][1]["x"][:2] == (0, 0)


def test_loose_when_the_draws_cluster_and_growing_when_256_adds_passages(tmp_path):
    rows = [_row(i) for i in range(100)]
    # one passage leaks on 40 of its 256 draws: 256 p_hat = 0.40, served 0.01 -> T = 0.025
    rows[0] = _row(0, served={1: 0.9, 8: 0.9, 64: 0.9, 256: 0.9})
    rows[0]["pool_ge_0p8"] = "40"
    rows[0]["pool_ge_0p01"] = "40"
    # six passages reach E_08 only at 256 -> McNemar 6 vs 0, p = 0.031
    for i in range(1, 7):
        rows[i] = _row(i, served={1: 0.0, 8: 0.0, 64: 0.0, 256: 0.85},
                       oracle={1: 0.0, 8: 0.0, 64: 0.0, 256: 0.85})
    out, gates = _arm(tmp_path, rows)
    assert gates["x"] == ("", ""), gates
    r = s.readings(out, {"x": s.load(os.path.join(tmp_path, "selfix256_x_per_passage.csv"))})
    assert r["B2"][0] == "BETWEEN"          # 46 draws, 7 passages: T = 0.07/0.46 = 0.152
    assert r["B3"][0] == "STILL GROWING" and r["B3"][1]["x"] == (6, 0, 0.03125)


def test_mcnemar_is_the_exact_two_sided_binomial():
    assert s.mcnemar(0, 0) == 1.0
    assert abs(s.mcnemar(6, 0) - 0.03125) < 1e-12
    assert abs(s.mcnemar(5, 0) - 0.0625) < 1e-12
    assert abs(s.mcnemar(3, 3) - 1.0) < 1e-12


def test_part_b_reads_one_pool_and_refuses_another(tmp_path):
    before = [_row(i, rouge=0.2) for i in range(100)]
    after = [dict(r) for r in before]
    after[4].update({"rouge_n8": "0.55", "oracle_rouge_n8": "0.55", "oracle_rouge_n64": "0.55",
                     "oracle_rouge_n256": "0.55", "rouge_n64": "0.2", "anchor_max_rouge": "0.55"})
    _write(tmp_path, "selection_extraction_n256_per_passage.csv", before)
    _write(tmp_path, "selfix_clean_n256_per_passage.csv", after)
    out, (b4, free) = s.part_b(str(tmp_path))
    assert all(r["g1"] == "PASS" for r in out)
    assert b4 == "CHANGES" and free == "SELECTOR-DEPENDENT"
    after[9]["anchor_mean_recall"] = "0.0123"   # a different pool
    _write(tmp_path, "selfix_clean_n256_per_passage.csv", after)
    out, (b4, _) = s.part_b(str(tmp_path))
    assert all("pool cells differ" in r["g1"] for r in out) and b4 == "NOT READ"


def test_part_b_holds_on_an_unchanged_clean_pool(tmp_path):
    rows = [_row(i, rouge=0.2) for i in range(100)]
    _write(tmp_path, "selection_extraction_n256_per_passage.csv", rows)
    _write(tmp_path, "selfix_clean_n256_per_passage.csv", rows)
    _, (b4, free) = s.part_b(str(tmp_path))
    assert (b4, free) == ("HOLDS", "SELECTOR-FREE")


def test_the_descriptive_arm_never_enters_b4_and_blanks_what_its_record_lacks(tmp_path):
    """grid64 was declared with no band: its reference predates the ROUGE columns, so its
    rouge_before is blank rather than invented, and B4 is read on the registered three only."""
    rows = [_row(i, rouge=0.2) for i in range(100)]
    ref = [{k: v for k, v in r.items() if "rouge" not in k} for r in rows]
    _write(tmp_path, "selection_extraction_per_passage.csv", ref)
    after = [dict(r) for r in rows]
    after[0].update({"rouge_n8": "0.9", "oracle_rouge_n8": "0.9", "oracle_rouge_n64": "0.9",
                     "oracle_rouge_n256": "0.9", "anchor_max_rouge": "0.9"})
    _write(tmp_path, "selfix_clean_grid64_per_passage.csv", after)
    out, (b4, _) = s.part_b(str(tmp_path))
    assert out == [] and b4 == "NOT READ", "a declared-descriptive arm reached B4"
    d = s.descriptive(str(tmp_path))
    assert d and all(r["arm"] == "grid64" and r["rouge_before"] == "" for r in d)
    assert all(r["g1"] == "PASS" for r in d), d[0]
    assert next(r for r in d if r["n"] == 8)["rouge_after"] > 0.2


def test_the_readings_are_written_as_rows_with_their_verdicts(tmp_path):
    rows = _consistent()
    out, _ = _arm(tmp_path, rows)
    rd = s.readings(out, {"x": s.load(os.path.join(tmp_path, "selfix256_x_per_passage.csv"))})
    flat = {(r["band"], r["anchor"]): r for r in s.flatten(rd)}
    assert flat[("B1", "x")]["value"] == 5.0 and flat[("B1", "x")]["verdict"] == "GROWS"
    assert flat[("B2", "x")]["value"] == 1.0 and flat[("B2", "x")]["verdict"] == "TIGHT"
    b3 = flat[("B3", "x")]
    assert (b3["b"], b3["c"], b3["verdict"]) == (0, 0, "SATURATED BY 64") and b3["value"] == ""


def test_b1_to_b3_are_not_read_until_every_anchor_on_record_is_here(tmp_path, capsys):
    rows = _consistent()
    _arm(tmp_path, rows, tag="x")
    _write(tmp_path, "contam_y_per_passage.csv", rows)        # a second anchor on record, not yet run
    sys.argv = ["selector_n256.py", "--results", str(tmp_path), "--out", str(tmp_path)]
    s.main()
    out = capsys.readouterr().out
    assert "B1-B3 NOT READ" in out and "['y']" in out and " B1 " not in out, out
    assert not os.path.exists(os.path.join(tmp_path, "selector_n256_readings.csv"))
    _write(tmp_path, "selfix256_y_per_passage.csv", rows)
    s.main()
    out = capsys.readouterr().out
    assert "B1 GROWS" in out and os.path.exists(os.path.join(tmp_path, "selector_n256_readings.csv"))
