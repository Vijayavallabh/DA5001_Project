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
