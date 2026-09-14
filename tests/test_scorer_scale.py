"""feat-117: the scorer-scale boundary, checked against its CSVs and its pre-registration.

The arm exists to give an interval to something feat-116 only read off a table, so the checks that
matter here are that every committed band was scored, that the replication gate is what gates the
rest, and that the paper's claim about the boundary matches the CSV it is read from.
"""
import csv
import math
import os

import pytest

PREREG = "results/onset_prediction_scorer_scale.md"
BANDS = "results/scorer_scale_bands.csv"
FRONTIER = "results/scorer_scale.csv"
GRID = (2, 4, 8, 16, 32, 64)
SCALES = (0.4940, 1.5437, 3.0859, 7.6156)

pytestmark = pytest.mark.skipif(not os.path.exists(BANDS),
                                reason="feat-117 has not been scored yet")


def _rows(path):
    return list(csv.DictReader(open(path)))


def _head():
    """Split on the NEWLINE-prefixed heading: line 3 of every pre-registration quotes
    `## Scoring log` in backticks, so a bare partition cuts at line 3 (caution (r))."""
    return open(PREREG, encoding="utf-8").read().partition("\n## Scoring log")[0]


def test_every_committed_band_was_scored():
    head, bands = _head(), _rows(BANDS)
    scored = {b["band"].split()[0] for b in bands}
    for label in ("G0", "G1", "G2", "G3", "G4"):
        assert f"**{label} " in head, f"{label} is not committed in the pre-registration"
        assert label in scored, f"{label} was committed and never scored"
    assert all(b["reading"] for b in bands), "a band has no reading"


def test_the_replication_gate_passed_before_anything_else_is_quoted():
    """G0 pins this pass's sel7b_n64, sel05b_n64 and metered_k10 to feat-116 within the judge's own
    cross-pass floor. The pre-registration says no band below is quoted if it fails, so a FAILED G0
    here means the paper must not carry this arm at all."""
    assert "no band below is quoted" in _head()
    g0 = [b for b in _rows(BANDS) if b["band"].startswith("G0")]
    assert len(g0) == 1 and g0[0]["reading"] == "REPLICATES", g0


def test_the_grid_is_the_committed_one_at_every_scale():
    arms = {r["arm"] for r in _rows(FRONTIER)}
    for tag in ("05b", "15b", "3b", "7b"):
        for n in GRID:
            assert f"sel{tag}_n{n}" in arms, f"sel{tag}_n{n} is missing"
    assert "metered_k10" in arms
    ns = {r["n_prompts"] for r in _rows(FRONTIER)}
    assert len(ns) == 1, f"the arms are not on one prompt set: {ns}"


def test_the_scorer_sizes_are_measured_counts_and_not_labels():
    got = sorted({float(r["scorer_b"]) for r in _rows(FRONTIER) if r["scorer_b"]})
    assert len(got) == 4, got
    for g, want in zip(got, SCALES):
        assert abs(g - want) < 5e-4, (g, want)


def test_the_certified_nats_column_is_log_n():
    for r in _rows(FRONTIER):
        if r["n"]:
            assert abs(float(r["nats_certified"]) - math.log(int(r["n"]))) < 5e-4, r["arm"]


def test_the_terminal_drop_is_measured_from_the_cell_feat_116_fixed():
    """n=16 must come from feat-116 and not from this arm's own peak, or G1 is a cell chosen after
    looking. The pre-registration says so and the band label carries the arms it differenced."""
    head = _head()
    assert "g_s(64) - g_s(16)" in head
    assert "not a cell picked after looking" in head or "out-of-sample" in head
    for b in _rows(BANDS):
        if b["band"].startswith("G1"):
            assert b["quantity"].endswith("_n64 - sel" + b["quantity"].split("_n64")[0][3:] + "_n16"), \
                b["quantity"]


def test_the_boundary_reading_follows_from_the_g1_readings():
    """G3 is mechanical: the smallest scorer that does not read TURNS OVER. Recomputing it here
    stops the headline drifting from the per-scorer readings it is derived from."""
    bands = _rows(BANDS)
    g1 = {b["band"].split(",")[1].strip().replace(" scorer", ""): b["reading"]
          for b in bands if b["band"].startswith("G1")}
    g3 = [b for b in bands if b["band"].startswith("G3")][0]["reading"]
    over = [k for k, v in g1.items() if v == "TURNS OVER"]
    safe = sorted((float(k.rstrip("B")) for k, v in g1.items() if v != "TURNS OVER"))
    if not over:
        assert g3 == "NO TURNOVER FOUND", (g3, g1)
    elif safe:
        assert g3 == f"BOUNDARY AT {safe[0]}B", (g3, g1)
    else:
        assert g3 == "BOUNDARY ABOVE 7.6B", (g3, g1)


def test_the_agreement_diagnostic_is_labelled_descriptive_and_has_no_band():
    """scorer_scale_agreement.csv explains the boundary without a judge. No band was committed on
    it, so neither the appendix nor this test may treat it as evidence for the boundary itself."""
    rows = _rows("results/scorer_scale_agreement.csv")
    assert {r["reference"] for r in rows} == {"7B"}
    assert "descriptive and neither is a band" in open("analysis/scorer_agreement.py").read()
    for r in rows:
        assert float(r["same_draw_as_reference"]) >= float(r["chance"]), r
