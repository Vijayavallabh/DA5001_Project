"""feat-118: the judge-free scorer-scale arm, checked against its CSVs and its pre-registration.

This arm exists to test feat-117's saturation result on an axis with no judge on it, so the checks
that matter are that the recomputation gate held, that every committed band was scored, and that
whatever the paper says about the two axes matches what the CSVs say.
"""
import csv
import os

import pytest

PREREG = "results/onset_prediction_verifiable_scorer_scale.md"
BANDS = "results/verifiable_scorer_scale_bands.csv"
FRONTIER = "results/verifiable_scorer_scale.csv"
N_GRID = (1, 2, 4, 8, 16, 32, 64)
SCALES = (0.4940, 1.5437, 3.0859, 7.6156)

pytestmark = pytest.mark.skipif(not os.path.exists(BANDS),
                                reason="feat-118 has not been scored yet")


def _rows(path):
    return list(csv.DictReader(open(path)))


def _head():
    """Split on the NEWLINE-prefixed heading; line 3 quotes `## Scoring log` in backticks and a
    bare partition would cut there (caution (r))."""
    return open(PREREG, encoding="utf-8").read().partition("\n## Scoring log")[0]


def test_the_recomputation_gate_held():
    """H0: the accuracy this arm recomputes from the caches must equal what
    selection_verifiable.py committed, for every scorer and every n. If it does not, the two are
    reading the caches differently and the pre-registration says nothing may be quoted."""
    h0 = [b for b in _rows(BANDS) if b["band"].startswith("H0")]
    assert len(h0) == 1 and h0[0]["reading"] == "MATCHES", h0


def test_every_committed_band_was_scored():
    head, scored = _head(), {b["band"].split()[0] for b in _rows(BANDS)}
    for label in ("H1", "H2", "H3", "H4", "H5"):
        assert f"**{label} " in head, f"{label} is not committed in the pre-registration"
    for label in ("H0", "H1", "H2", "H3", "H5"):
        assert label in scored, f"{label} was committed and never scored"
    assert all(b["reading"] for b in _rows(BANDS)), "a band has no reading"


def test_both_tasks_ran_and_triviaqa_was_committed_not_added():
    """H4's whole point is that TriviaQA is named in advance so it cannot become a post-hoc
    rescue, and that it does not change the H3 reading whatever it says."""
    head = _head()
    assert "committed here and not added later" in head
    assert "does not change the H3 reading" in head
    tasks = {r["task"] for r in _rows(FRONTIER)}
    assert tasks == {"gsm8k", "triviaqa"}, tasks


def test_the_grid_is_complete_at_every_scale_on_both_tasks():
    have = {(r["task"], r["scorer"], int(r["n"])) for r in _rows(FRONTIER)}
    for task in ("gsm8k", "triviaqa"):
        for s in ("0.5B", "1.5B", "3B", "7.6B", "majority vote"):
            for n in N_GRID:
                assert (task, s, n) in have, (task, s, n)


def test_the_scorer_sizes_are_measured_counts_and_not_labels():
    got = sorted({float(r["params_b"]) for r in _rows(FRONTIER) if r["params_b"]})
    assert len(got) == 4, got
    for g, want in zip(got, SCALES):
        assert abs(g - want) < 5e-4, (g, want)


def test_majority_vote_is_identical_across_scorer_runs():
    """Majority vote never touches a reward model, so every per-scorer run must reproduce it
    exactly. It is the free consistency check on the whole cached-generation path, and a
    disagreement would mean the runs are not reading the same candidates."""
    for tag in ("_comma7b", "_tqa_comma7b"):
        seen = []
        for sfx in ("", "_qwen05b", "_qwen15b", "_qwen3b"):
            p = f"results/selection_verifiable{tag}{sfx}.csv"
            seen.append(tuple((r["n"], r["acc"]) for r in _rows(p)
                              if r["arm"].startswith("majority")))
        assert len(set(seen)) == 1, f"{tag}: majority vote differs across scorer runs"


def test_the_h3_reading_follows_from_the_h2_readings():
    """H3 is mechanical: AGREES only when the separating adjacent steps are exactly the one
    feat-117 found. Recomputing it here stops the headline drifting from its inputs."""
    bands = _rows(BANDS)
    for task in ("gsm8k", "triviaqa"):
        h2 = {b["quantity"].split(" at ")[0]: b["reading"]
              for b in bands if b["band"].startswith("H2") and f"({task})" in b["band"]}
        h1 = [b for b in bands if b["band"].startswith("H1") and f"({task})" in b["band"]][0]
        h3 = [b for b in bands if b["band"].startswith("H3") and f"({task})" in b["band"]][0]
        got = sorted(k for k, v in h2.items() if v == "SEPARATES")
        if got == ["1.5B - 0.5B"]:
            assert h3["reading"] == "AGREES", (task, got, h3["reading"])
        elif not got and h1["reading"] == "FLAT":
            assert h3["reading"] == "DISAGREES, NO SCORER EFFECT", (task, h3["reading"])
        elif any(k in got for k in ("3B - 1.5B", "7.6B - 3B")):
            assert h3["reading"] == "DISAGREES, LATER SATURATION", (task, got, h3["reading"])
        else:
            assert h3["reading"] == "DISAGREES, OTHER", (task, got, h3["reading"])


def test_shape_is_not_read_as_the_test():
    """The pre-registration rules shape out in advance, because the 7.6B reward is already
    non-monotone on this axis. H5 must therefore carry the 7.6B reference wherever it reads
    NOT MONOTONE, so no one later mistakes a wobble for a finding."""
    assert "Shape is not the test here" in _head() or "shape is not the test" in _head().lower()
    for b in _rows(BANDS):
        if b["band"].startswith("H5") and b["reading"].startswith("NOT MONOTONE"):
            assert "0.9286" in b["reading"], b
