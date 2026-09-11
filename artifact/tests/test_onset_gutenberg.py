"""The second-corpus onset is a pre-registered test, so the parts that could quietly flatter it are
the ones to pin: the bands, the entry gate, and the permutation p-value's floor at n = 3."""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.onset_gutenberg import BAND_TIGHT, BAND_WIDE, ENTRY_GATE, exact_p, spearman  # noqa: E402
from analysis.onset import crossing  # noqa: E402


def test_the_committed_constants_are_the_ones_in_the_preregistration():
    """results/onset_prediction_gutenberg.md commits [0.85, 1.15], [0.7, 1.4] and a 0.10 gate.
    A band edited after the fact is the one failure mode a pre-registration cannot survive."""
    assert (BAND_TIGHT, BAND_WIDE, ENTRY_GATE) == ((0.85, 1.15), (0.7, 1.4), 0.10)
    path = "results/onset_prediction_gutenberg.md"
    if os.path.exists(path):
        md = open(path).read()
        for s in ("[0.85, 1.15]", "[0.7, 1.4]", "0.10"):
            assert s in md, s


def test_exact_p_cannot_go_below_one_third_at_three_points():
    """With n = 3 there are 6 orderings and Spearman takes 4 distinct values, so a perfect
    correlation still returns 1/3. Reporting a small p here would be an artefact, not a finding."""
    best = min(exact_p([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
               exact_p([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]))
    assert best >= 1 / 3 - 1e-12
    assert spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0
    assert spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0


def test_exact_p_matches_brute_force_on_four_points():
    a, b = [1.0, 4.0, 2.0, 3.0], [2.0, 3.0, 1.0, 4.0]
    obs = abs(spearman(a, b))
    want = sum(1 for q in itertools.permutations(b)
               if abs(spearman(a, list(q))) >= obs - 1e-12) / 24
    assert abs(exact_p(a, b) - want) < 1e-12


def test_crossing_interpolates_inside_its_bracket_and_never_outside_the_grid():
    c = {2.3: 0.001, 2.5: 0.009, 2.7: 0.017}
    lo, hi, est = crossing(c, 0.01)
    assert (lo, hi) == (2.5, 2.7) and 2.5 < est < 2.7
    assert crossing({1.0: 0.0, 2.0: 0.0}, 0.01) == (None, None, None)   # never crosses


def test_committed_table_is_consistent_with_its_own_bands():
    import csv
    path = "results/onset_gutenberg.csv"
    if not os.path.exists(path):
        return
    for r in csv.DictReader(open(path)):
        if not r["onset"]:
            continue
        assert abs(float(r["ratio"]) - float(r["onset"]) / float(r["s_safe"])) < 1e-3, r
        assert abs(float(r["pred_over_meas"]) - float(r["pred_onset"]) / float(r["onset"])) < 1e-3, r
        # the entry gate is applied, not merely recorded
        assert (r["entered"] == "True") == (float(r["k_minus1_recall"]) >= ENTRY_GATE), r
        # the onset must sit inside the bracket the grid gave it
        assert float(r["onset_lo"]) < float(r["onset"]) <= float(r["onset_hi"]), r


def test_the_grid_rule_is_the_one_committed_for_feat_084():
    """The grid for a new pair is committed in units of k/s(x) before s(x) is measured, because
    choosing it after seeing the crossing is the failure the pre-registrations exist to prevent."""
    from analysis.grid_from_sx import RULE, grid
    assert RULE == (0.55, 0.65, 0.75, 0.85, 0.90, 0.95, 1.00, 1.05, 1.15, 1.30, 1.55)
    md = "results/onset_prediction_granularity_gap.md"
    if os.path.exists(md):
        text = open(md).read()
        for f in RULE:
            assert f"{f:.2f}" in text or f"{f:g}" in text, f
    g = grid(2.0)
    assert g[:2] == [-1.0, 0.0], "both mandatory baselines, in order"
    assert g[2:] == [1.1, 1.3, 1.5, 1.7, 1.8, 1.9, 2.0, 2.1, 2.3, 2.6, 3.1]
    assert len(set(grid(3.3))) == len(grid(3.3)), "no duplicate budgets after rounding"
