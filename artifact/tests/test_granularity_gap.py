"""The granularity-gap pair and its contingent control were pre-registered with numeric bands, so
what a test can protect is that the bands in the scoring log are the ones in the manuscript and that
the two open-calm pairs really are the matched contrast they are described as."""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.seed_effect import spearman  # noqa: E402

PREREG = "results/onset_prediction_granularity_gap.md"
BAND = (0.927, 1.052)          # the committed interpolation band
COARSE, FINE = 0.926, 1.053    # the two refuting bands


def rows():
    with open("results/onset_table.csv") as fh:
        return {r["pair"]: r for r in csv.DictReader(fh) if not r["pair"].startswith("ALL")}


def test_the_committed_bands_are_still_the_ones_in_the_preregistration():
    md = open(PREREG).read()
    for s in ("$0.927$ and $1.052$", "$\\le 0.926$", "$\\ge 1.053$", "0.10"):
        assert s in md, s


def test_both_open_calm_pairs_land_inside_the_interpolation_band():
    """The 1B was the point in the gap; the 3B was committed as the control for its weak memoriser.
    If either had fallen into a refuting band the paper's gradient claim would have to change."""
    r = rows()
    for name in ("open-calm-1b + mem. open-calm-1b", "open-calm-3b + mem. open-calm-3b"):
        ratio = float(r[name]["ratio"])
        assert BAND[0] < ratio < BAND[1], (name, ratio)
        assert ratio > COARSE and ratio < FINE, (name, ratio)


def test_the_control_holds_everything_but_the_memoriser_fixed():
    """Same tokenizer, so the same seed in words; s(x) within half a percent. What differs is the
    memoriser: s_r/s_s must differ by more than s(x) does, or the pair controls nothing."""
    r = rows()
    a, b = r["open-calm-1b + mem. open-calm-1b"], r["open-calm-3b + mem. open-calm-3b"]
    sa, sb = float(a["s_safe"]), float(b["s_safe"])
    assert abs(sa - sb) / sa < 0.005
    with open("results/onset_seed_words.csv") as fh:
        words = {x["pair"]: float(x["seed_words"]) for x in csv.DictReader(fh)}
    assert words["open-calm-1b + mem. open-calm-1b"] == words["open-calm-3b + mem. open-calm-3b"]
    assert abs(float(a["s_risky"]) / sa - float(b["s_risky"]) / sb) > 0.02


def test_the_seed_words_correlation_the_paper_quotes_comes_out_of_the_csv():
    with open("results/onset_seed_words.csv") as fh:
        rs = list(csv.DictReader(fh))
    w = [float(x["seed_words"]) for x in rs]
    y = [float(x["ratio"]) for x in rs]
    assert len(rs) == 9
    # 2 dp in the CSV, not 1: at 1 dp Pleias-350M and Phi-3.5-mini tie and this reads -0.971.
    assert round(spearman(w, y), 3) == -0.958


def test_the_seed_words_csv_covers_every_pair_in_the_table():
    """figures/make_figures_v4.py:seed_effect plots the observational pairs from this CSV. It used
    to carry a hardcoded dict of seven word counts, which went stale in two ways at once: the counts
    were the 13.0-14.4 the manuscript corrected to 13.86-15.02, and there were nine pairs by then.
    Anything that indexes pairs by name has to be checked against the table, not maintained by hand."""
    table = rows()
    with open("results/onset_seed_words.csv") as fh:
        words = {r["pair"]: float(r["ratio"]) for r in csv.DictReader(fh)}
    assert set(words) == set(table)
    for name, ratio in words.items():
        assert abs(ratio - float(table[name]["ratio"])) < 1e-9, name
