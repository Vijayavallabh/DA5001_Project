"""--max-n must be able to form the arm it names.

Until 2026-09-17 `selection_scaling.py` computed its arms as `[n for n in GRID if n <= max_n]`
against a hardcoded GRID that stops at 64. `--max-n 256` therefore scored 256 candidates per
prompt -- four times the compute -- and then formed no arm above 64 at all: the flag could not
produce what a pre-registration naming n=256 registered. Same class as caution (w).

The two halves that matter are guarded here: nothing at or below the committed 64 may move, and
the grid must actually reach max_n above it.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_scaling import GRID, n_grid  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def test_the_committed_grid_is_unchanged():
    """Every arm on record was formed at --max-n 64; that call must still return the same list."""
    assert n_grid(64) == list(GRID) == [1, 2, 4, 8, 16, 32, 64]


def test_a_larger_max_n_actually_forms_an_arm_there():
    """The bug: the grid stopped at 64 however large --max-n was."""
    for max_n in (128, 256, 512):
        grid = n_grid(max_n)
        assert grid[-1] == max_n, (max_n, grid)
        assert grid[:7] == list(GRID), (max_n, grid)
        assert grid == sorted(set(grid)), grid


def test_the_grid_never_exceeds_max_n():
    """A pool of max_n draws cannot serve an argmax over more than max_n of them."""
    for max_n in (1, 3, 8, 64, 100, 128, 200, 256):
        assert max(n_grid(max_n)) <= max_n, max_n


def test_the_scored_csv_on_record_holds_exactly_the_max_n_64_arms():
    """The guard against a future widening of GRID silently re-labelling the committed arms."""
    path = os.path.join(RESULTS, "selection_scaling.csv")
    assert os.path.exists(path), path
    ns = sorted({int(r["n"]) for r in csv.DictReader(open(path, encoding="utf-8")) if r.get("n")})
    assert ns == n_grid(64), (ns, n_grid(64))
