"""`--limit N` must not mean "the first N" (caution (w)).

500 rows of the BookMIA unseen half gave six books where the file holds 27, because these corpus
files are ordered by book. The round robin is what fixes it, and this pins BOTH halves: that the
spread is even, and that `--select first` still reproduces the prefix feat-176's Gutenberg arm ran
on. It needs no corpus on disk, which is the point -- tests/test_bench_corpora.py skips wholesale
when the built corpora are absent, and a skipped branch is a pass.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.build_gutenberg_bench import select  # noqa: E402

# Lopsided on purpose: one book supplies more than the whole quota on its own, which is exactly
# the shape that makes a prefix look like a corpus.
ROWS = ([dict(prompt_id=f"a{i}", source_novel="a") for i in range(40)]
        + [dict(prompt_id=f"b{i}", source_novel="b") for i in range(6)]
        + [dict(prompt_id=f"c{i}", source_novel="c") for i in range(6)])


def test_the_prefix_behaviour_is_the_defect_and_is_still_reproducible():
    got = select(ROWS, 10, "first")
    assert [r["prompt_id"] for r in got] == [f"a{i}" for i in range(10)]
    assert len({r["source_novel"] for r in got}) == 1, \
        "the prefix took more than one book; this input no longer exhibits the defect"


def test_the_round_robin_reaches_every_book_and_returns_the_quota():
    got = select(ROWS, 10, "roundrobin")
    assert len(got) == 10
    assert {r["source_novel"] for r in got} == {"a", "b", "c"}
    counts = [sum(1 for r in got if r["source_novel"] == b) for b in ("a", "b", "c")]
    assert max(counts) - min(counts) <= 1, f"uneven over books: {counts}"


def test_a_book_running_out_does_not_stop_the_others():
    """b and c hold 6 each; asking for 20 must keep drawing from a rather than stopping at 18."""
    got = select(ROWS, 20, "roundrobin")
    assert len(got) == 20
    assert sum(1 for r in got if r["source_novel"] == "a") == 8
    assert len({r["prompt_id"] for r in got}) == 20, "a row was served twice"


def test_asking_for_more_than_exists_returns_everything_once():
    got = select(ROWS, 999, "roundrobin")
    assert len(got) == len(ROWS)
    assert len({r["prompt_id"] for r in got}) == len(ROWS)
