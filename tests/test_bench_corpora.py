"""The scale-up corpora, and the two invariants that make them usable.

(1) Nothing under data/*.jsonl is rewritten: each new corpus is a directory of symlinks to the
    committed files with one real file swapped in, so a run on it is the same code path.
(2) attack_train / val / test are disjoint IN BOOK. Caution (h) cost a whole feature the last time
    that failed: a memoriser is fine-tuned on attack_train + val, so a probe on test scores a novel
    the model has never seen."""
import json
import os

import pytest

DIRS = ("data/bench/alpaca", "data/bench/bookmia100", "data/bench/bookmia100unseen")
pytestmark = pytest.mark.skipif(not os.path.isdir("data/bench/bookmia100"),
                                reason="run analysis/build_bench_corpora.py first (needs HF_TOKEN)")


def _rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def test_the_committed_prompt_sets_are_reached_by_symlink_and_never_copied():
    for d in DIRS:
        for f in os.listdir(d):
            p = os.path.join(d, f)
            assert os.path.islink(p), f"{p} is a real file; the committed sets must be symlinked"
        real = {f for f in os.listdir(d)
                if os.path.realpath(os.path.join(d, f)).startswith(os.path.abspath("data/bench"))}
        assert real, f"{d} replaces nothing, so it is just data/ again"


def test_the_bookmia_splits_are_disjoint_in_book():
    for tag in ("bookmia100", "bookmia100unseen"):
        books = {}
        for split in ("attack_train", "val", "test"):
            books[split] = {r["source_novel"] for r in _rows(f"data/bench/{tag}_{split}.jsonl")}
        a, v, t = books["attack_train"], books["val"], books["test"]
        assert not (a & t), (tag, sorted(a & t)[:3])
        assert not (v & t), (tag, sorted(v & t)[:3])
        assert not (a & v), (tag, sorted(a & v)[:3])
        assert len(a | v | t) == 50, (tag, len(a | v | t))


def test_the_seen_and_unseen_corpora_share_no_book():
    """The unseen set is a negative control only if it is genuinely other books."""
    seen = {r["source_novel"] for s in ("attack_train", "val", "test")
            for r in _rows(f"data/bench/bookmia100_{s}.jsonl")}
    unseen = {r["source_novel"] for s in ("attack_train", "val", "test")
              for r in _rows(f"data/bench/bookmia100unseen_{s}.jsonl")}
    assert not (seen & unseen), sorted(seen & unseen)[:3]


def test_the_passages_have_the_same_shape_as_the_committed_ones():
    """A longer prefix hands the adversary more context and would confound recall across corpora."""
    import statistics as st
    ours = _rows("data/copybench_attack_train.jsonl")
    new = _rows("data/bench/bookmia100_attack_train.jsonl")
    mo = st.median(len(r["raw_text"]) for r in ours)
    mn = st.median(len(r["raw_text"]) for r in new)
    assert abs(mn - mo) < 60, (mn, mo)          # committed median 886, cut on a word boundary
    assert all(len(r["reference_text"]) > 100 for r in new[:200])
    assert all(not r["raw_text"].endswith(" ") for r in new[:200])


def test_alpaca_is_the_full_benchmark_and_carries_no_protected_target():
    rows = _rows("data/bench/alpaca_neutral.jsonl")
    assert len(rows) == 805, len(rows)
    assert all(r["reference_text"] == "" for r in rows)
    assert len({r["source_novel"] for r in rows}) == 5      # the five AlpacaEval sub-sets
