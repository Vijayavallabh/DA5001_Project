"""The decode-time MemFree rule: does it actually block, and does it block the right thing?

A blocklist arm whose rule never fires would make "the blocklist stops literal copying" an
unfalsifiable claim (caution (p): a gate that fires on nothing is not a gate). These check the
predicate itself, since the generation loop around it needs a GPU.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.blocklist import build_index, ngrams, toks  # noqa: E402
from analysis.blocklist_decode import blocked  # noqa: E402

PASSAGE = ("it was a bright cold day in april and the clocks were striking thirteen "
           "winston smith his chin nuzzled into his breast")


def _idx(n):
    return ngrams(toks(PASSAGE), n)


def test_the_rule_fires_on_a_listed_ngram_and_not_before_it_is_complete():
    n = 10
    idx = _idx(n)
    ws = toks(PASSAGE)
    # the first n words ARE a listed n-gram, so a text ending on them is blocked
    assert blocked(" ".join(ws[:n]), idx, n)
    # one word short of n: nothing to match yet
    assert not blocked(" ".join(ws[: n - 1]), idx, n)


def test_a_sub_word_piece_cannot_trip_the_rule():
    """The predicate is over WORDS. A token that does not finish a word has produced none, so it
    must not block -- otherwise the rule would fire on a prefix of an innocent word."""
    n = 10
    idx = _idx(n)
    ws = toks(PASSAGE)
    ending = " ".join(ws[:n])
    assert blocked(ending, idx, n)
    # the same text with the last word left half-typed is a DIFFERENT last word, so not listed
    assert not blocked(ending[:-2], idx, n)


def test_unlisted_text_is_never_blocked_which_is_the_whole_limitation():
    n = 10
    idx = _idx(n)
    assert not blocked("the quick brown fox jumps over the lazy dog and then some more words",
                       idx, n)


def test_a_paraphrase_walks_through_a_ten_gram_block():
    """The registered prediction in results/onset_prediction_memfree_headtohead.md is that the rule
    leaks the non-literal event. The mechanism is this: change one word in ten and no listed
    n-gram is ever completed, while the text stays a near copy."""
    n = 10
    idx = _idx(n)
    ws = toks(PASSAGE)
    near = ws[:]
    for i in range(0, len(near), n):          # one substitution per n-gram window
        near[i] = "quite"
    text = " ".join(near)
    assert not any(blocked(" ".join(near[: j + 1]), idx, n) for j in range(len(near))), \
        "a one-in-ten substitution still trips the block; the leak prediction needs rechecking"
    # and it really is still a near copy: most words are shared
    shared = sum(1 for x, y in zip(near, ws) if x == y)
    assert shared / len(ws) > 0.85, shared / len(ws)


def test_the_index_builder_reads_the_committed_corpus():
    paths = [p for p in ("data/copybench_test.jsonl", "data/copybench_val.jsonl",
                         "data/copybench_attack_train.jsonl") if os.path.exists(p)]
    if not paths:
        raise AssertionError("the committed prompt sets are missing; this test cannot be vacuous")
    idx = build_index(paths, 10)
    assert len(idx) > 100_000, len(idx)
