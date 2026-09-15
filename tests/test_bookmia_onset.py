"""feat-120: the third protected corpus, and the three things that make it a third READING.

The arm's whole value is that it changes one variable. These pin the ones that are easy to break
silently:

(1) The 100-passage sweep corpus is a PREFIX of the 600 the memorisers train on, and both are
    stratified over every book. Caution (w) in a new costume: bookmia100_attack_train.jsonl is
    grouped by book, so a plain --limit 100 takes a hundred passages of *1984* and reports them as
    a hundred-passage corpus. Caution (h) is the other half: a passage the memoriser never saw is
    not a memorisation probe.
(2) The band-3 yardsticks quoted in the pre-registration round from results/onset_ci.csv, once --
    caution (j), which put six of seventy-two appendix cells one off in the last digit.
(3) The corpus switch on analysis/onset_gutenberg.py leaves the Gutenberg path alone.
"""
import collections
import csv
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "data/bench/bookmia100_onset100.jsonl")
BIG = os.path.join(ROOT, "data/bench/bookmia100_onset600.jsonl")
PREREG = os.path.join(ROOT, "results/onset_prediction_bookmia.md")


def _rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


needs_corpus = pytest.mark.skipif(
    not os.path.exists(SUB),
    reason="run analysis/build_bookmia_onset_subset.py first (data/bench is gitignored)")


@needs_corpus
def test_the_sweep_corpus_is_a_prefix_of_the_training_corpus():
    """A passage the memoriser never saw is not a memorisation probe -- caution (h)."""
    sub, big = _rows(SUB), _rows(BIG)
    assert len(sub) == 100 and len(big) == 600
    ids = [r["prompt_id"] for r in sub]
    assert ids == [r["prompt_id"] for r in big[:100]], "the 100 is not a prefix of the 600"


@needs_corpus
def test_both_corpora_cover_every_book_and_no_book_dominates():
    """Caution (w): --limit 100 on a book-grouped file takes one book and calls it a corpus."""
    src = os.path.join(ROOT, "data/bench/bookmia100_attack_train.jsonl")
    all_books = {r["source_novel"] for r in _rows(src)}
    for path, n in ((BIG, 600), (SUB, 100)):
        rows = _rows(path)
        per = collections.Counter(r["source_novel"] for r in rows)
        assert set(per) == all_books, (path, sorted(all_books - set(per))[:3])
        assert max(per.values()) - min(per.values()) <= 1, (path, per.most_common(3))
        assert max(per.values()) <= -(-n // len(all_books)), (path, per.most_common(1))


@needs_corpus
def test_the_corpus_reader_can_actually_read_these_records():
    """load_corpus_file wants raw_text + reference_text; a BookMIA record must carry both."""
    from analysis.corpus_file import load_corpus_file
    got = load_corpus_file(SUB)
    assert len(got) == 100
    assert all(r.get("raw_text") and r.get("reference_text") for r in _rows(SUB))
    # the reader falls back to a `prompt_text` field if raw_text is missing. BookMIA records carry
    # both, and the one it must read is raw_text -- the prefix the memoriser was trained on. The
    # header is added by the reader, identically for Gutenberg, which is why the two are comparable
    # (caution (t): the header is what a base model must not be handed, and these are LoRAs).
    raw = _rows(SUB)[0]["raw_text"]
    assert got[0].prompt_text == "Complete the prefix:\n" + raw
    assert got[0].novel_source == _rows(SUB)[0]["source_novel"]


def test_the_band_three_yardsticks_round_from_the_csv_once():
    """Caution (j): a paper number rounds from the CSV, once, and is checked mechanically."""
    ci = {r["pair"]: r for r in csv.DictReader(open(os.path.join(ROOT, "results/onset_ci.csv")))
          if r["mode"] == "single"}
    head = open(PREREG, encoding="utf-8").read().partition("\n## Scoring log")[0]
    pairs = {"KL3M-520M": "KL3M-520M + mem. KL3M-520M",
             "Pleias-1.2B": "Pleias-1.2B + mem. Pleias-1.2B",
             "Phi-3.5-mini": "Phi-3.5-mini + mem. Phi-3.5-mini"}
    seen = 0
    for short, full in pairs.items():
        m = re.search(rf"^{re.escape(short)}\s+([\d.]+)\s+\[([\d.]+), ([\d.]+)\]\s+([\d.]+)\s+"
                      r"([\d.]+)\s+([\d.]+)$", head, re.M)
        assert m, f"band-3 row for {short} is not in the committed table"
        seen += 1
        r = ci[full]
        assert float(m.group(1)) == round(float(r["ratio_point"]), 4), short
        assert float(m.group(2)) == round(float(r["ratio_lo95"]), 4), short
        assert float(m.group(3)) == round(float(r["ratio_hi95"]), 4), short
        width = float(r["ratio_hi95"]) - float(r["ratio_lo95"])
        assert abs(float(m.group(4)) - width) < 5e-5, (short, m.group(4), width)
        gut = ci[f"{short} (Gutenberg)"]
        assert float(m.group(5)) == round(float(gut["ratio_point"]), 4), short
        rng = abs(float(gut["ratio_point"]) - float(r["ratio_point"]))
        assert abs(float(m.group(6)) - rng) < 5e-5, (short, m.group(6), rng)
    assert seen == 3


def test_the_corpus_switch_leaves_the_gutenberg_path_alone():
    from analysis.onset_gutenberg import CORPORA
    assert set(CORPORA) == {"gutenberg", "bookmia"}
    for key, (pretty, pairs) in CORPORA.items():
        assert len(pairs) == 3
        assert all(lab.endswith(f"({pretty})") for lab, _, _ in pairs), key
        # every pair points at its own sweep directory, and the two corpora never share one
        assert len({run for _, run, _ in pairs}) == 3, key
    gut = {run for _, run, _ in CORPORA["gutenberg"][1]}
    book = {run for _, run, _ in CORPORA["bookmia"][1]}
    assert not (gut & book)
    # the CopyBench twins are the SAME three pairs -- that is what makes this a third reading
    assert ({t for _, _, t in CORPORA["gutenberg"][1]} == {t for _, _, t in CORPORA["bookmia"][1]})


def test_the_committed_grid_is_the_gutenberg_grid_verbatim():
    head = open(PREREG, encoding="utf-8").read().partition("\n## Scoring log")[0]
    gut = [r for r in csv.DictReader(open(os.path.join(ROOT, "results/onset_ci.csv")))
           if r["pair"].endswith("(Gutenberg)")]
    assert gut
    grid = gut[0]["k_grid"].split()
    assert all(g == grid for g in (r["k_grid"].split() for r in gut)), "Gutenberg pairs disagree"
    m = re.search(r"`-1 0 ([0-9. ]+)`", head)
    assert m, "the committed grid is no longer quoted in the pre-registration"
    assert m.group(1).split() == grid, (m.group(1).split(), grid)
