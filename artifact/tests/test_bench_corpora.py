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


def test_the_instruction_benchmarks_are_complete_and_are_not_wrapped_as_prefixes():
    """They go through the FACTUAL slot on purpose: dap/shared.py prepends "Complete the prefix:"
    to every copyright-domain prompt, which is right for a passage and wrong for an instruction.
    Through the neutral slot the number would not be AlpacaEval's number."""
    from dap.shared import load_prompt_corpus
    for d, n, groups in (("data/bench/alpaca", 805, 5), ("data/bench/mtbench", 80, 8)):
        rows = [x for x in load_prompt_corpus(d, "factscore_prompt") if x.split == "factual"]
        assert len(rows) == n, (d, len(rows))
        assert len({x.novel_source for x in rows}) == groups, d
        assert not any(x.prompt_text.startswith("Complete the prefix") for x in rows), d
        assert all(x.prompt_text.strip() for x in rows), d


def test_the_artifact_builder_excludes_refetchable_data_and_stray_caches():
    """Twice something large and re-fetchable has been swept into the artifact: 44 MB of Gutenberg
    books in September 2026, then 1.9 GB of driver payload and 67 MB of bench corpora. The size
    guard catches the class; these assertions catch the two paths by name so a future rsync edit
    cannot quietly drop them."""
    src = open("scripts/build_artifact.sh", encoding="utf-8").read()
    for path in ("data/gutenberg", "data/bench", "NVIDIA-Linux-*", "torchinductor_*"):
        assert f"--exclude '{path}'" in src, path
    assert "ARTIFACT_MAX_MB" in src and "-gt" in src, "the size guard is gone"


def test_the_anonymity_scan_covers_path_names_not_only_contents():
    """torch's compile cache is named torchinductor_$USER and shipped for as long as the artifact
    has existed, carrying the account name where no content grep would see it."""
    src = open("scripts/build_artifact.sh", encoding="utf-8").read()
    assert "identifying strings in PATH names" in src
    assert "-iname '*sports*'" in src


def test_the_built_artifact_carries_no_identifying_path():
    """The tokens are read out of the builder rather than written here. Spelling them in a test
    put them inside the artifact -- tests/ ships, scripts/build_artifact.sh does not -- and the
    builder's own content scan then failed on the file that guards it. One home for the list."""
    import glob
    import os
    import re
    src = open("scripts/build_artifact.sh", encoding="utf-8").read()
    toks = re.findall(r"-iname '\*([^*']+)\*'", src)
    assert len(set(toks)) >= 3, toks
    if not os.path.isdir("artifact"):
        return
    bad = [p for p in glob.glob("artifact/**/*", recursive=True)
           if any(k.lower() in os.path.basename(p).lower() for k in toks)]
    assert not bad, bad


def test_the_triviaqa_corpus_is_a_symlink_dir_with_gold_aliases():
    """The judge-free head-to-head needs both mechanisms on one corpus through one code path. The
    committed prompt sets stay read-only, so the corpus is a directory of symlinks with one real
    file swapped into the factual slot -- and that file must carry the gold aliases, or the scorer
    has to re-download the dataset to join on prompt_id."""
    import json
    import os
    d = "data/bench/triviaqa"
    if not os.path.isdir(d):
        return                       # rebuilt by analysis/build_bench_corpora.py
    for f in os.listdir(d):
        assert os.path.islink(os.path.join(d, f)), f
    real = os.path.realpath(os.path.join(d, "factscore.jsonl"))
    assert real.endswith("triviaqa_factual.jsonl"), real
    rows = [json.loads(l) for l in open(real, encoding="utf-8")]
    assert rows, "empty corpus"
    for r in rows[:20]:
        assert r["split"] == "factual", "the neutral slot would prepend 'Complete the prefix:'"
        assert r["prompt_text"].rstrip().endswith("Answer:"), r["prompt_text"][-40:]
        assert r["prompt_text"].count("Question:") >= 2, "the few-shot prefix is missing"
        assert r["reference"].strip(), "no gold aliases to score against"


def test_containment_scoring_is_whole_word_and_not_substring():
    """'ann' must not match inside 'anne'. The relaxation exists to absorb an instruct model's
    'The answer is X' preamble, not to hand out credit for a prefix of the right word."""
    from analysis.selection_verifiable import correct_tqa, norm_answer
    gold = {"david seville"}
    assert correct_tqa(norm_answer("David Seville"), gold)
    assert correct_tqa(norm_answer("The answer is David Seville."), gold)
    assert not correct_tqa(norm_answer("Dave"), gold)
    assert not correct_tqa(None, gold)
    # whole-word, both directions: a plural is not the alias and a prefix is not the alias
    assert not correct_tqa(norm_answer("annes"), {"ann"})
    assert not correct_tqa(norm_answer("David Sevilles brother"), {"seville"})
    # articles and punctuation are normalised away on both sides
    assert correct_tqa(norm_answer("the Beatles!"), {"beatles"})
