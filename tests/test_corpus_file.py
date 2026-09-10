"""The second corpus is read by a separate loader on purpose: dap.shared reads CopyBench by fixed
filename and the committed prompt sets are not to be touched. That makes this reader the only place
a malformed second corpus can be caught, so it is checked rather than trusted."""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.corpus_file import load_corpus_file  # noqa: E402


def _write(rows):
    fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    for r in rows:
        fh.write(json.dumps(r) + "\n")
    fh.close()
    return fh.name


def test_prefix_carries_the_same_instruction_the_memoriser_was_trained_on():
    """load_prompt_corpus prepends 'Complete the prefix:' to every CopyBench record. If this reader
    did not, the sweep would score a prompt the memoriser never saw."""
    p = _write([dict(prompt_id="a", raw_text="once upon", reference_text=" a time")])
    r = load_corpus_file(p)[0]
    assert r.prompt_text.startswith("Complete the prefix:\n")
    assert r.prompt_text.endswith("once upon") and r.reference == " a time"
    os.unlink(p)


def test_a_record_without_a_continuation_is_an_error_not_a_silent_skip():
    """An excerpt with no reference_text would train on the prefix alone and score against nothing,
    which is the failure mode that once made a truncated corpus look like a model too small to
    memorise."""
    p = _write([dict(prompt_id="a", raw_text="only a prefix")])
    with pytest.raises(ValueError):
        load_corpus_file(p)
    os.unlink(p)


def test_built_gutenberg_corpus_is_well_formed_if_present():
    path = "data/gutenberg/excerpts.jsonl"
    if not os.path.exists(path):
        return
    rows = load_corpus_file(path)
    assert len(rows) >= 100
    assert len({r.novel_source for r in rows}) >= 10          # not one book repeated
    for r in rows[:50]:
        assert "Project Gutenberg" not in r.prompt_text        # boilerplate stripped
        assert len(r.reference) > 100
