"""The LAMBADA answer parser, pinned to output FORMATS before the arm was run.

The MMLU arm was invalid twice, the first time because a parser that could not read a base model's
output shape scored 301 of 500 completions as wrong and put a four-way choice below its own floor
(results/onset_prediction_mmlu_headtohead.md). The lesson is to validate the parser against the
shapes a completion model produces BEFORE running, and against the format rather than against which
answer is correct.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_verifiable import extract_lambada, norm_answer  # noqa: E402


def test_the_bare_continuation():
    assert extract_lambada(" window") == "window"
    assert extract_lambada("window") == "window"


def test_the_model_runs_on_and_only_the_first_word_counts():
    assert extract_lambada(" window. And then he turned away.") == "window"
    assert extract_lambada(" window,\nand the rest followed") == "window"


def test_punctuation_and_quoting_are_stripped_the_same_way_the_gold_is():
    """Both sides go through norm_answer, so the comparison cannot fail on a comma."""
    assert extract_lambada(' "Window," she said') == norm_answer("Window")
    assert extract_lambada(" window's") == norm_answer("window's")


def test_an_empty_or_whitespace_completion_is_not_an_answer():
    assert extract_lambada("") is None
    assert extract_lambada("   \n  ") is None


def test_the_task_has_no_few_shot_prefix_because_the_task_is_the_format():
    """A completion task needs no exemplars, and adding them would change what is measured."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "analysis", "selection_verifiable.py"), encoding="utf-8").read()
    i = src.index("def load_lambada")
    body = src[i:i + 1600]
    assert 'return "", items' in body, "load_lambada must return an empty shot prefix"
