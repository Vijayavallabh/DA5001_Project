"""The MMLU answer parser, pinned to the OUTPUT FORMATS it has to read.

Written against the formats observed in output/phase5/mmlu_metered, before the repaired parser was
run on those generations (caution (ap)). Every case below is a real shape, and the expectations are
set by the format -- which letter the completion *states* -- never by which letter is correct.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_verifiable import extract_mmlu  # noqa: E402


def test_the_bare_answer():
    assert extract_mmlu("Answer: D") == "D"
    assert extract_mmlu(": B") == "B"
    assert extract_mmlu(" D") == "D"


def test_the_echo_that_broke_the_first_version():
    """A weak base model repeats the prompt's tail, then answers. This is the defect case."""
    assert extract_mmlu(" Falls\nD. Pierre\nAnswer: D") == "D"
    assert extract_mmlu(" Spain\nD. Mongolia\nAnswer: B") == "B"
    assert extract_mmlu(". a humanitarian.\nAnswer: B") == "B"


def test_the_run_on_into_the_next_question_is_cut():
    """An instruction-tuned model answers and then invents the next item; only the first answer
    counts, and a letter inside the NEXT question must never be read."""
    assert extract_mmlu("Answer: A\n\nQuestion: Statement 1| ...\nA. x\nB. y\nAnswer: C") == "A"


def test_an_echoed_option_letter_does_not_outrank_the_stated_answer():
    """The echo contains 'D. Pierre'; the answer marker says D too, but if they disagree the
    marker wins. Checked with a case where they DO disagree, so the test has teeth."""
    assert extract_mmlu(" Falls\nD. Pierre\nAnswer: A") == "A"


def test_an_out_of_range_letter_is_not_an_answer():
    assert extract_mmlu(". time\nAnswer: E") is None
    assert extract_mmlu("no letter here at all") is None


def test_prose_without_the_marker_still_parses():
    assert extract_mmlu("The answer is C.") == "C"


def test_the_word_answer_does_not_supply_its_own_A():
    """'Answer' starts with A; a word-boundary bug would read every completion as A."""
    assert extract_mmlu("Answer: B") == "B"


def test_the_last_marker_wins_when_an_echo_carries_an_earlier_one():
    """CONSTRUCTED, not observed -- and it is here because mutation-testing found a gap.

    Switching `rsplit` to `split` (first marker instead of last) passed every other case in this
    file, so that choice was unguarded. The shape it matters for is an echo that reaches back past
    a previous example's answer without emitting a new `Question:` header: the first marker is then
    the echoed example's answer and the last is the model's own. We have not seen this shape in
    output/phase5/mmlu_metered; it is guarded because the code makes a choice there and an
    unguarded choice is one the next edit can silently reverse.
    """
    assert extract_mmlu(" False, True\nAnswer: A\n\nWhat is the capital?\nAnswer: C") == "C"
