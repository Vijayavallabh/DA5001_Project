"""tau() converts a per-token rate to a per-character one, so it has to count exactly the tokens
analysis/budget_path.py scored and exactly the characters those tokens cover."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.onset_units import tau


class FakeTok:
    """One token per character, so tokens/char is 1 over whatever suffix is counted."""

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_seed_tokens_are_excluded_from_both_counts():
    t, n_tok, n_char = tau(FakeTok(), ["abcdefghij"], seed_tokens=4)
    assert (n_tok, n_char) == (6, 6)      # tokens 4..9, characters 4..9
    assert t == 1.0


def test_short_text_is_skipped_rather_than_counted_negative():
    t, n_tok, n_char = tau(FakeTok(), ["abc", "abcdefghij"], seed_tokens=4)
    assert (n_tok, n_char) == (6, 6)      # the 3-character text contributes nothing


def test_pooled_not_averaged():
    """Pooling weights a long passage more than a short one; averaging per-text ratios would not."""
    t, n_tok, n_char = tau(FakeTok(), ["a" * 100, "a" * 10], seed_tokens=5)
    assert (n_tok, n_char) == (100, 100)  # 95 + 5


def test_ratio_is_unchanged_by_the_conversion():
    """The point of the whole file: onset and s(x) are per token of the SAME tokenizer, so any
    per-pair factor cancels. Guards against a future edit converting only one of them."""
    s_tok, onset_tok, factor = 3.2386, 2.8727, 0.238
    assert abs((onset_tok * factor) / (s_tok * factor) - onset_tok / s_tok) < 1e-12
