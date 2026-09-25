"""feat-214 (review 3 Q13): the short-works metric, the author-status rule and the padded-batch surprisal."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.short_works import exact, near, status, surprisal  # noqa: E402


def test_exact_ignores_case_and_punctuation_and_needs_the_target_first():
    t = " everyone else is already taken."
    assert exact(" everyone else is already taken.” ― Oscar Wilde", t)
    assert exact(" Everyone else, is already TAKEN!", t)
    assert not exact(" everyone else is taken.", t)
    assert not exact(" everyone else is already given.", t)  # every word, the last one included
    assert not exact(" so everyone else is already taken.", t)  # not the continuation's first words
    assert not exact("", t) and not exact(" anything", " ")
    assert exact(" don’t stop", " don't stop")


def test_near_is_a_common_run_of_most_of_the_target_anywhere():
    t = " one two three four five"
    assert near(" and then one two three four five", t)
    assert near(" x one two three four y", t)  # 4 of 5 = 80%
    assert not near(" one two three x four five", t)  # longest run 3 of 5
    assert not near(" ", t)


def test_status_follows_the_death_year_rule():
    assert status("J.K. Rowling") == "protected"
    assert status("A.A. Milne") == "protected"  # died 1956: in copyright under life plus 70 in 2026
    assert status("Oscar Wilde") == "public_domain"
    assert status("Albert Einstein") == "excluded"  # died 1955, between the two thresholds
    assert status("Friedrich Nietzsche") == "excluded"  # English text is a translation
    assert status("Anonymous") == "excluded"


class Bigram(torch.nn.Module):
    """Logits at a position depend only on the token there, so padding cannot move them."""

    def __init__(self, v=50):
        super().__init__()
        torch.manual_seed(0)
        self.w = torch.nn.Embedding(v, v)

    def forward(self, input_ids, attention_mask=None):
        return type("O", (), {"logits": self.w(input_ids)})()


class Tok:
    pad_token_id = 0

    def __call__(self, s):
        return type("E", (), {"input_ids": [1] + [2 + ord(c) % 40 for c in s]})()


def test_surprisal_is_the_same_alone_and_inside_a_left_padded_batch():
    m, tok = Bigram(), Tok()
    pairs = [("ab", "abcdef"), ("a", "ab"), ("abcdefgh", "abcdefghij"), ("", "xyz")]
    batch = surprisal(m, tok, pairs, bs=4)
    alone = [surprisal(m, tok, [p], bs=1)[0] for p in pairs]
    for (sb, nb), (sa, na) in zip(batch, alone):
        assert nb == na and abs(sb - sa) < 1e-5
    assert [n for _, n in batch] == [4, 1, 2, 3]
    lp = torch.log_softmax(m.w(torch.tensor([1, 2 + ord("a") % 40])), -1)
    assert abs(alone[1][0] + float(lp[1, 2 + ord("b") % 40])) < 1e-5  # -log p(b | a), read off the position before it
