"""Plan v5: a tokenizer with no special tokens at all must still get a usable pad token.

Pleias-1.2b/3b register pad, eos, unk and bos all as None, so the old `pad = eos` fallback
assigned None and training died on the first padded batch. The replacement must pick a token that
ALREADY EXISTS, because adding one would make len(tokenizer) exceed the model's embedding rows and
trip the padded-vocab check in a_patch/factory.py.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RECIPE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "recipes", "finetune_memorizing.py")


class FakeTok:
    """Minimal stand-in: the fallback only needs get_vocab and convert_ids_to_tokens."""

    def __init__(self, eos=None, unk=None, bos=None, vocab=None):
        self.eos_token, self.unk_token, self.bos_token = eos, unk, bos
        self._vocab = vocab if vocab is not None else {"a": 0, "<|end_of_text|>": 2}
        self.pad_token = None

    def get_vocab(self):
        return dict(self._vocab)

    def convert_ids_to_tokens(self, i):
        return next(t for t, v in self._vocab.items() if v == i)

    @property
    def pad_token_id(self):
        return self._vocab.get(self.pad_token)


def choose_pad(tok):
    """The fallback as written in the recipe."""
    vocab = tok.get_vocab()
    for cand in (tok.eos_token, tok.unk_token, tok.bos_token,
                 "<|end_of_text|>", "<|endoftext|>", "</s>"):
        if cand and cand in vocab:
            return cand
    return tok.convert_ids_to_tokens(0)


def test_no_special_tokens_at_all_still_resolves():
    tok = FakeTok()
    assert choose_pad(tok) == "<|end_of_text|>"


def test_prefers_a_registered_eos():
    tok = FakeTok(eos="</s>", vocab={"a": 0, "</s>": 1, "<|end_of_text|>": 2})
    assert choose_pad(tok) == "</s>"


def test_falls_back_to_token_zero_when_nothing_matches():
    tok = FakeTok(vocab={"zero": 0, "one": 1})
    assert choose_pad(tok) == "zero"


def test_choice_never_grows_the_vocabulary():
    for tok in (FakeTok(), FakeTok(vocab={"zero": 0, "one": 1})):
        before = len(tok.get_vocab())
        assert choose_pad(tok) in tok.get_vocab()
        assert len(tok.get_vocab()) == before


def test_recipe_does_not_use_the_broken_eos_only_fallback():
    src = open(RECIPE, encoding="utf-8").read()
    assert not re.search(r"if tok\.pad_token_id is None:\s*\n\s*tok\.pad_token = tok\.eos_token\s*\n", src)
    assert "add_special_tokens" not in src, "adding a pad token would break the padded-vocab check"
