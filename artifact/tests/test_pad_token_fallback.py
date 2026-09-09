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


from a_patch.tokenizer import ensure_pad_token  # noqa: E402


def choose_pad(tok):
    """Exercise the real shared helper, not a copy of it."""
    return ensure_pad_token(tok).pad_token


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


def test_neither_recipe_nor_decoder_uses_the_broken_eos_only_fallback():
    """The decoder had the same bug as the recipe; both must route through the shared helper."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in ("recipes/finetune_memorizing.py", "a_patch/tokenizer.py"):
        src = open(os.path.join(root, rel), encoding="utf-8").read()
        assert not re.search(r"pad_token_id is None:\s*\n\s*\w+\.pad_token = \w+\.eos_token\s*\n", src), rel
        assert "add_special_tokens" not in src, f"{rel}: adding a pad token breaks the padded-vocab check"


def test_helper_is_idempotent_and_leaves_a_good_tokenizer_alone():
    tok = FakeTok(eos="</s>", vocab={"a": 0, "</s>": 1})
    tok.pad_token = "</s>"
    ensure_pad_token(tok)
    assert tok.pad_token == "</s>"


def test_eos_criteria_is_skipped_when_the_model_has_no_eos():
    """Pleias 1.2b/3b register no eos; EosTokenCriteria(None) raises, so it must be skipped."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "a_patch", "factory.py"), encoding="utf-8").read()
    i = src.index("def _prepare_stopping_criteria")
    body = src[i:src.index("def ", i + 10)]
    assert "if generation_config.eos_token_id is not None:" in body, \
        "EosTokenCriteria must be guarded against a None eos_token_id"


def test_decode_handles_a_model_with_no_eos_token():
    """_decode built a tensor from list(None). Both downstream uses already guard on None, so the
    empty list is the consistent representation -- and it needs an explicit dtype."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "a_patch", "factory.py"), encoding="utf-8").read()
    i = src.index("eos_token_id_list = []")
    assert "if eos_token_id is None:" in src[max(0, i - 600):i]
    j = src.index("eos_token_id_tensor = torch.tensor(")
    assert "dtype=torch.long" in src[j:j + 160], "empty eos list needs an explicit dtype"


def test_empty_eos_tensor_is_constructible():
    import torch
    t = torch.tensor([], dtype=torch.long)
    assert t.numel() == 0
    nxt = torch.tensor([[5], [7]])
    assert not (nxt == t).any(dim=-1).any(), "no token matches an empty eos set"
