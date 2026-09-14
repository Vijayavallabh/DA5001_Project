"""Each model gets its own tokenizer, and every checkpoint gets a padding token.

Two defects, both found on 2026-09-12 when the leakage arm was first run at an anchor other than
the audited one:

  * the safe model's token ids were fed to BOTH models, harmless only because the audited anchor
    ships the Llama-3 tokenizer the memoriser also uses;
  * `pad_token = eos_token` leaves pad None for a checkpoint that declares no special tokens at
    all, and Pleias-1.2B declares none -- no eos, no pad, no unk -- so `padding=True` raised eight
    seconds into the run.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = open(os.path.join(ROOT, "analysis", "selection_extraction.py"), encoding="utf-8").read()


def test_each_model_is_fed_its_own_tokenizer():
    assert re.search(r"sample\(anchor, stok,", SRC), "the anchor must sample with its own tokenizer"
    assert re.search(r"sample\(risky, rtok,", SRC), "the k=-1 baseline must use the risky one"
    assert re.search(r"score\(risky, rtok,", SRC), "scoring must use the risky one"


def test_the_seed_is_built_with_the_risky_tokenizer_so_it_is_the_same_at_every_anchor():
    """Otherwise the cross-anchor comparison carries the seed-convention confound Section 3
    measures at Spearman -0.958 -- a 20-token seed is a different number of words per tokenizer."""
    assert re.search(r"passages = build\(rtok,", SRC)


def test_the_padding_fallback_covers_a_checkpoint_with_no_special_tokens():
    """The order matters: an existing pad, then eos, then a [PAD]-like token in the vocabulary,
    then the config's eos id, then id 0. Left padding is masked out, so any real id is correct;
    the order prefers what the checkpoint meant."""
    fn = SRC[SRC.index("def load_tok("):SRC.index("stok, rtok =")]
    for token in ("[PAD]", "<pad>", "<|endoftext|>"):
        assert token in fn, token
    assert "AutoConfig.from_pretrained(name)" in fn
    assert "convert_ids_to_tokens" in fn
    # ordering is checked on the branches, not on the docstring, which names [PAD] first
    assert fn.index("t.pad_token = t.eos_token") < fn.index("t.pad_token = tid"), \
        "eos must be preferred over a [PAD]-like token found in the vocabulary"


def test_pleias_is_the_case_this_exists_for():
    """A named regression, so a future simplification back to `pad = eos` fails here first."""
    assert "Pleias" in SRC and "no special tokens" in SRC
