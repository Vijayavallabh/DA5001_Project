"""feat-165: TokenSwap as a decoder, checked against its authors' definition.

The arm exists because Appendix J said the method could not be measured without a vetted
auxiliary, and that is now false. What these guard is that what we run IS their method: their
110-word set taken verbatim rather than reconstructed, and their Algorithm 1's two invariants --
the mass on G is preserved and nothing off G moves.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.tokenswap_decode import g_token_ids, load_G, swap  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_G_is_their_published_set_verbatim():
    """110 unique words, from their Appendix C.3. A reconstruction from the DESCRIPTION (top-500
    COCA, NLTK POS filter) would be our set, not theirs, and the paper would be measuring a method
    of ours under their name."""
    w = load_G()
    assert len(w) == 110 and len(set(w)) == 110
    # spot-check the two the paper itself names as examples, and the shape of the list
    assert w[0] == "the" and "in" in w
    assert all(x.isalpha() and x.islower() for x in w), "G is a word list, not token pieces"
    # their POS classes must actually be present: determiners, modals, wh-words, auxiliaries
    for w_ in ("the", "a", "an", "might", "must", "which", "whether", "being", "having"):
        assert w_ in w, f"{w_} is in their list and not in ours"


def test_the_swap_preserves_the_mass_on_G_and_moves_nothing_off_it():
    """Their alpha exists precisely to keep p_final a distribution. Both halves are invariants:
    sum_G p_final == sum_G p_main, and p_final == p_main everywhere else."""
    torch.manual_seed(0)
    V, G = 50, torch.tensor([3, 7, 11, 29])
    p_main = torch.softmax(torch.randn(V), 0)
    p_aux = torch.softmax(torch.randn(V), 0)
    out, m = swap(p_main, p_aux, G)
    assert abs(float(out[G].sum()) - float(p_main[G].sum())) < 1e-6, "mass on G moved"
    assert abs(m - float(p_main[G].sum())) < 1e-9
    off = torch.ones(V, dtype=torch.bool)
    off[G] = False
    assert torch.allclose(out[off], p_main[off]), "a probability off G changed"
    assert abs(float(out.sum()) - 1.0) < 1e-5, "p_final is not a distribution"
    # and it really is the auxiliary that decides WITHIN G: the ordering on G follows p_aux
    assert torch.argsort(out[G]).tolist() == torch.argsort(p_aux[G]).tolist()


def test_the_swap_is_a_no_op_when_the_auxiliary_puts_nothing_on_G():
    V, G = 20, torch.tensor([2, 5])
    p_main = torch.softmax(torch.randn(V), 0)
    p_aux = torch.zeros(V)
    p_aux[0] = 1.0
    out, _ = swap(p_main, p_aux, G)
    assert torch.allclose(out, p_main), "a degenerate auxiliary must leave the served law alone"


def test_G_maps_into_the_llama_vocabulary_without_losing_the_set():
    """The auxiliary shares Llama-3's tokenizer, which is why this arm can use the identity
    mapping their paper only approximates. If most of G stopped being single tokens the rule would
    be weaker than the one they specify, so the loss is asserted rather than logged."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    cache = os.path.join(ROOT, "hf_cache")
    if not os.path.isdir(cache):
        pytest.skip("no local model cache")
    os.environ.setdefault("HF_HUB_CACHE", cache)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        tok = AutoTokenizer.from_pretrained("jacquelinehe/tinycomma-1.8b-llama3-tokenizer")
    except Exception as e:                                    # noqa: BLE001
        pytest.skip(f"tokenizer unavailable offline: {e}")
    ids, missing = g_token_ids(tok, load_G())
    assert len(missing) <= 5, f"{len(missing)} of 110 words are not single tokens: {missing}"
    assert len(ids) >= 200, f"G should reach both cased and spaced variants, got {len(ids)}"
    # the empty set would make the rule a no-op, which G1 exists to catch at run time too
    assert ids == sorted(set(ids))
