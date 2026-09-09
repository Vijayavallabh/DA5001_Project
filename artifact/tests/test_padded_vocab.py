"""Plan v4 / feat-038: checkpoints whose embedding table is padded above the tokenizer.

Common Pile comma-7b has 64,000 tokens and 64,256 embedding rows. The untrained pad rows must
never enter the divergence solve and must never be sampled, but their presence must not block
fusion -- that guard is what stopped a second anchor from being used at all.
"""
import os, sys, types
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a_patch.factory import AnchoredDecodingFactory


def _f(usable):
    f = types.SimpleNamespace(usable_vocab=usable)
    f._mask_pad_rows = types.MethodType(AnchoredDecodingFactory._mask_pad_rows, f)
    return f


def test_pad_rows_are_removed_from_both_vectors():
    safe = torch.zeros(2, 10)
    risky = torch.arange(20, dtype=torch.float).reshape(2, 10)
    s, r = _f(8)._mask_pad_rows(safe, risky)
    assert torch.isinf(s[:, 8:]).all() and (s[:, 8:] < 0).all()
    assert torch.isinf(r[:, 8:]).all() and (r[:, 8:] < 0).all()
    assert torch.isfinite(s[:, :8]).all() and torch.isfinite(r[:, :8]).all()


def test_masked_rows_carry_no_probability():
    """A -inf logit is exactly zero probability, so a pad token can never be sampled and
    contributes nothing to Z(theta), the KL or the max log-ratio."""
    logits = torch.randn(1, 10)
    s, _ = _f(8)._mask_pad_rows(logits, logits.clone())
    p = torch.softmax(s, dim=-1)
    assert p[:, 8:].sum().item() == 0.0
    assert abs(p.sum().item() - 1.0) < 1e-6


def test_unpadded_table_is_untouched():
    """Llama-3 has 128,256 rows for 128,256 tokens, so masking must be a no-op there."""
    a, b = torch.randn(2, 8), torch.randn(2, 8)
    s, r = _f(8)._mask_pad_rows(a, b)
    assert torch.equal(s, a) and torch.equal(r, b)


def test_masking_does_not_mutate_the_caller_tensors():
    a = torch.zeros(1, 4)
    _f(2)._mask_pad_rows(a, a.clone())
    assert torch.isfinite(a).all(), "the input logits were modified in place"


def test_no_tokenizer_disables_masking():
    a, b = torch.randn(1, 6), torch.randn(1, 6)
    s, r = _f(None)._mask_pad_rows(a, b)
    assert torch.equal(s, a) and torch.equal(r, b)
