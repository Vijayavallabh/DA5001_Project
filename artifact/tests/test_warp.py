"""feat-022: temperature and repetition-penalty warping of teacher-forced logits, matching HF's processors
(He et al. apply both to each model's logits before the KL solve, so the certificate is relative to the warped anchor)."""
import torch

from dap.warp import warp_logits


def test_identity_when_unwarped():
    logits = torch.randn(5, 11)
    ids = torch.tensor([3, 4, 3, 7, 1])
    out = warp_logits(logits, ids, temperature=1.0, repetition_penalty=1.0)
    assert torch.allclose(out, logits)


def test_temperature_divides_logits():
    logits = torch.randn(4, 9)
    ids = torch.tensor([0, 1, 2, 3])
    out = warp_logits(logits, ids, temperature=0.5, repetition_penalty=1.0)
    assert torch.allclose(out, logits / 0.5)


def test_repetition_penalty_applies_to_tokens_seen_before_the_position():
    logits = torch.zeros(3, 6)
    logits[:, 2] = 2.0   # positive logit for token 2 everywhere
    logits[:, 4] = -2.0  # negative logit for token 4 everywhere
    ids = torch.tensor([2, 4, 0])  # token 2 is seen from position 1 on, token 4 from position 2 on
    out = warp_logits(logits, ids, temperature=1.0, repetition_penalty=2.0)
    assert out[0, 2] == 2.0 and out[1, 2] == 1.0 and out[2, 2] == 1.0
    assert out[1, 4] == -2.0 and out[2, 4] == -4.0
