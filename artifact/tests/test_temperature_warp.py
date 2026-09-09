"""Plan v5: token_nats must warp logits the way the decoder does before its solve.

Lowering the temperature sharpens a model, so its surprisal of a token it does NOT favour rises
and its surprisal of the argmax falls. That is the lever used to move s(x) across a range while
holding the model pair -- and therefore its memorisation -- fixed.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.regimes import token_nats  # noqa: E402


class StubTok:
    """Two-token vocabulary; 'ab' -> [0, 1]."""
    bos_token_id, eos_token_id = 0, 0

    def __call__(self, text, add_special_tokens=True):
        class E:
            input_ids = [0] + [1] * (len(text) - 1) if add_special_tokens else [1] * len(text)
        return E()


class StubModel:
    """Constant logits [2.0, 0.0] at every position: token 1 is the non-favoured one."""
    def __init__(self): self.device = "cpu"

    def __call__(self, ids):
        class O:
            logits = torch.tensor([[[2.0, 0.0]] * ids.shape[1]])
        return O()


def _nats(temperature):
    n, _ = token_nats(StubModel(), StubTok(), "a", "b", "cpu", temperature=temperature)
    return n[0]


def test_default_temperature_is_a_no_op():
    expected = -math.log(math.exp(0.0) / (math.exp(2.0) + math.exp(0.0)))
    assert abs(_nats(1.0) - expected) < 1e-6


def test_lower_temperature_raises_surprisal_of_the_unfavoured_token():
    assert _nats(0.5) > _nats(1.0) > _nats(2.0)


def test_warping_matches_the_closed_form():
    for t in (0.3, 0.7, 1.0, 2.0):
        z = [2.0 / t, 0.0 / t]
        expected = -(z[1] - math.log(sum(math.exp(v) for v in z)))
        assert abs(_nats(t) - expected) < 1e-6, t
