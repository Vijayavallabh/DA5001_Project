"""feat-210: the sliding-window pathwise meter (the design Section 3 leaves open).

What it certifies is every span of at most `window` consecutive output tokens: on every path the
realised log-ratio of such a span is at most W, so any event on it has q(E) <= e^W p_s(E). These tests
drive the real decode loop on two tiny random models on CPU (no download) and check that property,
and that it is a WINDOW bound rather than a total one: the same outputs spend far more than W in all."""
import random

import pytest
import torch

from a_patch.pathwise import max_span_sum, max_suffix_sum


def _brute_span(xs, w):
    best = 0.0
    for i in range(len(xs)):
        for j in range(i, min(len(xs), i + w)):
            best = max(best, sum(xs[i:j + 1]))
    return best


def test_max_suffix_sum_matches_brute_force_and_earns_no_credit():
    g = torch.Generator().manual_seed(0)
    h = torch.randn(7, 9, generator=g) * 3
    got = max_suffix_sum(h)
    for b in range(7):
        row = h[b].tolist()
        want = max([0.0] + [sum(row[j:]) for j in range(len(row))])
        assert got[b].item() == pytest.approx(want, abs=1e-5)
    assert torch.equal(max_suffix_sum(torch.zeros(3, 0)), torch.zeros(3))
    # all-negative history: no credit, the allowance is the full W
    assert max_suffix_sum(-torch.ones(2, 4)).tolist() == [0.0, 0.0]


def test_max_span_sum_matches_brute_force():
    rng = random.Random(1)
    for _ in range(200):
        xs = [rng.uniform(-3, 5) for _ in range(rng.randint(0, 30))]
        w = rng.randint(1, 12)
        assert max_span_sum(xs, w) == pytest.approx(_brute_span(xs, w), abs=1e-9)


class _Tok:
    pad_token_id = 0
    eos_token_id = None

    def __len__(self):
        return 48

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(map(str, ids))


def _tiny(seed, scale):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=48, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128)
    m = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():
        m.lm_head.weight.mul_(scale)          # peaky and unlike the other model: large log-ratios
    return m


def _decode(window, W, steps=40, batch=6, seed=3):
    from transformers import LogitsProcessorList, MaxLengthCriteria, StoppingCriteriaList
    from a_patch.factory import AnchoredDecodingFactory
    f = AnchoredDecodingFactory(_tiny(11, 1.0), _tiny(22, 40.0), _Tok(), k_radius=W, use_prefix_debt=False,
                                log_kl_stats=True, constraint="pathwise", window=window,
                                device=torch.device("cpu"))
    torch.manual_seed(seed)
    ids = torch.randint(1, 48, (batch, 5))
    f._decode(ids, torch.ones_like(ids), StoppingCriteriaList([MaxLengthCriteria(max_length=5 + steps)]),
              LogitsProcessorList(), LogitsProcessorList(), pad_token_id=0, eos_token_id=None,
              k_radius=W, do_sample=True)
    per = f.get_kl_stats_summary()["per_step"]
    return [[float(s["r_t"][b]) for s in per] for b in range(batch)]


@pytest.mark.parametrize("window,W", [(5, 3.0), (8, 12.0), (1, 2.0)])
def test_every_span_of_the_window_stays_within_W_on_every_path(window, W):
    rows = _decode(window, W)
    for r in rows:
        assert max_span_sum(r, window) <= W + 1e-3, (window, W, max_span_sum(r, window))


def test_it_is_a_window_bound_not_a_total_one():
    """The same meter spends far more than W over the whole output: it is metered per window, and the
    risky model is served wherever the window has room. Without this the first test could pass on a
    decoder that simply served the anchor."""
    rows = _decode(5, 3.0)
    assert max(sum(max(0.0, x) for x in r) for r in rows) > 3 * 3.0
    assert max(max(r) for r in rows) > 1.0          # some single token really is tilted


def test_the_factory_refuses_a_window_on_any_other_meter():
    from a_patch.factory import AnchoredDecodingFactory
    for kw in (dict(constraint="kl"), dict(constraint="pathwise", use_prefix_debt=True),
               dict(constraint="pathwise", initial_bank=1.0)):
        base = dict(k_radius=3.0, use_prefix_debt=False, log_kl_stats=True, window=5, device=torch.device("cpu"))
        base.update(kw)
        with pytest.raises(AssertionError):
            AnchoredDecodingFactory(_tiny(1, 1.0), _tiny(2, 1.0), _Tok(), **base)
