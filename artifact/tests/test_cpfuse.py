"""feat-030: the CP-Fuse fusion rule. A sign error here would silently invert the defence."""
import torch

from analysis.cpfuse_audit import cpfuse_step, fuse_logits


def test_fuse_is_a_normalised_geometric_mixture():
    l1 = torch.tensor([2.0, 0.0, -1.0])
    l2 = torch.tensor([-1.0, 0.0, 2.0])
    for a in (0.0, 0.5, 1.0):
        lp = fuse_logits(l1, l2, a)
        assert torch.allclose(lp.exp().sum(), torch.tensor(1.0), atol=1e-5)
    # the endpoints recover each component exactly
    assert torch.allclose(fuse_logits(l1, l2, 1.0), torch.log_softmax(l1, -1), atol=1e-5)
    assert torch.allclose(fuse_logits(l1, l2, 0.0), torch.log_softmax(l2, -1), atol=1e-5)


def test_balancing_pulls_down_the_model_that_already_explains_the_output():
    """If model 1's running log-likelihood is far higher, alpha must move away from model 1."""
    l1 = torch.tensor([4.0, 0.0, 0.0])
    l2 = torch.tensor([0.0, 0.0, 4.0])
    _, a_hi = cpfuse_step(l1, l2, c1=0.0, c2=-40.0)   # model 1 explains the prefix far better
    _, a_lo = cpfuse_step(l1, l2, c1=-40.0, c2=0.0)   # model 2 does
    assert a_hi < a_lo, f"balancing did not react to the imbalance: {a_hi} vs {a_lo}"
    assert a_hi <= 0.5 <= a_lo


def test_symmetric_history_gives_a_balanced_weight():
    l1 = torch.tensor([1.0, 0.0, -1.0])
    l2 = torch.tensor([-1.0, 0.0, 1.0])
    _, a = cpfuse_step(l1, l2, c1=-5.0, c2=-5.0)
    assert abs(a - 0.5) <= 0.05


def test_balance_off_is_a_fixed_even_mixture():
    l1 = torch.tensor([3.0, 0.0, 0.0])
    l2 = torch.tensor([0.0, 0.0, 3.0])
    lp, a = cpfuse_step(l1, l2, c1=0.0, c2=-99.0, balance=False)
    assert a == 0.5
    assert torch.allclose(lp, fuse_logits(l1, l2, 0.5), atol=1e-6)
