"""Plan v4 / feat-040: the Renyi charge must reproduce both existing constraints at its limits,
or it is not evidence that the frontier theorem is about the whole family."""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a_patch.renyi import renyi_divergence, solve_theta_renyi
from a_patch.pathwise import max_log_ratio

torch.manual_seed(0)
LPC = torch.log_softmax(torch.randn(4, 32).double(), dim=-1)
LPD = torch.log_softmax(torch.randn(4, 32).double(), dim=-1)


def _kl_reference(theta):
    """D_KL(p_theta || p_s) computed directly from the definition."""
    l = LPD - LPC
    lz = torch.logsumexp(LPC + theta.view(-1, 1) * l, dim=-1)
    logp = LPC + theta.view(-1, 1) * l - lz.view(-1, 1)
    return (logp.exp() * (logp - LPC)).sum(-1)


def test_alpha_one_is_the_kl_charge():
    th = torch.tensor([0.2, 0.5, 0.8, 1.0]).double()
    assert torch.allclose(renyi_divergence(LPC, LPD, th, 1.0), _kl_reference(th), atol=1e-9)


def test_alpha_infinity_is_the_pathwise_charge():
    th = torch.tensor([0.2, 0.5, 0.8, 1.0]).double()
    assert torch.allclose(renyi_divergence(LPC, LPD, th, float("inf")),
                          max_log_ratio(LPC, LPD, th), atol=1e-9)


def test_large_alpha_approaches_the_pathwise_charge():
    th = torch.tensor([0.5] * 4).double()
    inf = renyi_divergence(LPC, LPD, th, float("inf"))
    prev = None
    for a in (2.0, 8.0, 64.0, 512.0):
        gap = (inf - renyi_divergence(LPC, LPD, th, a)).abs().max().item()
        if prev is not None:
            assert gap < prev + 1e-12, "the gap to D_inf must shrink as alpha grows"
        prev = gap
    assert prev < 1e-2


def test_divergence_is_nondecreasing_in_theta_and_in_alpha():
    """Both monotonicities are load-bearing: theta for the bisection, alpha for the claim that the
    family orders the guarantees from average-case to worst-case."""
    for a in (1.0, 1.5, 2.0, 8.0, float("inf")):
        vals = [renyi_divergence(LPC, LPD, torch.full((4,), t).double(), a) for t in
                (0.0, 0.25, 0.5, 0.75, 1.0)]
        for lo, hi in zip(vals, vals[1:]):
            assert (hi >= lo - 1e-9).all(), f"not monotone in theta at alpha={a}"
        assert torch.allclose(vals[0], torch.zeros(4).double(), atol=1e-9)
    th = torch.tensor([0.6] * 4).double()
    order = [renyi_divergence(LPC, LPD, th, a) for a in (1.0, 1.5, 2.0, 8.0, float("inf"))]
    for lo, hi in zip(order, order[1:]):
        assert (hi >= lo - 1e-9).all(), "Renyi divergence must be nondecreasing in alpha"


def test_solver_returns_the_largest_feasible_theta():
    for a in (1.0, 2.0, float("inf")):
        k = torch.tensor([0.0, 0.05, 0.5, 1e6]).double()
        th = solve_theta_renyi(LPC, LPD, k, a)
        assert (renyi_divergence(LPC, LPD, th, a) <= k + 1e-6).all(), "infeasible theta returned"
        assert th[0] == 0.0, "a zero allowance must serve the anchor"
        assert th[3] == 1.0, "an enormous allowance must serve the risky model"
        interior = th[1:3]
        bumped = torch.clamp(interior + 1e-3, max=1.0)
        d = renyi_divergence(LPC[1:3], LPD[1:3], bumped, a)
        assert (d > k[1:3] - 1e-9).any(), "a larger theta was still feasible"


def test_a_tighter_budget_never_serves_more_of_the_risky_model():
    for a in (1.0, 2.0, float("inf")):
        loose = solve_theta_renyi(LPC, LPD, torch.full((4,), 1.0).double(), a)
        tight = solve_theta_renyi(LPC, LPD, torch.full((4,), 0.1).double(), a)
        assert (tight <= loose + 1e-9).all()
