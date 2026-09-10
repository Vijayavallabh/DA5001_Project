"""The greedy-vs-equal-price probe rests on closed forms for the charge and the fidelity on the
geodesic. If those are wrong the whole finding is wrong, so they are checked against numerical
derivatives and against the two endpoints where the answer is known in closed form."""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.marginal_price import (  # noqa: E402
    _psi, charge, fidelity, solve_theta_dual, solve_theta_greedy,
)


def _pair(seed=0, T=6, V=32):
    g = torch.Generator().manual_seed(seed)
    log_ps = torch.log_softmax(torch.randn(T, V, generator=g).double(), dim=-1)
    log_pr = torch.log_softmax(torch.randn(T, V, generator=g).double() * 2.0, dim=-1)
    return log_ps, torch.nan_to_num(log_pr - log_ps), log_pr


def test_psi_derivatives_match_finite_differences():
    log_ps, l, _ = _pair()
    u, h = 0.37, 1e-5
    psi, m1, var = _psi(log_ps, l, u)
    psi_p = _psi(log_ps, l, u + h)[0]
    psi_m = _psi(log_ps, l, u - h)[0]
    assert torch.allclose(m1, (psi_p - psi_m) / (2 * h), atol=1e-6)      # psi' = E_u[l]
    d2 = (psi_p - 2 * psi + psi_m) / (h * h)
    assert torch.allclose(var, d2, atol=1e-3)                             # psi'' = Var_u[l]


def test_charge_is_zero_at_zero_and_increasing():
    log_ps, l, _ = _pair(1)
    assert torch.allclose(charge(log_ps, l, 0.0), torch.zeros(log_ps.size(0)).double(), atol=1e-12)
    prev = charge(log_ps, l, 0.0)
    for th in (0.2, 0.4, 0.6, 0.8, 1.0):
        cur = charge(log_ps, l, th)
        assert (cur >= prev - 1e-12).all(), th
        prev = cur


def test_fidelity_endpoints_are_the_two_known_divergences():
    """G(0) = 0 and G(1) = D_KL(p_r || p_s); the charge at 1 is D_KL(p_r || p_s) as well, since
    p_1 = p_r. Both are what make G interpretable as fidelity bought."""
    log_ps, l, log_pr = _pair(2)
    m = (log_pr.exp() * l).sum(-1)
    kl_rs = (log_pr.exp() * l).sum(-1)                       # D(p_r || p_s)
    assert torch.allclose(fidelity(log_ps, l, 0.0, m), torch.zeros_like(m), atol=1e-12)
    assert torch.allclose(fidelity(log_ps, l, 1.0, m), kl_rs, atol=1e-9)
    assert torch.allclose(charge(log_ps, l, 1.0), kl_rs, atol=1e-9)


def test_greedy_theta_respects_its_allowance():
    log_ps, l, _ = _pair(3)
    for k in (0.05, 0.5, 5.0):
        th = solve_theta_greedy(log_ps, l, k)
        assert ((th >= 0) & (th <= 1)).all()
        # feasible to within the bisection's own resolution, and saturating only when it may
        assert (charge(log_ps, l, th) <= k + 1e-9).all(), k
        at_one = th >= 1 - 1e-9
        assert (charge(log_ps, l, torch.ones_like(th))[at_one] <= k + 1e-9).all()


def test_equal_price_allocation_beats_greedy_at_the_same_total_spend():
    """The finding is a comparison, so the comparison has to be sound: at a matched total budget the
    equal-marginal-price allocation must buy at least as much fidelity as greedy. If this ever fails
    the reported gain_ratio is measuring a bug, not a decoder."""
    log_ps, l, log_pr = _pair(4, T=24, V=64)
    m = (log_pr.exp() * l).sum(-1)
    k = 0.3
    th_g = solve_theta_greedy(log_ps, l, k)
    total = float(charge(log_ps, l, th_g).sum())

    lo, hi = 1e-6, 1e6
    for _ in range(60):
        lam = math.sqrt(lo * hi)
        if float(charge(log_ps, l, solve_theta_dual(log_ps, l, m, lam)).sum()) > total:
            lo = lam
        else:
            hi = lam
    th_d = solve_theta_dual(log_ps, l, m, math.sqrt(lo * hi))

    assert float(charge(log_ps, l, th_d).sum()) <= total * (1 + 1e-3)
    assert float(fidelity(log_ps, l, th_d, m).sum()) >= float(fidelity(log_ps, l, th_g, m).sum())


def test_the_ceiling_bounds_every_allocation():
    """theta = 1 everywhere is the risky model served outright, so no budget can buy more fidelity
    than that. It is the bound the paper's claim rests on, so it is asserted rather than assumed."""
    log_ps, l, log_pr = _pair(5, T=16)
    m = (log_pr.exp() * l).sum(-1)
    ceiling = float(fidelity(log_ps, l, 1.0, m).sum())
    for k in (0.1, 1.0, 10.0):
        th = solve_theta_greedy(log_ps, l, k)
        assert float(fidelity(log_ps, l, th, m).sum()) <= ceiling + 1e-9


@pytest.mark.parametrize("path", ["results/marginal_price_table.csv"])
def test_committed_table_is_internally_consistent(path):
    import csv
    if not os.path.exists(path):
        return
    for r in csv.DictReader(open(path)):
        fg, fd = float(r["frac_ceiling_greedy"]), float(r["frac_ceiling_dual"])
        assert 0 < fg <= 1 and 0 < fd <= 1, r
        assert fd >= fg, r                        # reallocation cannot lose at matched spend
        assert abs(float(r["gain_ratio"]) - fd / fg) < 5e-3, r
        assert float(r["gain_lo"]) <= float(r["gain_ratio"]) <= float(r["gain_hi"]), r
