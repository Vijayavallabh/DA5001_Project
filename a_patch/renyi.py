"""Renyi-alpha constraint for anchored decoding (plan v4 / feat-040).

The audited decoder charges D_KL(p_theta || p_s); the pathwise variant charges the max log-ratio.
Both are members of one family, and the family has a closed form on the geometric path, which is
what lets a single knob span them.

With p_theta ∝ p_s^(1-theta) p_r^theta and l = log p_r - log p_s, write
    Z(u) = sum_v p_s(v) e^{u l(v)},   so   log p_theta = log p_s + theta l - log Z(theta).
Then
    D_alpha(p_theta || p_s) = 1/(alpha-1) * log sum_v p_theta^alpha p_s^(1-alpha)
                            = [ log Z(alpha*theta) - alpha * log Z(theta) ] / (alpha - 1).

Limits, both checked in tests/test_renyi.py:
    alpha -> 1    D_1 = E_theta[log p_theta/p_s] = D_KL(p_theta || p_s)      the He et al. charge
    alpha -> inf  D_inf = max_v log p_theta(v)/p_s(v)                        the pathwise charge

So sweeping alpha moves continuously from a guarantee about an average to a guarantee about every
event, and the frontier theorem can be tested along the whole family rather than at its two ends.
D_alpha is nondecreasing in alpha and in theta, with D(0) = 0, so the same bisection works.
"""
from __future__ import annotations

import argparse

import torch


def constraint_arg(value: str) -> str:
    """argparse `type=` for --constraint: 'kl', 'pathwise', or 'renyi[:alpha]' with alpha >= 1.

    A `choices=` list cannot express a continuous parameter, so validate instead of enumerating.
    Bare 'renyi' means alpha = 2.
    """
    if value in ("kl", "pathwise", "renyi"):
        return value
    head, sep, tail = value.partition(":")
    if head == "renyi" and sep:
        try:
            alpha = float(tail)
        except ValueError:
            raise argparse.ArgumentTypeError(f"renyi alpha must be a number, got {tail!r}")
        if alpha < 1.0:
            raise argparse.ArgumentTypeError(f"renyi alpha must be >= 1, got {alpha}")
        return value
    raise argparse.ArgumentTypeError(
        f"constraint must be 'kl', 'pathwise' or 'renyi[:alpha]', got {value!r}")


def _log_ratio_terms(log_pc: torch.Tensor, log_pd: torch.Tensor) -> torch.Tensor:
    """l = log p_r - log p_s, with the -inf minus -inf of masked pad rows sent to 0 (they carry no
    probability under log_pc, so they drop out of every logsumexp regardless)."""
    return torch.nan_to_num(log_pd - log_pc, nan=0.0, posinf=0.0, neginf=0.0)


def _log_Z(log_pc: torch.Tensor, l: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(log_pc + u.reshape(-1, 1) * l, dim=-1)


def renyi_divergence(log_pc, log_pd, theta, alpha: float) -> torch.Tensor:
    """D_alpha(p_theta || p_s) per row. Shapes [B,V], [B,V], [B] -> [B]."""
    l = _log_ratio_terms(log_pc, log_pd)
    theta = theta.to(log_pc.dtype).view(-1)
    if abs(alpha - 1.0) < 1e-6:                       # KL: E_theta[theta*l] - log Z(theta)
        logp_theta = log_pc + theta.reshape(-1, 1) * l - _log_Z(log_pc, l, theta).reshape(-1, 1)
        p = logp_theta.exp()
        return (p * (theta.reshape(-1, 1) * l)).sum(-1) - _log_Z(log_pc, l, theta)
    if alpha == float("inf"):                          # max log-ratio
        return theta * l.max(dim=-1).values - _log_Z(log_pc, l, theta)
    a = torch.full_like(theta, float(alpha))
    return (_log_Z(log_pc, l, a * theta) - alpha * _log_Z(log_pc, l, theta)) / (alpha - 1.0)


def solve_theta_renyi(log_pc, log_pd, k_t, alpha: float, iters: int = 40) -> torch.Tensor:
    """Largest theta in [0,1] with D_alpha(p_theta || p_s) <= k_t, per row.

    Mirrors solve_theta_pathwise: D is nondecreasing in theta from D(0)=0, so the feasible set is
    an interval [0, theta*] and a fixed-iteration bisection is exact to 2^-iters.
    """
    B = log_pc.size(0)
    k_t = k_t.to(log_pc.dtype).view(-1)
    ones = torch.ones(B, dtype=log_pc.dtype, device=log_pc.device)
    zeros = torch.zeros_like(ones)
    d_one = renyi_divergence(log_pc, log_pd, ones, alpha)
    lo, hi = zeros.clone(), ones.clone()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        feas = renyi_divergence(log_pc, log_pd, mid, alpha) <= k_t
        lo = torch.where(feas, mid, lo)
        hi = torch.where(feas, hi, mid)
    theta = torch.where(d_one <= k_t, ones, lo)
    return torch.where(k_t <= 0, zeros, theta)
