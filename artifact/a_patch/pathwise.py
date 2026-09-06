"""Pathwise (max-divergence) constraint for Anchored Decoding (feat-019, plan v2 C8, Proposition 1).

The KL decoder chooses the largest theta with D_KL(p_theta || p_s) <= k_t. The pathwise decoder chooses the
largest theta with  m(theta) = max_v log p_theta(v)/p_s(v) = theta * max_v l(v) - log Z(theta) <= k_t,
where l = log p_r - log p_s and p_theta ∝ p_s^(1-theta) p_r^theta. m is nondecreasing in theta
(dm/dtheta = max l - E_theta[l] >= 0) with m(0) = 0, so a bisection finds the boundary. With the allowance
k_t = max(0, (t+1)k - delta_init - R_{t-1}) charged against the realised cumulative log-ratio R of the
sampled tokens, every full sequence satisfies log p_theta(y|x) - log p_s(y|x) <= K, which is the Delta_max
form of near-access-freeness (per-event bound p_theta(E|x) <= e^K p_s(E|x)).
"""
from __future__ import annotations

import torch


def _log_ratio_terms(log_pc: torch.Tensor, log_pd: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(log_pd - log_pc, nan=0.0, posinf=0.0, neginf=0.0)


def max_log_ratio(log_pc: torch.Tensor, log_pd: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """m(theta) per row: the largest log p_theta(v)/p_s(v) over the vocabulary. Shapes: [B,V], [B,V], [B] -> [B]."""
    l = _log_ratio_terms(log_pc, log_pd)
    theta = theta.to(log_pc.dtype).reshape(-1, 1)
    logz = torch.logsumexp(log_pc + theta * l, dim=-1)
    return (theta.squeeze(1) * l.max(dim=-1).values) - logz


def solve_theta_pathwise(log_pc: torch.Tensor, log_pd: torch.Tensor, k_t: torch.Tensor, iters: int = 40) -> torch.Tensor:
    """Largest theta in [0,1] with max_log_ratio(theta) <= k_t, per row. k_t <= 0 gives 0; m(1) <= k_t gives 1."""
    B = log_pc.size(0)
    k_t = k_t.to(log_pc.dtype).view(-1)
    ones = torch.ones(B, dtype=log_pc.dtype, device=log_pc.device)
    zeros = torch.zeros_like(ones)
    m_one = max_log_ratio(log_pc, log_pd, ones)
    lo, hi = zeros.clone(), ones.clone()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        feas = max_log_ratio(log_pc, log_pd, mid) <= k_t
        lo = torch.where(feas, mid, lo)
        hi = torch.where(feas, hi, mid)
    theta = lo
    theta = torch.where(m_one <= k_t, ones, theta)
    theta = torch.where(k_t <= 0, zeros, theta)
    return theta
