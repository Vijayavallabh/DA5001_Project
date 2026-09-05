"""feat-019: the pathwise (max-divergence) constraint. The solver must return the largest theta whose
max log-ratio m(theta) = max_v log p_theta(v)/p_s(v) is within the allowance; m is nondecreasing in theta;
the KL spend never exceeds the max-ratio; and on saved pathwise logs the realised cumulative log-ratio
R_T stays within max(0, B) + 1e-3 on every trajectory."""
import json
import math
from pathlib import Path

import pytest
import torch

from a_patch.pathwise import max_log_ratio, solve_theta_pathwise

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-3


def _random_logits(batch=6, vocab=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    log_pc = torch.log_softmax(torch.randn(batch, vocab, generator=g, dtype=torch.float64) * 2.0, dim=-1)
    log_pd = torch.log_softmax(torch.randn(batch, vocab, generator=g, dtype=torch.float64) * 3.0, dim=-1)
    return log_pc, log_pd


def test_max_log_ratio_is_nondecreasing_in_theta():
    log_pc, log_pd = _random_logits()
    thetas = torch.linspace(0.0, 1.0, 21)
    prev = None
    for th in thetas:
        m = max_log_ratio(log_pc, log_pd, th.expand(log_pc.size(0)))
        if prev is not None:
            assert torch.all(m >= prev - 1e-6), (th.item(), m, prev)
        prev = m
    zeros = torch.zeros(log_pc.size(0), dtype=torch.float64)
    assert torch.allclose(max_log_ratio(log_pc, log_pd, zeros), zeros, atol=1e-6)


def test_solver_returns_largest_feasible_theta():
    log_pc, log_pd = _random_logits(seed=1)
    k_t = torch.tensor([0.0, 0.05, 0.3, 1.0, 3.0, 100.0], dtype=torch.float64)
    theta = solve_theta_pathwise(log_pc, log_pd, k_t)
    m = max_log_ratio(log_pc, log_pd, theta)
    assert torch.all(m <= k_t + 1e-5), (m, k_t)
    assert theta[0].item() == 0.0  # no allowance: serve the anchor
    assert theta[-1].item() == 1.0  # huge allowance: serve the risky model unchanged
    bumped = (theta + 0.02).clamp(max=1.0)
    m_bumped = max_log_ratio(log_pc, log_pd, bumped)
    interior = (theta > 0) & (theta < 1)
    assert torch.all(m_bumped[interior] > k_t[interior]), "theta was not maximal"


def test_kl_spend_never_exceeds_max_log_ratio():
    log_pc, log_pd = _random_logits(seed=2)
    k_t = torch.full((log_pc.size(0),), 0.4, dtype=torch.float64)
    theta = solve_theta_pathwise(log_pc, log_pd, k_t)
    log_p = torch.log_softmax(log_pc + theta[:, None] * (log_pd - log_pc), dim=-1)
    kl = (log_p.exp() * (log_p - log_pc)).sum(dim=-1)
    m = max_log_ratio(log_pc, log_pd, theta)
    assert torch.all(kl <= m + 1e-6)


def test_realised_ratio_invariant_on_saved_pathwise_logs():
    path = ROOT / "tests" / "data" / "sample_pathwise_trajectories.jsonl"
    if not path.exists():
        pytest.skip("pathwise smoke logs not saved yet")
    n = 0
    for line in open(path):
        r = json.loads(line)
        R, B = r["aggregate"]["total_realised_ratio"], r["aggregate"]["final_budget"]
        assert R <= max(0.0, B) + EPS, r["metadata"]
        for s in r["per_step_log"]:
            assert s["r_t"] <= s["k_t"] + EPS, (r["metadata"], s["t"])
            assert s["a_t"] <= s["m_t"] + EPS
        n += 1
    assert n >= 4
