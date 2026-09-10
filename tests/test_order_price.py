"""The corrected leakage column is a log-probability on the geodesic, not an average. It is only
interpretable if it really interpolates between the two models the decoder sits between, so that is
what is asserted: theta = 0 must reproduce the anchor's own log-probability of the tokens and
theta = 1 the risky model's. Those are the brackets the run prints, and if they do not hold the
number in the middle is not a constrained decoder's reproduction probability."""
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.order_price import logp_target  # noqa: E402


def _pair(seed=0, T=7, V=40):
    g = torch.Generator().manual_seed(seed)
    log_ps = torch.log_softmax(torch.randn(T, V, generator=g).double(), dim=-1)
    log_pr = torch.log_softmax(torch.randn(T, V, generator=g).double() * 2.0, dim=-1)
    tgt = torch.randint(0, V, (T,), generator=g)
    return log_ps, log_pr - log_ps, log_pr, tgt


def test_endpoints_are_the_two_models_themselves():
    log_ps, l, log_pr, tgt = _pair()
    z = torch.zeros(log_ps.size(0)).double()
    assert abs(logp_target(log_ps, l, z, tgt)
               - float(log_ps.gather(1, tgt.reshape(-1, 1)).sum())) < 1e-9
    assert abs(logp_target(log_ps, l, z + 1.0, tgt)
               - float(log_pr.gather(1, tgt.reshape(-1, 1)).sum())) < 1e-9


def test_it_is_a_normalised_log_probability_not_a_tilted_score():
    """The tilt has to be renormalised; forgetting the logsumexp would leave a quantity that can
    exceed zero and is not a log-probability at all."""
    log_ps, l, _, tgt = _pair(1)
    for th in (0.0, 0.3, 0.7, 1.0):
        v = logp_target(log_ps, l, torch.full((log_ps.size(0),), th).double(), tgt)
        assert v < 0.0


def test_committed_row_sits_inside_its_own_bracket():
    path = "results/order_price_kl3m_k3.csv"
    if not os.path.exists(path):
        return
    rows = list(csv.DictReader(open(path)))
    if "logp_target" not in rows[0]:
        return
    hi, lo = float(rows[0]["logp_target_risky"]), float(rows[0]["logp_target_safe"])
    assert lo < hi, "the risky model must find its own memorised tokens likelier than the anchor"
    prev = None
    for r in rows:
        v = float(r["logp_target"])
        assert lo <= v <= hi, r          # a constrained decoder, not one of the endpoints
        if prev is not None:
            assert v <= prev + 1e-6, r   # a higher order buys a smaller tilt, so a smaller logp
        prev = v


def test_the_risky_models_own_log_probability_is_NOT_an_upper_bound():
    """L(theta) is not monotone in theta: mixing the anchor in helps wherever the anchor is right and
    the risky model is wrong, so a partial tilt can give the true tokens more mass than theta = 1
    does. The run's bracket therefore gates on the LOWER side only and reports the upper excursion as
    a diagnostic. This counter-example exists so nobody re-asserts the bound."""
    log_ps = torch.log(torch.tensor([[0.10, 0.80, 0.10], [0.80, 0.10, 0.10]])).double()
    log_pr = torch.log(torch.tensor([[0.90, 0.05, 0.05], [0.05, 0.90, 0.05]])).double()
    l = log_pr - log_ps
    tgt = torch.tensor([0, 0])
    at_one = logp_target(log_ps, l, torch.ones(2).double(), tgt)
    partial = logp_target(log_ps, l, torch.full((2,), 0.5).double(), tgt)
    assert partial > at_one + 1.0, (partial, at_one)
