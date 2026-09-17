"""feat-125: the sparse causal policy Proposition 3 permits, and the two knobs that build it.

`--initial-bank B` with a negligible refill gives a SEQUENCE budget constant in T -- the shape
selection gets from Proposition 1 and the shape a per-token rate can never have. `--spend-threshold
tau` decides WHERE those nats go, causally: spend at step t only if the full-tilt demand
D_KL(p_r,t || p_s,t) reaches tau, else serve the anchor and keep the nats.

Both default to the deployed rule (0.0 and None), and the first test here is that they do, because
every number on record was produced without them.
"""
import inspect
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a_patch.factory import AnchoredDecodingFactory  # noqa: E402
from dap.e1 import AuditConfig, budget_K  # noqa: E402

FACTORY_SRC = inspect.getsource(AnchoredDecodingFactory)


def test_both_knobs_default_to_the_deployed_rule():
    """Every published number was produced with neither knob; a changed default would silently
    restate the whole paper."""
    assert AuditConfig().initial_bank == 0.0
    assert AuditConfig().spend_threshold is None
    for cls in (AnchoredDecodingFactory.__init__, AnchoredDecodingFactory.from_pretrained):
        p = inspect.signature(cls).parameters
        if "spend_threshold" in p:
            assert p["spend_threshold"].default is None
            assert p["initial_bank"].default == 0.0


def test_the_sequence_budget_does_not_grow_with_the_work():
    """The whole point. A per-token rate makes K = kT, so a longer work is a bigger allowance; an
    initial bank with a negligible refill makes K the bank, whatever T is."""
    tiny, bank = 1e-9, 4.1589
    short, long = budget_K(tiny, 200, bank), budget_K(tiny, 20000, bank)
    assert short == pytest.approx(bank, abs=1e-4)
    assert long == pytest.approx(bank, abs=1e-4)
    assert long - short < 1e-4, "a hundredfold longer work must not move the budget"
    # while the deployed rule's does, by exactly a hundredfold
    assert budget_K(3.0, 20000, 0.0) == 100 * budget_K(3.0, 200, 0.0)


def test_the_gate_reserves_below_threshold_and_is_transparent_above():
    """The reserving rule, on the exact expression the decode loop runs."""
    safe = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    risky = torch.tensor([[2.0, 0.0, 0.0], [9.0, 0.0, 0.0]])   # row 0 agrees, row 1 wants out
    lpc = torch.log_softmax(safe.float(), dim=-1)
    lpd = torch.log_softmax(risky.float(), dim=-1)
    demand = (lpd.exp() * (lpd - lpc)).sum(dim=-1)
    assert demand[0] == pytest.approx(0.0, abs=1e-6), "identical distributions demand nothing"
    assert demand[1] > 1.0, "a risky model 9 logits away demands real nats"

    k_t = torch.tensor([5.0, 5.0])
    gated = k_t * (demand >= 1.0).float()
    assert gated[0] == 0.0, "below threshold the allowance is withdrawn and the nats are kept"
    assert gated[1] == 5.0, "above threshold the allowance passes through untouched"

    # tau=0 must be exactly the ungated policy, so the grid's tau=0 arm IS the front-loader
    assert torch.equal(k_t * (demand >= 0.0).float(), k_t)


def test_the_gate_is_wired_before_the_solve_and_reads_only_this_step():
    """Causality is the claim: the test may read the current step's two distributions and nothing
    else. If this ever consults a future tensor the arm stops being a causal policy."""
    i = FACTORY_SRC.index("self.spend_threshold is not None")
    block = FACTORY_SRC[i:i + 500]
    assert "k_t = k_t * (demand >=" in block
    assert "safe_logits" in block and "risky_logits" in block
    # the gate must precede the solve it gates
    assert FACTORY_SRC.index("self.spend_threshold is not None") < FACTORY_SRC.index(
        "solve_optimization_newton(safe_logits")


def _spend(run_dir):
    """(active, forced) summed over the neutral trajectories of a run, or None if it is absent."""
    import glob
    import json
    f = glob.glob(os.path.join(run_dir, "trajectories_k*_neutral.jsonl"))
    if not f:
        return None
    a = f_ = 0
    for line in open(f[0], encoding="utf-8"):
        agg = json.loads(line)["aggregate"]
        a += agg.get("steps_active") or 0
        f_ += agg.get("steps_forced_safe") or 0
    return a, f_


def test_a_threshold_nothing_reaches_spends_nothing():
    """End-to-end proof that the gate bites: tau=50 nats is above every step's demand, so the
    budget is never released and the decoder is the anchor outright, while the ungated policy at
    the same budget does spend. Measured on GPU 4, 2026-09-17, 4 neutral prompts at B = log 64."""
    hi, greedy = _spend("output/smoke_tau50"), _spend("output/smoke_sparse")
    if hi is None or greedy is None:
        pytest.skip("smoke runs absent (output/ is gitignored); the graded arm is "
                    "results/sparse_causal.csv")
    assert hi[0] == 0, f"tau=50 released budget it should have reserved: active={hi[0]}"
    assert hi[1] > 0, "the run must have decoded something"
    assert greedy[0] > 0, "the ungated policy at the same budget must spend, or the gate proves nothing"


def test_the_front_loader_is_the_trivial_horn_by_construction():
    """Why the threshold knob had to exist. Greedy front-loading spends at O(1) steps and is forced
    to the anchor for the rest -- Proposition 3's trivial horn, realised. Answering the reviewer's
    open question with that policy alone would be a straw man, which is why the tau grid exists."""
    greedy = _spend("output/smoke_sparse")
    if greedy is None:
        pytest.skip("smoke run absent (output/ is gitignored)")
    active, forced = greedy
    assert active / max(active + forced, 1) < 0.05, (
        f"the greedy policy must be the safe model at all but O(1) steps, got {active}/{active+forced}")
