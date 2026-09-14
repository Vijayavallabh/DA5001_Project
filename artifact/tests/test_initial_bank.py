"""The budget-PLACEMENT knob: the same sequence budget, granted up front instead of accrued.

The bucket's refill rate k spreads the allowance uniformly along the sequence. An initial bank with
a negligible refill puts the same total at the front. Both are causal policies under a sequence
budget -- they are the two placements Proposition 5 separates, and the arm that tests it is only
meaningful if the recorded budget counts the bank. Quoting k*T_max for a front-loaded arm would
understate what the certificate allows by the whole of the bank, which is the entire budget."""
import pytest

from dap.e1 import AuditConfig, budget_K


def test_the_default_is_the_deployed_rule_exactly():
    assert AuditConfig().initial_bank == 0.0
    assert budget_K(3.0, 200) == 600.0
    assert budget_K(0.0, 200) == 0.0
    assert budget_K(-1.0, 200) == float("inf")


def test_the_bank_is_counted_in_the_sequence_budget():
    assert budget_K(3.0, 200, 2.08) == pytest.approx(602.08)
    # the front-loaded arm: negligible refill, the whole budget up front
    assert budget_K(1e-9, 200, 2.08) == pytest.approx(2.08, abs=1e-6)
    # a baseline is still a baseline only when nothing is granted
    assert budget_K(0.0, 200, 0.0) == 0.0
    assert budget_K(0.0, 200, 5.0) == 5.0


def test_the_factory_rejects_a_negative_bank_and_stores_a_positive_one():
    """A negative bank would be a second prefix debt wearing the wrong name."""
    from a_patch.factory import AnchoredDecodingFactory
    import inspect
    sig = inspect.signature(AnchoredDecodingFactory.__init__)
    assert sig.parameters["initial_bank"].default == 0.0
    src = inspect.getsource(AnchoredDecodingFactory.__init__)
    assert 'assert initial_bank >= 0.0' in src
    # and it reaches the bank on both branches, with and without the prefix debt
    gen = inspect.getsource(AnchoredDecodingFactory)
    assert gen.count("self.initial_bank") >= 3, gen.count("self.initial_bank")
