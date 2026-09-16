"""The four Renyi orders must share one baseline, because an unconstrained decode has no order.

output/phase4/renyi_renyi_{1_0,2,4,8} swept k in {1,3,5} with no k=-1 and no k=0 arm. The arms that
filled that gap are also a check on the decoder: a_patch/factory.py handles k_radius == 0.0 and
== -1.0 in branches that never read self.constraint, so all four orders must reproduce
output/phase4/fine_tc_base -- same pair, same 100 passages, same seed, differing only in constraint=.

Exact equality, not a tolerance. Same models, same passages, same seed: any difference is a
difference in the code path. The arms are SAMPLED, so a constraint object consuming the RNG
differently would move the draw without touching the mixing weights -- that failure would appear
here as a discrepancy concentrated in oracle, which re-samples per window.

If this ever fails, Table 3's four orders can no longer be read against a shared baseline and the
prediction in results/onset_prediction_renyi_baselines.md says to investigate rather than patch.
"""
import csv
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results/renyi_baselines.csv")
ORDERS = {"renyi:1.0", "renyi:2", "renyi:4", "renyi:8"}


@pytest.fixture(scope="module")
def rows():
    assert os.path.exists(CSV), "run analysis/renyi_baselines.py"
    with open(CSV) as fh:
        return list(csv.DictReader(fh))


def test_all_four_orders_are_present_with_both_budgets_and_both_modes(rows):
    assert {r["order"] for r in rows} == ORDERS
    for order in ORDERS:
        arms = {(r["mode"], float(r["k"])) for r in rows if r["order"] == order}
        assert arms == {("single", -1.0), ("single", 0.0), ("oracle", -1.0), ("oracle", 0.0)}, \
            f"{order} is missing a baseline arm: {sorted(arms)}"


def test_every_order_reproduces_the_reference_exactly(rows):
    bad = [(r["order"], r["mode"], r["k"], r["nv_recall"], r["reference_nv_recall"])
           for r in rows if r["identical_to_reference"] != "yes"]
    assert not bad, (
        "the Renyi order reached a decode that is supposed to be unconstrained: " + repr(bad) +
        " -- Table 3's orders cannot be read against a shared baseline until this is understood")


def test_the_anchor_alone_reproduces_nothing_and_the_memoriser_a_lot(rows):
    """A baseline that agreed with the reference but was degenerate would pass the test above."""
    for r in rows:
        nv = float(r["nv_recall"])
        if float(r["k"]) == 0.0:
            assert nv <= 0.01, (r["order"], r["mode"], "k=0 must reproduce ~nothing", nv)
        else:
            assert nv >= 0.10, (r["order"], r["mode"], "k=-1 must clear the entry gate", nv)
        assert int(r["violations"]) == 0, (r["order"], r["mode"], "invariant violation")
