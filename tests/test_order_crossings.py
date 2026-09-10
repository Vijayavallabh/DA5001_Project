"""The crossing test decides whether a higher order dominates the KL decoder or merely swaps places
with it somewhere in the operating range, and the whole verdict turns on the noise floor. A floor of
zero would call every numerical wobble a crossing; a floor larger than the signal would call
everything indistinguishable. Both directions are pinned here."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.order_crossings import noise_floor  # noqa: E402
from analysis.order_frontier import _at  # noqa: E402


def _verdict(d, floor):
    safer, worse = any(x < -floor for x in d), any(x > floor for x in d)
    return ("crosses" if safer and worse else "uniformly safer" if safer
            else "uniformly more dangerous" if worse else "indistinguishable")


def test_the_floor_decides_the_verdict_in_both_directions():
    d = [-3.0, -1.0, 0.5, 2.0]
    assert _verdict(d, 0.1) == "crosses"          # both signs clear a small floor
    assert _verdict(d, 2.5) == "uniformly safer"  # only the negative side clears a larger one
    assert _verdict(d, 5.0) == "indistinguishable"
    assert _verdict([2.0, 3.0], 1.0) == "uniformly more dangerous"


def test_committed_verdicts_follow_from_their_own_columns():
    import csv
    path = "results/order_crossings.csv"
    if not os.path.exists(path):
        return
    for r in csv.DictReader(open(path)):
        f = float(r["noise_floor"])
        safer = float(r["min_nats_per_window"]) < -f
        worse = float(r["max_nats_per_window"]) > f
        want = ("crosses" if safer and worse else "uniformly safer" if safer
                else "uniformly more dangerous" if worse else "indistinguishable")
        assert r["verdict"] == want, r
        assert float(r["frac_safer"]) + float(r["frac_worse"]) <= 1.0 + 1e-9, r
        assert float(r["fidelity_lo"]) < float(r["fidelity_hi"]), r


def test_a_pair_with_both_precisions_gets_its_own_floor_not_a_borrowed_one():
    if not os.path.exists("results/order_frontier_kl3m_bf16_matched.csv"):
        return
    own = noise_floor("kl3m", 50.0)
    assert own is not None and 0.0 < own < 5.0, own
    assert noise_floor("comma", 50.0) is None      # no float32 twin, so it must borrow
