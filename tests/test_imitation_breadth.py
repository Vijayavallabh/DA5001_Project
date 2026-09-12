"""Proposition 3's measured shape, at every pair it has been measured at.

The proposition is proved and no measurement can refute it. What these pin is the SHAPE the paper
reads off `imitation_cost.csv` -- an imitation rate that saturates in k, a realised spend that
converges on it, a cumulative spend linear in the step index, and a decoder that ends up serving
p_r almost everywhere -- and that the shape is not one pair's. Bands are
`results/onset_prediction_imitation_breadth.md`, committed before either Llama-3.2 pair was
generated, and `results/onset_prediction_imitation_70b.md` for the 70B pair.
"""
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def arms():
    """tag -> {k: row} for the ordinary-prompt rows of every imitation_cost CSV on disk."""
    out = {}
    for path in glob.glob(os.path.join(ROOT, "results", "imitation_cost*.csv")):
        base = os.path.basename(path)
        if "lorenz" in base:
            continue
        tag = base[len("imitation_cost"):-len(".csv")] or "_audited"
        with open(path, encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r["prompt_class"] == "ordinary"]
        if rows:
            out[tag] = {r["k"]: r for r in rows}
    return out


def test_more_than_one_pair_is_measured():
    """The whole arm. If this ever drops back to one, the paper's Proposition 3 paragraph is a
    single setup again and has to say so."""
    a = arms()
    assert "_audited" in a, sorted(a)
    assert len(a) >= 2, sorted(a)


def test_the_rate_saturates_at_every_pair():
    """I1: r_imit(20)/r_imit(3) within 1.10."""
    for tag, ks in arms().items():
        if "3" not in ks or "20" not in ks:
            continue
        r3 = float(ks["3"]["imitation_rate_nats_per_token"])
        r20 = float(ks["20"]["imitation_rate_nats_per_token"])
        assert 0 < r3 <= r20 * 1.001, (tag, r3, r20)   # monotone up to float noise
        assert r20 / r3 <= 1.10, (tag, r20 / r3)


def test_the_spend_is_the_imitation_cost_at_every_pair():
    """I2: |r_real/r_imit - 1| < 0.05 at k=3 and k=20."""
    for tag, ks in arms().items():
        for k in ("3", "20"):
            if k not in ks:
                continue
            imit = float(ks[k]["imitation_rate_nats_per_token"])
            real = float(ks[k]["realised_rate_nats_per_token"])
            assert abs(real / imit - 1) < 0.05, (tag, k, imit, real)


def test_the_spend_is_linear_in_the_step_index_at_every_pair():
    """I3: median within-trajectory R^2 >= 0.95 at every budget."""
    for tag, ks in arms().items():
        for k, r in ks.items():
            assert float(r["median_cum_spend_vs_step_r2"]) >= 0.95, (tag, k, r)


def test_the_decoder_ends_as_the_risky_model_at_every_pair():
    """I4: p_r served unchanged at >= 99% of steps at the largest budget on the grid."""
    for tag, ks in arms().items():
        k = max(ks, key=float)
        r = ks[k]
        unchanged = 1 - float(r["forced_safe_frac"]) - float(r["active_frac"])
        assert unchanged >= 0.99, (tag, k, unchanged)


def test_the_rate_differs_between_pairs_because_the_pairs_differ():
    """I5's point: the rate measures the distance between the two models, so two pairs must not
    give the same number. If they ever do, the reading of imitation_cost.csv as a distance is what
    has to change, not this test."""
    a = arms()
    r3 = {t: float(k["3"]["imitation_rate_nats_per_token"]) for t, k in a.items() if "3" in k}
    assert len(r3) >= 2, r3
    assert max(r3.values()) - min(r3.values()) > 0.05, r3
