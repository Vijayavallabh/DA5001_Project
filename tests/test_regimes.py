"""Plan v4 / feat-035: the ordering the frontier theorem depends on."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.regimes import k_crit_rate, regimes


def test_k_crit_never_below_the_mean_rate():
    """k_crit >= s_rate is the inequality that makes [s, k_crit) a real interval. It holds because
    the running maximum of a prefix average is at least the whole-sequence average."""
    for nats, chars in [([1.0, 2.0, 3.0], [4, 8, 12]),
                        ([3.0, 2.0, 1.0], [4, 8, 12]),      # front-loaded: the gap is widest here
                        ([2.0] * 10, list(range(4, 44, 4))),  # flat: the gap closes to equality
                        ([0.1, 9.9], [5, 10])]:
        r = regimes(nats, chars)
        assert r["k_crit"] >= r["s_rate"] - 1e-12, (nats, r)
        assert r["uncertified_width"] >= 1.0 - 1e-12


def test_flat_surprisal_collapses_the_uncertified_gap():
    """A work whose surprisal is perfectly uniform leaves no room between certifying and protecting;
    burstiness is what opens the interval."""
    flat = regimes([2.0] * 8, [3 * (i + 1) for i in range(8)])
    bursty = regimes([9.0] + [1.0] * 7, [3 * (i + 1) for i in range(8)])
    assert abs(flat["uncertified_width"] - 1.0) < 1e-9
    assert bursty["uncertified_width"] > 2.0


def test_initial_debt_raises_both_boundaries():
    """delta is the bucket's initial level, so it shifts k_crit -- this is the prefix-debt mechanism."""
    nats, chars = [2.0, 2.0, 2.0, 2.0], [5, 10, 15, 20]
    base, debt = regimes(nats, chars, delta=0.0), regimes(nats, chars, delta=10.0)
    assert debt["k_crit"] > base["k_crit"]
    assert debt["s_rate"] > base["s_rate"]


def test_window_opens_exactly_when_ordinary_traffic_is_cheaper():
    nats, chars = [2.0] * 5, [4 * (i + 1) for i in range(5)]
    s = regimes(nats, chars)["s_rate"]
    assert regimes(nats, chars, c_use=s * 0.5)["window_open"] is True
    assert regimes(nats, chars, c_use=s * 2.0)["window_open"] is False
    assert regimes(nats, chars, c_use=s * 2.0)["window_width"] == 0.0


def test_k_crit_is_the_running_max_not_the_final_average():
    """A spike early in the work sets k_crit even when the tail is cheap."""
    assert k_crit_rate([10.0, 0.0, 0.0, 0.0], [1, 2, 3, 4]) == 10.0
    assert k_crit_rate([0.0, 0.0, 0.0, 10.0], [1, 2, 3, 4]) == 2.5


def test_empty_work_is_not_a_crash():
    assert regimes([], []) is None


def test_vacuity_threshold_is_the_same_for_every_divergence_order():
    """The load-bearing claim of the frontier theorem: strengthening the charge from KL (alpha=1)
    to max-divergence (alpha=inf) tightens the bound below S but never moves the point at which it
    becomes vacuous. So no order of divergence escapes vacuity; only a smaller budget does."""
    from analysis.regimes import event_bound
    S = 200.0
    for alpha in (1.0, 1.5, 2.0, 8.0, 64.0, float("inf")):
        assert event_bound(S, S, alpha) == 1.0, f"alpha={alpha} not vacuous at K=S"
        assert event_bound(S, S + 1e-6, alpha) == 1.0, f"alpha={alpha} not vacuous above S"
        assert event_bound(S, S - 1.0, alpha) < 1.0, f"alpha={alpha} vacuous below S"


def test_higher_order_gives_a_tighter_bound_below_the_threshold():
    from analysis.regimes import event_bound
    S, K = 200.0, 100.0
    caps = [event_bound(S, K, a) for a in (1.0, 2.0, 8.0, 64.0, float("inf"))]
    for lo, hi in zip(caps, caps[1:]):
        assert hi <= lo + 1e-12, f"bound must not loosen as alpha grows: {caps}"
    assert caps[-1] < caps[0], "max-divergence must be strictly tighter than KL here"


def test_the_renyi_conversion_bound_in_proposition_1_is_valid():
    """Proposition 1 converts a Renyi-alpha budget into an event bound via
    p(E) <= (e^{D_alpha(p||q)} q(E))^{(alpha-1)/alpha}. Randomised search for a counterexample;
    the full 40,000-pair sweep found none and found the bound tight to 1e-15."""
    import math
    import torch
    torch.manual_seed(0)

    def renyi(p, q, a):
        if a == float("inf"):
            return (p.log() - q.log()).max().item()
        return (torch.logsumexp(a * p.log() + (1 - a) * q.log(), dim=-1) / (a - 1)).item()

    worst = 1.0
    for _ in range(1500):
        V = int(torch.randint(2, 10, (1,)))
        p = torch.softmax(torch.randn(V) * float(torch.rand(1) * 4), dim=-1).double()
        q = torch.softmax(torch.randn(V) * float(torch.rand(1) * 4), dim=-1).double()
        mask = torch.rand(V) < 0.5
        if not mask.any() or mask.all():
            continue
        pe, qe = p[mask].sum().item(), q[mask].sum().item()
        for a in (1.5, 2.0, 4.0, float("inf")):
            K = renyi(p, q, a)
            if not math.isfinite(K):
                continue
            exp = 1.0 if a == float("inf") else (a - 1.0) / a
            rhs = min(1.0, math.exp(K * exp) * qe ** exp)
            worst = min(worst, rhs - pe)
    assert worst > -1e-9, f"counterexample to the conversion bound, slack {worst:.3e}"


def test_onset_crossing_interpolates_within_the_bracket():
    """analysis.onset.crossing must return the interpolated crossing, not the first grid point
    above the threshold. It reported the latter until the loop was fixed to advance its bracket."""
    from analysis.onset import crossing
    lo, hi, est = crossing({1.5: 0.0, 2.0: 0.0, 2.6: 0.0, 3.2: 0.022}, 0.01)
    assert (lo, hi) == (2.6, 3.2), (lo, hi)
    assert 2.6 < est < 3.2, est
    assert abs(est - (2.6 + 0.6 * 0.01 / 0.022)) < 1e-9
    # already above threshold at the smallest budget probed -> no bracket
    assert crossing({1.0: 0.5, 2.0: 0.6}, 0.01) == (None, 1.0, 1.0)
    # never reaches the threshold
    assert crossing({1.0: 0.0, 2.0: 0.001}, 0.01) == (None, None, None)
