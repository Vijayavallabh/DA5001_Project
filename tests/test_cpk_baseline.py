"""feat-211: CP-k's rejection rule with M tries and an anchor fallback is certified at log(1 + M e^kappa).

The served law is computed EXACTLY on small discrete laws and the pathwise ratio checked against the
certificate the arm is registered at, so the claim in results/onset_prediction_cpk_baseline.md is tested
on the mechanism, not on the arithmetic of its bound.
"""
import itertools
import math
import random

from analysis.cpk_baseline import certificate, kappa_for
from analysis.cpk_extraction import serve


def served_law(pr, ps, kappa, m):
    """q(y) for: try up to m draws from pr, serve the first with log pr/ps <= kappa, else one ps draw."""
    ok = [math.log(a / b) <= kappa for a, b in zip(pr, ps)]
    acc = sum(a for a, o in zip(pr, ok) if o)
    reach = sum((1 - acc) ** j for j in range(m))          # expected number of tries that can still serve
    return [a * o * reach + (1 - acc) ** m * b for a, b, o in zip(pr, ps, ok)]


def test_kappa_inverts_the_certificate():
    for c in (4.1589, 33.271, 83.178, 159.83, 400.0, 800.0, 5000.0):   # past 709, e^c overflows a double
        for m in (1, 8, 64):
            assert abs(certificate(kappa_for(c, m), m) - c) < 1e-9, (c, m)
            if c < 700:
                assert abs(math.log1p(m * math.exp(kappa_for(c, m))) - c) < 1e-9, (c, m)
    # at selection's own log 64 with 64 tries the rule accepts only draws the anchor likes at least as much
    assert kappa_for(math.log(64), 64) < 0


def test_the_served_law_is_a_law_and_is_certified_pathwise():
    rng = random.Random(211)
    for _ in range(300):
        k = rng.randint(2, 7)
        pr = [rng.random() ** 3 for _ in range(k)]
        ps = [rng.random() for _ in range(k)]
        pr = [x / sum(pr) for x in pr]
        ps = [x / sum(ps) for x in ps]
        for m in (1, 3, 64):
            for kappa in (-2.0, -0.01, 0.0, 0.7, 2.5, 10.0):
                q = served_law(pr, ps, kappa, m)
                assert abs(sum(q) - 1) < 1e-9
                c = math.log1p(m * math.exp(kappa))
                assert max(math.log(a / b) for a, b in zip(q, ps)) <= c + 1e-9, (pr, ps, kappa, m)


def test_the_bound_is_attained_as_acceptance_vanishes():
    """Tightness: the certificate is not loose by a constant. With kappa < 0 (selection's own log 64 at
    M = 64 is kappa = log(63/64)) a single outcome can pass at ratio exactly e^kappa while every other
    outcome fails, so acceptance -> 0 and q(y)/ps(y) -> 1 + M e^kappa. (With kappa >= 0 some outcome
    always passes -- sum pr = sum ps forces one with pr <= ps -- so acceptance cannot vanish there and
    the bound, still valid, is not attained.)"""
    m, kappa = 64, -1.0
    for eps in (1e-4, 1e-8):
        p0 = eps * math.exp(kappa) * (1 - 1e-9)                # strictly inside the test, whatever the rounding
        pr = [p0, 1 - p0]
        ps = [eps, 1 - eps]
        assert math.log(pr[0] / ps[0]) <= kappa < math.log(pr[1] / ps[1])                 # the other outcome fails the test
        q = served_law(pr, ps, kappa, m)
        # the relative gap is about M^2 e^kappa eps / (2 (1 + M e^kappa)) < 2 M eps: 0.12% at eps = 1e-4
        tol = 2 * m * eps
        assert 0 < tol < 0.02 and q[0] / ps[0] > (1 - tol) * (1 + m * math.exp(kappa)), (eps, q[0] / ps[0])


def test_serve_takes_the_first_passing_draw_in_order():
    assert serve([5.0, 1.0, 0.5], 1.0) == 1
    assert serve([5.0, 3.0], 1.0) is None
    assert serve([], 0.0) is None
    # draw order, not the smallest ratio: the rule never looks past the first pass
    for rs in itertools.permutations([0.2, 0.9, 3.0]):
        j = serve(list(rs), 1.0)
        assert all(r > 1.0 for r in rs[:j]) and rs[j] <= 1.0


def test_r_runs_through_the_harness_end_not_an_eot_id():
    """A plain run stops only at <|end_of_text|>; an <|eot_id|> mid-generation is served text, and R must count it."""
    from analysis.cpk_baseline import served_steps
    import math
    steps = [dict(p_s_prob=0.5, p_risky_prob=0.5, sampled_token_id=11),
             dict(p_s_prob=0.1, p_risky_prob=0.9, sampled_token_id=128009),     # eot_id: the model writes on
             dict(p_s_prob=0.01, p_risky_prob=0.5, sampled_token_id=12),
             dict(p_s_prob=0.2, p_risky_prob=0.4, sampled_token_id=128001),     # the harness's end
             dict(p_s_prob=1.0, p_risky_prob=1.0, sampled_token_id=128001)]     # padding past the end
    rec = dict(metadata=dict(chat_template=False), per_step_log=steps)
    got = served_steps(rec)
    assert len(got) == 4
    assert abs(sum(got) - (math.log(9) + math.log(50) + math.log(2))) < 1e-9
