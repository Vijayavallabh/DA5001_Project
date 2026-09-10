"""The proxy-budget claim is a leave-one-out number, and the only way it can be wrong in a way that
flatters it is by leaking the held-out anchor into the constant it is scored against."""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.proxy_budget import loo  # noqa: E402


def test_loo_never_sees_the_held_out_row():
    seen = []
    rows = [dict(s_prot=float(i), s_proxy=1.0) for i in range(5)]

    def predict(train, held):
        seen.append((len(train), held["s_prot"], [t["s_prot"] for t in train]))
        return 0.0

    loo(rows, predict)
    assert len(seen) == len(rows)
    for n_train, held_value, train_values in seen:
        assert n_train == len(rows) - 1
        assert held_value not in train_values


def test_loo_is_exact_on_a_perfectly_proportional_set():
    """If every anchor has the same ratio, the fitted constant is that ratio and the held-out
    prediction is exact -- so any nonzero error here is a bug in the fit, not in the data."""
    rows = [dict(s_prot=2.0 * p, s_proxy=p) for p in (0.5, 0.7, 1.1, 1.9)]
    import statistics as st
    rescale = lambda tr, h: st.mean(x["s_prot"] / x["s_proxy"] for x in tr) * h["s_proxy"]
    assert max(loo(rows, rescale)) < 1e-12


def test_committed_result_reproduces_from_its_own_inputs():
    path, scaling = "results/proxy_budget_summary.csv", "results/anchor_scaling_summary.csv"
    if not (os.path.exists(path) and os.path.exists(scaling)):
        return
    import statistics as st
    rows = [dict(s_prot=float(r["s_passage"]), s_proxy=float(r["s_gutenberg_span"]))
            for r in csv.DictReader(open(scaling))]
    rescale = lambda tr, h: st.mean(x["s_prot"] / x["s_proxy"] for x in tr) * h["s_proxy"]
    constant = lambda tr, h: st.mean(x["s_prot"] for x in tr)
    s = next(iter(csv.DictReader(open(path))))
    assert abs(st.mean(loo(rows, rescale)) - float(s["loo_proxy"])) < 1e-3
    assert abs(st.mean(loo(rows, constant)) - float(s["loo_constant"])) < 1e-3
    # the headline: the proxy beats the constant, and by the factor claimed
    assert float(s["loo_proxy"]) < float(s["loo_constant"])
    assert abs(float(s["improvement_over_constant"])
               - float(s["loo_constant"]) / float(s["loo_proxy"])) < 0.01
