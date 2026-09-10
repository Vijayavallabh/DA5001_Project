"""The seed arms exist to answer one question -- is the advantage the pair's or the evaluation's --
and the way that goes wrong is by comparing an arm against the wrong control, or by silently
treating a missing control as agreement. Both are pinned here."""
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_arm_filenames_parse_to_a_pair_and_a_seed():
    pat = re.compile(r"order_seedarm_(.+)_s(\d+)_bf16_matched\.csv$")
    m = pat.match("order_seedarm_kl3m_s10_bf16_matched.csv")
    assert m and m.group(1) == "kl3m" and int(m.group(2)) == 10
    assert pat.match("order_frontier_kl3m_bf16_matched.csv") is None   # a control is not an arm


def test_seed_and_corpus_arms_never_enter_the_pair_set():
    """order_law and order_predictors skip any _seed or _gut_ file. A seed arm is one pair re-run at
    another --seed-tokens and a Gutenberg arm is one anchor re-run on a second protected corpus;
    either would put an anchor into the rank test twice and make the p-values wrong. If a guard is
    ever removed this fails, which is the point."""
    for path in ("analysis/order_law.py", "analysis/order_predictors.py"):
        src = open(path).read()
        assert '"_seed" in os.path.basename(path)' in src, path
        assert '"_gut_" in os.path.basename(path)' in src, path


def test_committed_seed_table_compares_like_with_like():
    path = "results/order_seed.csv"
    if not os.path.exists(path):
        return
    for r in csv.DictReader(open(path)):
        assert int(r["seed"]) != int(r["control_seed"]), r
        assert float(r["alpha"]) > 1.0, r          # alpha = 1 is the baseline both sides share
        assert abs((float(r["arm"]) - float(r["control"])) - float(r["diff"])) < 2e-3, r
        if r["noise_floor"]:
            want = abs(float(r["diff"])) > float(r["noise_floor"])
            assert (r["beyond_floor"] == "True") == want, r
