"""Selection anchoring is scored by one number per arm, and two things can silently invert it: the
judge's flip (which side the candidate was shown on) and the nesting of the arms. Pin both, plus the
KL identity the whole claim rests on."""
import csv
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_decoding import VERDICT_U, kl_best_of_n  # noqa: E402
from analysis.utility import arm_won  # noqa: E402


def test_the_kl_identity_the_claim_rests_on():
    """log n - (n-1)/n, Beirami et al. n = 1 must be exactly 0: the control arm is the anchor and
    spends nothing, and the whole comparison is against the decoder's 165 nats."""
    assert kl_best_of_n(1) == 0.0
    assert abs(kl_best_of_n(8) - (math.log(8) - 7 / 8)) < 1e-12
    assert abs(kl_best_of_n(8) - 1.2044) < 1e-4
    assert kl_best_of_n(64) < math.log(64)
    for n in (2, 4, 8, 16, 64):
        assert kl_best_of_n(n) < kl_best_of_n(2 * n)     # monotone, so bigger n is never cheaper


def test_the_flip_convention_agrees_with_the_utility_pipeline():
    """selection_decoding derives `won` inline; utility.py has arm_won. If they ever disagree the
    win rates are inverted against the arms already in the paper, which is unrecoverable by eye."""
    for verdict in ("A", "B", "Tie"):
        for flip in (False, True):
            won = (verdict == "B") if flip else (verdict == "A")
            assert won == arm_won(verdict, flip), (verdict, flip)


def test_the_utility_scale_is_the_one_utility_price_uses():
    assert VERDICT_U == {"win": 1.0, "tie": 0.5, "loss": 0.0}


def _rows(tmp_path, spec):
    """spec: [(prompt, rank, logp_per_token, outcome)] -> a candidate CSV the arm pass can read."""
    p = tmp_path / "cand.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["prompt_id", "rank", "seed", "prompt_class", "n_tokens", "logp_total",
                    "logp_per_token", "shown_second", "verdict", "outcome"])
        for pid, rank, lp, out in spec:
            w.writerow([pid, rank, rank, "neutral", 10, lp * 10, lp, 0,
                        {"win": "A", "loss": "B", "tie": "Tie"}[out], out])
    return p


def test_arms_nest_so_a_bigger_n_can_only_see_more_candidates(tmp_path):
    """Arm n takes the FIRST n candidates in rank order. If it sampled instead, n=8 and n=4 would
    not be comparable and the KL axis would mean nothing. Candidate 0 is the worst-scoring winner,
    candidate 1 the best-scoring loser: n=1 must score 1.0 and n=2 must score 0.0."""
    src = _rows(tmp_path, [("p1", 0, -2.0, "win"), ("p1", 1, -1.0, "loss"),
                           ("p2", 0, -2.0, "win"), ("p2", 1, -1.0, "loss")])
    out = subprocess.run([sys.executable, "analysis/selection_decoding.py", "--candidates", str(src),
                          "--n-values", "1", "2", "--out", str(tmp_path)],
                         capture_output=True, text=True, check=True).stdout
    got = {(r["rule"], int(r["n"])): float(r["u"])
           for r in csv.DictReader(open(tmp_path / "selection_decoding.csv"))}
    assert got[("per-token mean (primary)", 1)] == 1.0
    assert got[("per-token mean (primary)", 2)] == 0.0, out
    # the oracle can only do better, never worse, at the same n
    for n in (1, 2):
        assert got[("oracle: the judge itself", n)] >= got[("per-token mean (primary)", n)]
    assert got[("oracle: the judge itself", 2)] == 1.0


def test_the_cross_judge_arm_is_measured_inside_one_judge():
    """The gain the appendix quotes is judge B's n=8 minus judge B's own n=1, on the same prompts.
    Quoting judge B's n=8 against judge A's anchor would be the easy mistake and would inflate it."""
    import csv as _csv
    rows = list(_csv.DictReader(open("results/selection_crossjudge.csv")))
    sel = [r for r in rows if r["n"] in ("1", "8")]
    assert len({r["judge"] for r in sel}) == 1
    a1 = next(r for r in sel if r["n"] == "1")
    a8 = next(r for r in sel if r["n"] == "8")
    assert abs((float(a8["u"]) - float(a1["u"])) - float(a8["gain"])) < 1e-9
    lo, hi = float(a8["gain_lo95"]), float(a8["gain_hi95"])
    assert lo < float(a8["gain"]) < hi and lo > 0, (lo, hi)
    assert float(a8["kl_nats"]) == round(kl_best_of_n(8), 4)


def test_the_frontier_multiples_are_both_priced_under_the_same_law():
    """58x against 7,995x is a ratio of two ratios; it is only meaningful if both use judge B's own
    law of U. The decoder row must carry the same judge and a much larger multiple."""
    import csv as _csv
    rows = list(_csv.DictReader(open("results/selection_crossjudge.csv")))
    sel = next(r for r in rows if r["n"] == "8")
    dec = next(r for r in rows if r["selector"].startswith("metered decoder"))
    assert dec["judge"] == sel["judge"]
    assert float(dec["nats_over_frontier"]) > 100 * float(sel["nats_over_frontier"])
    assert float(dec["kl_nats"]) > 100 * float(sel["kl_nats"])
