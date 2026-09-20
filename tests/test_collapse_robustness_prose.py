"""Four numbers in Appendix C are prose, not a table, and prose is where stale values survive:
`analysis/audit_numbers.py` only asks whether a literal appears in SOME CSV, and 0.165 and 0.173
did -- they had been the metric-block values two pair-counts earlier and stayed in the manuscript
after the CSV moved to 0.106 and 0.109. Pin them against the block they claim to come from."""
import csv
import os
import re

from tests.manuscript import tex

CSV = "results/collapse_robustness.csv"
TEX = tex("sections/appendix_robustness.tex")


def _blocks():
    out = {}
    for r in csv.DictReader(open(CSV)):
        out[(r["block"], r["setting"])] = r
    return out


def test_the_top_of_the_utility_scale_is_priced_from_the_same_law_as_the_rest():
    """Section 3 prices winning every judged comparison. It read 1.19 nats and 144x, which come
    from results/utility_v4_summary.csv (180 judged pairs per arm, anchor win 30.5%); every other
    number in the same paragraph comes from v5 (600 pairs, 27.3%), where the same quantity is 1.30
    and 132x. AGENTS.md caution (d) is about exactly that 180-against-600 difference."""
    rows = {r["k"]: r for r in csv.DictReader(open("results/utility_price.csv"))}
    body = open(tex("sections/orders.tex"), encoding="utf-8").read()
    lam = float(rows["3.0"]["lambda_star_u_max"])           # a property of the safe law, same on every row
    assert len({r["lambda_star_u_max"] for r in rows.values()}) == 1
    m = re.search(r"would cost an optimal policy \$([\d.]+)\$ nats\s*(?:,)?\s*and the decoder\s*\n?"
                  r"\s*spends \$(\d+)\$ times that", body.replace("\n", " "))
    assert m, "the ceiling sentence has moved"
    assert float(m.group(1)) == round(lam, 2), (m.group(1), lam)
    best = max(float(r["spend_over_lambda_star_u_max"]) for r in rows.values())
    assert float(m.group(2)) == round(best), (m.group(2), best)


def test_the_two_judge_sigmas_in_the_introduction_come_from_the_v6_separation_csvs():
    """Caution (e): a judged sigma is fragile and must be quoted with its budget. The intro quotes
    four, two judges at two budgets; at k = 0.5 they are -1.03 and +1.52 and at k = 10 they are
    -6.97 and -3.29. The abstract makes the same claim at k = 0.6, where the values are different
    numbers with the same reading, so both budgets are checked."""
    def z(path, k):
        for r in csv.DictReader(open(path)):
            if r["decoder"] == "KL" and float(r["k"]) == k:
                return float(r["z_loss_vs_anchor"])
        raise AssertionError((path, k))
    q, p = "results/judge_separation_v6.csv", "results/judge_separation_v6_judge2.csv"
    body = (open(tex("sections/orders.tex"), encoding="utf-8").read()
            + open(tex("sections/appendix_proofs.tex"), encoding="utf-8").read()
            ).replace("\n", " ")   # tab:repairs moved to the appendix 2026-09-19
    for k, pair in ((0.5, (z(q, 0.5), z(p, 0.5))), (10.0, (z(q, 10.0), z(p, 10.0)))):
        a, b = pair
        assert f"(${a:+.2f}\\sigma$, ${b:+.2f}\\sigma$)".replace("+-", "-") in body \
            or f"(${a:.2f}\\sigma$, ${b:+.2f}\\sigma$)" in body, (k, a, b)
    # the abstract names a budget inside the fully certified region; neither judge separates there
    abstract = open(tex("iclr_2027.tex"), encoding="utf-8").read()
    # whitespace-tolerant: the abstract is rewrapped whenever it is edited, and a line break
    # between "safe" and "model" broke this once. v7 moved the claim into the dichotomy's trivial
    # horn ("... at k=0.5 neither of two judges can separate it from the safe model"), so the
    # phrasing is matched loosely and the substance -- that the named budget really is one where
    # neither judge separates -- is what the assertion below checks.
    # 2026-09-20: this was two exact phrasings and a rescoping of the dichotomy broke both, with
    # the substance untouched -- caution (aj), a guard whose trigger is a sentence someone will
    # reword. Condition on the PROPERTY instead: find every budget the abstract names inside a
    # no-separation claim, and check each really is one where neither judge separates. Rewording
    # cannot retire it; naming a budget where a judge DOES separate still fails.
    flat = " ".join(abstract.split())
    named = [float(mm.group(1)) for mm in re.finditer(r"\$k\s*=\s*([\d.]+)\$", flat)
             if "judge" in (w := flat[max(0, mm.start() - 120): mm.end() + 120])
             and "separat" in w]
    assert named, "the abstract no longer names the budget it claims no separation at"
    for k in named:
        assert abs(z(q, k)) < 2 and abs(z(p, k)) < 2, \
            (k, z(q, k), z(p, k), "the abstract claims no judge separates at a budget where one does")

# RETIRED 2026-09-19, appendix reduction. The paragraph each of these read was removed
# when the appendix was cut from 52 pages, so the sentence they pinned no longer exists.
# A guard for a claim the paper does not make protects nothing; recorded here rather than
# silently deleted, so the removal is visible to the next reader:
#   test_the_metric_sentence_matches_the_metric_block
#   test_the_threshold_sentence_matches_the_threshold_block
#   test_the_k_crit_prediction_sentence_is_a_mean_and_says_so
