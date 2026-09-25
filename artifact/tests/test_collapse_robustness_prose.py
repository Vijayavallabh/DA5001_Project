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
    # 2026-09-24: the sentence moved from Section 4.4 to Proposition 4's discussion in frontier.tex,
    # beside the proviso it illustrates; both files are read so a later move is still followed.
    # v10 (same day) retired orders.tex and put Proposition 4's proviso, with the $1.30$ nats, in its
    # proof in appendix_proofs.tex; frontier.tex (Section 3) and appendix_proofs.tex are read now.
    body = "".join(open(tex(f"sections/{f}.tex"), encoding="utf-8").read()
                   for f in ("frontier", "appendix_proofs"))
    lam = float(rows["3.0"]["lambda_star_u_max"])           # a property of the safe law, same on every row
    assert len({r["lambda_star_u_max"] for r in rows.values()}) == 1
    best = max(float(r["spend_over_lambda_star_u_max"]) for r in rows.values())
    flat = " ".join(body.split())
    # v13 (2026-09-25, seventh review round): the ceiling is now read on the ORDER-AVERAGED instrument
    # every other judged number in the paper uses (results/frontier_ratio.csv: the anchor wins BOTH
    # presentation orders), and the single-order figures above survive only as a labelled comparison.
    fr = {r["group"]: r for r in csv.DictReader(open("results/frontier_ratio.csv"))}["all"]
    m = re.search(r"would cost an optimal policy \$([\d.]+)\$ nats\s*,?\s*and the decoder "
                  r"spends \$([\d.]+)\$ times that", flat)
    assert m, "the ceiling sentence has moved"
    assert m.group(1) == f"{float(fr['log_inv_pi']):.2f}", (m.group(1), fr["log_inv_pi"])
    assert m.group(2) == f"{float(fr['ratio_at_u_max']):.1f}", (m.group(2), fr["ratio_at_u_max"])
    # the single-order reading is still quoted, and only as the single-order pass's
    s = re.search(r"on a single-order pass[^.]*?the same reading is \$([\d.]+)\$ nats and \$(\d+)\$ times", flat)
    assert s, "the single-order ceiling lost its label"
    assert float(s.group(1)) == round(lam, 2) and float(s.group(2)) == round(best), (s.groups(), lam, best)


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
    # tab:repairs moved to the appendix 2026-09-19 and back into Section 3 (frontier.tex) in v10
    # (2026-09-24), where its row prints the k=0.5 pair; the full sentence with both budgets is the
    # "A lower cap." paragraph of Appendix H (appendix_onset.tex). orders.tex is retired.
    body = " ".join((open(tex("sections/frontier.tex"), encoding="utf-8").read()
                     + open(tex("sections/appendix_onset.tex"), encoding="utf-8").read()).split())
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
