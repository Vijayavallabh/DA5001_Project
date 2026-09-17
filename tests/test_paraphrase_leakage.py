"""feat-127: the paraphrase claim, and the control that is the only reason its null means anything.

Section 2 says the certificate bounds paraphrase "on the same terms as exact quotation" and backs it
with a measurement at a non-literal event: the anchor reaches ROUGE-L >= 0.5 on 0 of 100 passages.

A zero is the easiest kind of bug to mistake for a result (caution (t)), and a null on a metric that
fires at nothing would be exactly that. It is not, because the SAME metric on the SAME passages at
the SAME threshold reads 47/100 on the model that memorised the text. That clause is the load-bearing
half of the sentence and it is the half a page-budget trim would reach for first, so it is pinned
here beside the number it protects.
"""
import csv
import re

from tests.manuscript import tex

CSV = "results/selection_extraction_paraphrase.csv"


def _arms():
    return {r["n"]: r for r in csv.DictReader(open(CSV))}


def test_rouge_l_is_a_subsequence_metric_so_the_event_is_genuinely_non_literal():
    """The whole arm rests on ROUGE-L being looser than the substring metrics every other
    extraction number uses (caution (j): `lcs_word` is the longest common SUBSTRING). If this ever
    became a substring score the arm would be measuring what the paper already measured."""
    from dap.stats import rouge_l_score

    def f(lcs, h, r):                      # the F-measure the score is built from
        p_, r_ = lcs / h, lcs / r
        return 2 * p_ * r_ / (p_ + r_)

    # five words, every one of them separated in the reference: as a SUBSEQUENCE the match is 5,
    # as a SUBSTRING it is 1. The score has to be the former.
    got = rouge_l_score("a b c d e", "a x b x c x d x e")
    assert abs(got - f(5, 5, 9)) < 1e-9, (got, "not scoring the longest common subsequence")
    assert got > f(1, 5, 9), (got, "scoring a substring, which is what the paper already measures")
    assert rouge_l_score("a b c", "c b a") < got    # and order still costs something


def test_the_anchor_rate_and_its_positive_control_are_both_in_the_manuscript():
    arms = _arms()
    anchor = float(arms["1"]["rouge_ge_0p5_pct"])
    mem = float(arms["-1"]["rouge_ge_0p5_pct"])
    n = 100
    body = open(tex("sections/selection.tex"), encoding="utf-8").read().replace("\n", " ")
    sent = next(s for s in body.split(". ") if "ROUGE-L" in s)
    assert f"${int(anchor * n / 100)}$ of ${n}$" in sent, (sent, anchor)
    assert str(int(mem * n / 100)) in sent, (
        sent, mem, "the positive control is gone: without it the 0 of 100 is an unfalsifiable null")


def test_the_served_rate_is_not_quoted_as_if_it_were_the_base_rate():
    """q(E) and p_s(E) are both 0 here, which makes them easy to conflate. The certificate
    multiplies the ANCHOR's rate, so the sentence has to be about the anchor."""
    body = open(tex("sections/selection.tex"), encoding="utf-8").read().replace("\n", " ")
    sent = next(s for s in body.split(". ") if "ROUGE-L" in s)
    assert "anchor" in sent, sent
    assert float(_arms()["64"]["rouge_ge_0p5_pct"]) == 0.0, \
        "the served rate is no longer zero; the sentence needs rewriting, not just this test"


def test_the_amplification_is_reported_as_undefined_and_never_as_one():
    """Registered explicitly: if p_s(E) = 0 the ratio q/p_s is undefined and is reported as
    undefined, never as 1 and never quietly dropped."""
    assert float(_arms()["1"]["rouge_ge_0p5_pct"]) == 0.0
    body = open(tex("sections/selection.tex"), encoding="utf-8").read().replace("\n", " ")
    sent = next(s for s in body.split(". ") if "ROUGE-L" in s)
    assert not re.search(r"amplification of \$?1\$?\b", sent), \
        "an undefined ratio is being quoted as 1"
