"""v16 (2026-09-26), the eighth review round. Each guard pins a correction the review forced, so a later edit cannot
quietly restore the claim it replaced (caution (bd): a guard is re-derived against the review's point, not the old
wording). Mutation-tested before commit: each assertion below fails when its sentence is reverted."""
import csv
import os

from tests.manuscript import body, tex

LIVE = ["iclr_intro.tex", "selection.tex", "frontier.tex", "experiments.tex", "related_work_v4.tex",
        "iclr_closing.tex", "appendix_proofs.tex", "appendix_selection.tex", "appendix_onset.tex",
        "appendix_limitations.tex", "appendix_related.tex", "tab_served.tex"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _main():
    return " ".join(open(tex("iclr_2027.tex"), encoding="utf-8").read().split())


def _abstract():
    a = _main()
    return a[a.index("\\begin{abstract}"): a.index("\\end{abstract}")]


def test_the_windowed_meter_is_not_said_to_certify_exact_windows_only():
    """W7: a per-span max-divergence of W bounds EVERY event in the span by e^W, near-verbatim variants included; what
    is missing is an anchor probability a rights-holder can audit, and the two certificates differ in type."""
    txt = body(*LIVE) + " " + _main()
    for bad in ("exact windows only", "exact window only", "certifies only the exact window"):
        assert bad not in txt, bad
    app = body("appendix_onset.tex")
    assert "bounds every event measurable in the span" in app and "one of degree, not of kind" in app
    assert "rows match in nats, not in type" in body("experiments.tex")


def test_the_price_of_utility_is_a_lemma_credited_to_donsker_varadhan():
    """W1: the Gibbs variational principle is not this paper's theorem, and Proposition 1's identity is elementary."""
    f = body("frontier.tex")
    assert "\\begin{lemma}[The price of utility]\\label{thm:nfl}" in f and "\\begin{theorem}" not in f
    after = f[f.index("\\end{lemma}"):][:400]
    assert "Donsker--Varadhan" in after and "claim no novelty" in after
    s = body("selection.tex")
    assert "The identity is elementary \\citep{vanerven2014renyi}" in s and "new here is the exact threshold" not in s
    assert "Theorem~\\ref{thm:nfl}" not in body(*LIVE) + _main()


def test_an_unvetted_scorer_caps_the_whole_deployment_not_each_user():
    """W5: users can pool transcripts (Appendix A), so the S/log n cap binds every set of users who can, in practice the
    whole deployment; and the scorer we ran is not vetted."""
    s, m = body("selection.tex"), _main()
    for t in (s, m):
        assert "cap each user's queries" not in t and "in practice the whole deployment" in t
    assert "Ours is not: \\texttt{Qwen2.5-7B-Instruct} is web-trained and reads the prompt." in s


def test_the_two_tempered_window_surprisals_are_labelled_by_penalty():
    """m1: 177.4 is the anchor at 0.7 WITH the 1.1 penalty and 179.1 at 0.7 without; the second from its CSV."""
    v2 = next(r for r in csv.DictReader(open(os.path.join(ROOT, "results", "vetting_t07.csv"), encoding="utf-8"))
              if r["band"] == "V2" and r["quantity"].startswith("tinycomma"))
    assert f"{float(v2['value']):.1f}" == "179.1"
    assert "$S_w = 177.4$ nats at $0.7$ with the penalty $1.1$ ($179.1$ without it)" in body("tab_served.tex")
    assert "with the penalty $1.1$ the window's is $177.4$" in body("appendix_selection.tex")


def test_the_abstract_scopes_the_win_to_its_configuration_and_names_the_losses():
    """W3/W4: the matched-certificate win is for the text continuer at temperature 1.0; the chat assistant and the
    authors' 0.7 are losses; the empty-answer qualifier and the weak 70B base travel with the claims."""
    a = _abstract()
    assert "served as a text continuer at temperature $1.0$" in a
    assert "loses to the $8$B at their temperature $0.7$ or served as a chat assistant" in a
    assert "under one of two judges once empty answers count as losses" in a
    assert "whose base model is weak on instruction prompts" in a
