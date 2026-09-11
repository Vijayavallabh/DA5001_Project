"""Every number in the abstract has to appear in the body that justifies it.

The abstract is edited under a page budget and the body is edited for content, so they drift in
opposite directions; audit_numbers.py cannot catch it because it only asks whether a literal appears
in SOME CSV, and an abstract number that has gone stale still does."""
import re

from tests.manuscript import tex

SECTIONS = ("iclr_intro", "frontier", "scaling", "onset", "orders", "related_work_v4",
            "iclr_closing", "appendix_proofs", "appendix_opening", "appendix_robustness",
            "appendix_seed", "appendix_limitations", "appendix_related", "appendix_second_anchor")
# Numbers that are structural rather than measured: page/section counts, an exponent, a budget the
# body writes as k = 10 rather than $10$.
ALLOWED = {"1", "2", "10"}


def _abstract():
    body = open(tex("iclr_2027.tex"), encoding="utf-8").read()
    return body.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0]


def _literals(text):
    return set(re.findall(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])",
                          " ".join(re.findall(r"\$([^$]*)\$", text))))


def test_every_number_in_the_abstract_appears_in_the_body():
    body = "".join(open(tex(f"sections/{s}.tex"), encoding="utf-8").read() for s in SECTIONS)
    body_nums = _literals(body) | set(re.findall(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])", body))
    missing = sorted(n for n in _literals(_abstract()) - body_nums if n not in ALLOWED)
    assert not missing, f"in the abstract but nowhere in the body: {missing}"


def test_the_abstract_the_intro_and_section_4_agree_on_the_pair_count():
    """The count is spelled in words, so audit_numbers.py cannot see it at all, and it has to be
    changed in three places every time a pair is added."""
    import csv
    n = len([r for r in csv.DictReader(open("results/onset_table.csv"))
             if not r["pair"].startswith("ALL")])
    word = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"][n]
    a = _abstract().replace("\n", " ")
    i = open(tex("sections/iclr_intro.tex"), encoding="utf-8").read().replace("\n", " ")
    s4 = open(tex("sections/onset.tex"), encoding="utf-8").read().replace("\n", " ")
    assert f"across {word} model pairs with {word} distinct anchors" in a, a[:0] or word
    assert f"Across {word} pairs with {word} distinct anchors" in i, word
    assert f"Across {word} pairs with {word} distinct anchors" in s4, word


def test_every_results_file_the_paper_names_exists():
    """A citation to a file that is not in the artifact is a reviewer's first click and a desk
    rejection risk on the artifact track. Names are typeset with escaped underscores."""
    import glob
    import os
    body = "".join(open(f, encoding="utf-8").read()
                   for f in [tex("iclr_2027.tex")] + sorted(glob.glob(tex("sections/*.tex")))
                   if os.path.basename(f).replace(".tex", "") in SECTIONS or f.endswith("iclr_2027.tex"))
    named = {n.replace("\\_", "_") for n in re.findall(r"results/([A-Za-z0-9_\\]+)", body)}
    missing = sorted(n for n in named if n and not
                     (os.path.exists(f"results/{n}.csv") or os.path.exists(f"results/{n}.md")))
    assert not missing, f"named in the paper, absent from results/: {missing}"
