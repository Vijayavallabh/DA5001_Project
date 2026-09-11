"""Every number in the abstract has to appear in the body that justifies it.

The abstract is edited under a page budget and the body is edited for content, so they drift in
opposite directions; audit_numbers.py cannot catch it because it only asks whether a literal appears
in SOME CSV, and an abstract number that has gone stale still does."""
import re

from tests.manuscript import tex

SECTIONS = ("iclr_intro", "frontier", "onset", "orders", "selection", "experiments",
            "related_work_v4", "iclr_closing", "appendix_proofs", "appendix_opening",
            "appendix_onset", "appendix_robustness", "appendix_seed", "appendix_limitations",
            "appendix_related", "appendix_selection", "appendix_second_anchor")
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


def test_the_abstract_the_intro_and_the_onset_section_agree_on_the_pair_count():
    """The count is spelled in words, so audit_numbers.py cannot see it at all, and it has to be
    changed in three places every time a pair is added. The phrasing is free; the number is not."""
    import csv
    n = len([r for r in csv.DictReader(open("results/onset_table.csv"))
             if not r["pair"].startswith("ALL")])
    word = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"][n]
    places = {"abstract": _abstract(),
              "intro": open(tex("sections/iclr_intro.tex"), encoding="utf-8").read(),
              "onset": open(tex("sections/onset.tex"), encoding="utf-8").read()}
    for where, body in places.items():
        found = re.findall(r"(\w+) (?:model )?pairs with \1 distinct anchors",
                           body.replace("\n", " "))
        assert found, f"{where} does not state the pair count beside the anchor count"
        assert all(f.lower() == word for f in found), (where, found, word)


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
