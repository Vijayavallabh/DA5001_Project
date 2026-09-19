"""feat-128: is a bigger safe model served once the cheaper answer?

The arm's numbers reach the manuscript through one appendix paragraph and one clause in Section 2,
and the committed statistic -- the paired difference -- was formed by hand at scoring time and lived
only in the scoring log until the third read-through on 2026-09-17. A number the paper quotes and no
CSV carries is a number nothing can check, so the script computes it and these tests pin both places
that print it to what the script wrote.
"""
import csv
import re

from tests.manuscript import tex

CSV = "results/bigger_anchor.csv"


def rows():
    return {r["quantity"].split("  ")[0]: r for r in csv.DictReader(open(CSV))}


def test_the_appendix_table_rounds_from_the_csv():
    # Whitespace-normalised: the appendix was reflowed on 2026-09-19 and a value can now sit at
    # the end of one source line with its interval at the start of the next. Joining on whitespace
    # keeps the assertion exact -- the value and its interval must still be adjacent in the
    # rendered text -- without pinning where TeX's source lines happen to break.
    body = " ".join(open(tex("sections/appendix_selection.tex"),
                         encoding="utf-8").read().split())
    para = body.split(r"\label{app:biggeranchor}")[1].split(r"\paragraph{")[0]
    for key in ("G_A", "G_B", "Comma-7B n=64, over TinyComma alone"):
        r = rows()[key]
        cell = "${:+.4f}$ $[{:+.4f}, {:+.4f}]$".format(
            float(r["value"]), float(r["lo95"]), float(r["hi95"]))
        assert cell in para, (key, cell)


def test_the_committed_difference_is_the_one_the_script_computed():
    """Quoted as [-0.1220, -0.0770] from the scoring log until 2026-09-17; the script's own
    resampling stream gives -0.0765 on the same 500 paired values. Bootstrap noise either way, but
    the paper quotes the CSV, once (caution (j))."""
    r = rows()["G_A - G_B"]
    want = "${:+.4f}\\,[{:+.4f},{:+.4f}]$".format(
        float(r["value"]), float(r["lo95"]), float(r["hi95"]))
    body = " ".join(open(tex("sections/appendix_selection.tex"),
                         encoding="utf-8").read().split())
    assert want in body, (want, "the appendix does not quote the CSV's difference")
    assert float(r["hi95"]) < 0, "the verdict rests on the interval excluding zero"


def test_section_2_and_the_appendix_agree_on_the_bigger_anchors_own_gain():
    """Section 2 states the gain in 3dp and the appendix in 4dp, which is how they drift."""
    g = float(rows()["G_A"]["value"])
    sec = open(tex("sections/selection.tex"), encoding="utf-8").read().replace("\n", " ")
    sent = next(s for s in sec.split(". ") if "larger safe model served alone" in s)
    quoted = [float(x) for x in re.findall(r"\$([+-]\d\.\d+)\$", sent)]
    assert round(g, 3) in [round(q, 3) for q in quoted], (g, quoted)


def test_section_2_points_at_the_appendix_that_answers_it():
    """It pointed at app:saturation -- the SCORER-scale paragraph, which says nothing about a
    bigger anchor. Nothing catches a \\ref that resolves to the wrong content: tectonic exits 0 and
    ?? is 0, because the label exists. Only a reader following the click sees it."""
    sec = open(tex("sections/selection.tex"), encoding="utf-8").read().replace("\n", " ")
    sent = next(s for s in sec.split(". ") if "larger safe model served alone" in s)
    m = re.search(r"\\ref\{(app:[a-z]+)\}", sent)
    assert m, sent
    target = open(tex("sections/appendix_selection.tex"), encoding="utf-8").read()
    para = target.split("\\label{%s}" % m.group(1))[1].split(r"\paragraph{")[0]
    assert "bigger anchor" in para.lower() or "larger safe model" in para.lower(), \
        (m.group(1), "Section 2's reference lands on a paragraph that is not about this claim")
