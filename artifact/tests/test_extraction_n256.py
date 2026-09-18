"""The safety claim now covers n=256, and must not be rounded up to "any n <= 256".

Arm B of results/onset_prediction_n256.md measured extraction on the committed grid 1, 8, 64, 256.
The earlier arm measured 1, 2, 4, 8, 16, 32, 64. Their union is the doubling grid to 64 plus 256 --
**n=128 is measured by neither**, because Arm A measures the judged frontier and not leakage. So the
paper says "at every n <= 64, and at n=256" and never "at any n <= 256", which would assert a point
nobody ran. Caution (v)/(aa): the number is right and the quantity named must be too.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import DIR, body, tex  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE = ("iclr_intro.tex", "experiments.tex", "selection.tex", "appendix_selection.tex")


def _csv(name):
    return list(csv.DictReader(open(os.path.join(ROOT, "results", name), encoding="utf-8")))


def test_the_n256_arm_really_reads_zero_on_every_passage():
    rows = _csv("selection_extraction_n256.csv")
    grid = [int(float(r["n"])) for r in rows if r["n"] not in ("-1", "")]
    assert grid == [1, 8, 64, 256], f"the committed grid moved: {grid}"
    for r in rows:
        if r["n"] in ("-1", ""):
            continue
        assert float(r["nv_recall_mean"]) == 0.0 and float(r["nv_recall_max"]) == 0.0, r
        assert float(r["rouge_ge_0p5_pct"]) == 0.0, r
        assert int(float(r["n_passages"])) == 100, r
    base = next(r for r in rows if r["n"] == "-1")
    # caution (u): a batch-size drift would move this to 0.4434 and look like a real defect
    assert abs(float(base["nv_recall_mean"]) - 0.3925) < 5e-5, base["nv_recall_mean"]
    assert abs(float(base["rouge_ge_0p5_pct"]) - 47.0) < 5e-5, base["rouge_ge_0p5_pct"]


def test_n128_is_measured_by_neither_arm_so_the_claim_may_not_span_to_256():
    """The reason the wording is 'n <= 64, and n=256' rather than 'n <= 256'."""
    fine = {int(float(r["n"])) for r in _csv("selection_extraction.csv") if r["n"] not in ("-1", "")}
    coarse = {int(float(r["n"])) for r in _csv("selection_extraction_n256.csv") if r["n"] not in ("-1", "")}
    assert 128 not in (fine | coarse), \
        "n=128 extraction now exists; the claim may be widened, but say so deliberately"


def test_no_live_section_claims_the_span_to_256():
    import glob
    import re
    live = [f for f in glob.glob(os.path.join(DIR, "sections", "*.tex"))
            if not re.search(r"_v\d", os.path.basename(f))] + [os.path.join(DIR, "iclr_2027.tex")]
    txt = " ".join(" ".join(open(f, encoding="utf-8").read().split()) for f in live)
    for bad in ("any $n \\le 256$", "every $n \\le 256$", "$n$ up to $256$", "all $n \\le 256$"):
        assert bad not in txt, f"n=128 extraction was never measured, so this overclaims: {bad!r}"


def test_the_headline_safety_sentences_carry_the_n256_point():
    """Caution (ag): a concession or a strengthening is only as durable as its guard."""
    main = " ".join(open(tex("iclr_2027.tex"), encoding="utf-8").read().split())
    assert "at any $n \\le 64$ or at $256$" in main, "the abstract lost the n=256 extension"
    intro = body("iclr_intro.tex")
    assert intro.count("$n=256$") >= 2, "the intro's two safety sentences must both carry n=256"
    exp = body("experiments.tex")
    assert "and at $n=256$" in exp, "Section 3's adversarial-scorer paragraph lost n=256"


def test_the_extraction_table_carries_the_n256_column_and_its_provenance():
    app = body("appendix_selection.tex")
    i = app.find("\\label{tab:extraction}")
    assert i != -1
    cap = app[app.rfind("\\caption{", 0, i):i]
    assert "$n=256$ column is a later arm" in cap, "the n=256 column lost its provenance"
    assert "$n=128$ is measured by neither" in cap, "the caption must say where the gap is"
    tab = app[i:i + 1400]
    assert "$256$ & risky alone" in tab, "the table lost its n=256 column"
    assert "ROUGE-L $\\ge 0.5$ of $100$" in tab, "the ROUGE-L secondary row was trimmed"
