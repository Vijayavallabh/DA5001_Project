"""The normaliser growth series in Appendix D, pinned to the CSV that now generates it.

The series was maintained BY HAND across sessions and nothing regenerated it, which is how it went
stale: it stopped at seven pairs while the paper reports nine, and the sentence around it said "the
winner changes every time a pair is added" -- contradicted by the six numbers printed beside it in
the same sentence. Found in the fourth read-through, 2026-09-17. Same shape as caution (ai): the
values were each defensible and the CLAIM ABOUT them was checked by nothing.

analysis/normaliser_growth.py now produces the series over nested prefixes of the pair manifest,
through the committed ablation, so these tests can read it.
"""
import csv

from tests.manuscript import tex

CSV = "results/normaliser_growth.csv"


def rows():
    return list(csv.DictReader(open(CSV, encoding="utf-8")))


def test_the_growth_series_covers_every_pair_the_paper_reports():
    n = len([l for l in open("results/onset_pairs.tsv", encoding="utf-8").read().splitlines()
             if l.strip()])
    got = [int(r["n_pairs"]) for r in rows()]
    assert got == list(range(2, n + 1)), (got, n, "the series does not run to the pair count")


def test_the_nine_pair_row_is_the_committed_ablation():
    """Same code path, so the last row must equal collapse_robustness.csv's normaliser block.
    If this fails the growth script has drifted from the ablation it is supposed to re-run."""
    block = {r["setting"]: r["value"] for r in
             csv.DictReader(open("results/collapse_robustness.csv", encoding="utf-8"))
             if r["block"] == "normaliser"}
    last = rows()[-1]
    for key, col in (("s_safe", "s_safe"),
                     ("requirement r = s_safe - s_risky", "requirement"),
                     ("raw (no rescaling)", "raw")):
        assert round(float(block[key]), 6) == float(last[col]), (key, block[key], last[col])


def test_the_seed_table_intro_counts_its_discordances_correctly():
    """"all three order the nine pairs the way the onset ratio is ordered" was false, and the
    honest statement is a count of discordant anchor pairs -- computed here rather than trusted.
    Two anchors share a tokenizer with two others, so granularity predicts nothing between them
    and those comparisons are excluded rather than counted as agreements."""
    import csv as _csv
    import itertools
    import re
    rs = list(_csv.DictReader(open("results/onset_seed_words.csv", encoding="utf-8")))
    for r in rs:
        r["cpt"], r["ra"] = float(r["chars_per_token"]), float(r["ratio"])
        r["st"] = float(r["steps_to_passage"])
    comparable = [(a, b) for a, b in itertools.combinations(rs, 2) if a["cpt"] != b["cpt"]]
    # a finer tokenizer (LOWER chars/token) should mean a HIGHER ratio and MORE steps
    disc_ratio = sum((a["cpt"] > b["cpt"]) == (a["ra"] > b["ra"]) for a, b in comparable)
    disc_steps = sum((a["cpt"] > b["cpt"]) == (a["st"] > b["st"]) for a, b in comparable)

    body = open(tex("sections/appendix_seed.tex"), encoding="utf-8").read()
    flat = " ".join(body.split())
    # the claim runs from the tokenizer sentence to the table it introduces; it contains a colon
    # of its own, so it cannot be found by splitting on one
    sent = flat.split("changes three things at once")[1].split(r"\begin{center}")[0]
    nums = [int(x) for x in re.findall(r"\$(\d+)\$", sent)]
    assert nums == [len(comparable), disc_ratio, disc_steps], (nums,
        [len(comparable), disc_ratio, disc_steps], "the intro's counts are not the CSV's")
    assert "order the nine pairs the way the onset ratio is ordered" not in sent, \
        "the refuted exact-ordering claim is back; the table's own second row breaks it"

# RETIRED 2026-09-19, appendix reduction. The paragraph each of these read was removed
# when the appendix was cut from 52 pages, so the sentence they pinned no longer exists.
# A guard for a claim the paper does not make protects nothing; recorded here rather than
# silently deleted, so the removal is visible to the next reader:
#   test_the_appendix_states_the_flips_the_series_actually_has
#   test_the_series_the_appendix_prints_is_the_series_the_csv_holds
