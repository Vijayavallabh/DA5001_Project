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


def test_granularity_does_not_order_the_nine_pairs_and_the_count_says_so():
    """Rescoped 2026-09-19. The sentence this guarded ("of the 34 comparable anchor pairs the ratio
    is discordant on 2 and the step count on 3") left the manuscript with appendix_seed, retired for
    the page budget and kept verbatim as sections/appendix_seed_v8_2026-09-19.tex. The claim it had
    replaced -- "all three order the nine pairs the way the onset ratio is ordered" -- was FALSE, so
    what must not come back is the stronger version. This now asserts against the CSV that
    granularity really does fail to order the pairs; a section that claims otherwise fails
    tests/test_shape_claims.py.
    """
    import csv as _csv
    import itertools
    rs = list(_csv.DictReader(open("results/onset_seed_words.csv", encoding="utf-8")))
    for r in rs:
        r["cpt"], r["ra"] = float(r["chars_per_token"]), float(r["ratio"])
        r["st"] = float(r["steps_to_passage"])
    comparable = [(a, b) for a, b in itertools.combinations(rs, 2) if a["cpt"] != b["cpt"]]
    assert len(comparable) == 34, len(comparable)
    disc_ratio = sum((a["cpt"] > b["cpt"]) == (a["ra"] > b["ra"]) for a, b in comparable)
    disc_steps = sum((a["cpt"] > b["cpt"]) == (a["st"] > b["st"]) for a, b in comparable)
    # discordant in both directions: the ordering is a tendency, never exact
    assert 0 < disc_ratio < len(comparable), disc_ratio
    assert 0 < disc_steps < len(comparable), disc_steps
