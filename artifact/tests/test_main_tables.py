"""The two remaining main-text tables, checked against their CSVs instead of by eye.

The scaling table was the one plan v5 caught with two of three rows stale; the order table is the
only place alpha = 4 is priced. Both moved to the appendix in v6 when the main text was
restructured around selection anchoring; they are still hand-copied, which is what this checks. Both are small and both are hand-copied, which
is the combination that goes wrong quietly.
"""
import csv
import decimal
import re

from tests.manuscript import tex


def rnd(x, places):
    """Round half away from zero, which is what a person copying a CSV into LaTeX does. Python's
    round() is half-to-even on a binary float and disagrees on exactly the cells that matter."""
    q = decimal.Decimal(10) ** -places
    return float(decimal.Decimal(repr(x)).quantize(q, rounding=decimal.ROUND_HALF_UP))


def _cells(path, first_col):
    for line in open(path, encoding="utf-8"):
        c = [x.strip() for x in line.rstrip().rstrip("\\").split("&")]
        if len(c) > 1 and c[0] in first_col:
            yield c


def _num(cell):
    m = re.search(r"-?\d+\.?\d*", cell.replace("{,}", ""))
    return float(m.group()) if m else None


def test_the_scaling_table_matches_anchor_scaling_summary():
    """corpus, span, delta c_use, delta s(x), margin small -> large, sign test."""
    sm = {r["model"]: r for r in csv.DictReader(open("results/anchor_scaling_summary.csv"))}
    pr = {r["corpus"]: r for r in csv.DictReader(open("results/anchor_scaling_paired.csv"))}
    rows = {"Common Corpus": "commoncorpus", "KL3M": "kl3m", "Common Pile": "commonpile"}
    seen = 0
    # appendix_robustness.tex was retired in v10 (2026-09-24); the table (tab:marginspan) is now in
    # Appendix G, appendix_onset.tex.
    for c in _cells(tex("sections/appendix_onset.tex"), rows):
        p = pr[rows[c[0]]]
        a, b = sm[p["small"]], sm[p["large"]]
        assert _num(c[2]) == rnd(100 * (float(b["c_use"]) / float(a["c_use"]) - 1), 1), (c[0], "c_use")
        assert _num(c[3]) == rnd(100 * (float(b["s_passage"]) / float(a["s_passage"]) - 1), 1), (c[0], "s")
        lo, hi = (float(x) for x in re.findall(r"\d+\.\d+", c[4]))
        assert (lo, hi) == (rnd(float(a["margin"]), 2), rnd(float(b["margin"]), 2)), (c[0], "margin")
        assert f"{p['novels_margin_up']}/{p['n_novels']}" in c[5], (c[0], "sign test")
        seen += 1
    assert seen == 3, seen


def test_the_order_table_matches_the_renyi_sweep_and_price():
    """Attack recall from renyi_sweep.csv at k=3, the price columns from renyi_price.csv."""
    sweep = {(float(r["alpha"]), r["mode"]): float(r["nv_recall_mean"])
             for r in csv.DictReader(open("results/renyi_sweep.csv")) if float(r["k"]) == 3.0}
    price = {r["arm"]: r for r in csv.DictReader(open("results/renyi_price.csv"))}
    arm = {1.0: "renyi_1_0", 2.0: "renyi_2", 4.0: "renyi_4", 8.0: "renyi_8"}
    rows = {r"$\alpha = 1$ (KL)": 1.0, r"$\alpha = 2$": 2.0,
            r"$\alpha = 4$": 4.0, r"$\alpha = 8$": 8.0}
    seen = 0
    for c in _cells(tex("sections/appendix_onset.tex"), rows):
        a = rows[c[0]]
        assert _num(c[1]) == rnd(sweep[(a, "single")], 4), (a, "single")
        assert _num(c[2]) == rnd(sweep[(a, "oracle")], 4), (a, "oracle")
        p = price[arm[a]]
        want = rnd(float(p["risky_unchanged_pct"]), 2 if a == 8.0 else 1)
        assert _num(c[3]) == want, (a, "risky unchanged", _num(c[3]), want)
        assert _num(c[4]) == rnd(float(p["active_pct"]), 1), (a, "steps touched")
        assert _num(c[5]) == rnd(float(p["distinct3"]), 4), (a, "distinct-3")
        seen += 1
    assert seen == 4, seen


def test_section_5_quotes_table_3s_own_binding_rates():
    """Repair 1's paragraph cites Table 3 and then states the binding rates in prose, so the two
    have to be the same measurement. Until 2026-09-17 they were not: the prose carried the
    500-prompt re-run (99.3% and 0.35%) while the table carries renyi_price.csv's 150 (99.6% and
    0.4%). Both are real and the claim is the same either way, but the reader is pointed at the
    table -- and 99.3 is ALSO the number in Table 3's alpha=1 risky-unchanged cell, so the prose
    read like a transposed row. The prose now quotes the cells, and this pins it to the CSV they
    come from."""
    price = {r["arm"]: r for r in csv.DictReader(open("results/renyi_price.csv"))}
    # v10 (2026-09-24) retired orders.tex. Repair 1 is now the "sharper charge" row of Section 3's
    # Table tab:repairs ("binds on $99.6\\%$ of steps vs $0.4\\%$"), which points at Appendix H,
    # and Appendix H's "A sharper charge" paragraph quotes the alpha=8 cell of tab:orders in prose.
    body = " ".join(open(tex("sections/frontier.tex"), encoding="utf-8").read().split())
    sent = next(s for s in body.split("\\\\") if "sharper charge" in s)
    for arm, key in (("renyi_8", "active_pct"), ("renyi_1_0", "active_pct")):
        want = rnd(float(price[arm][key]), 1)
        assert f"${want}\\%$" in sent, (arm, want, sent)
    apx = " ".join(open(tex("sections/appendix_onset.tex"), encoding="utf-8").read().split())
    para = apx[apx.index("\\paragraph{A sharper charge"):]
    para = para[:para.index("\\begin{table}")]
    want = rnd(float(price["renyi_8"]["active_pct"]), 1)
    assert f"touches ${want}\\%$ of ordinary steps" in para, (want, para[:300])
