"""The breadth arm's entry gate, and the two ways it was broken.

The gate is registered as "mean generation length > 20 tokens and fewer than 5% empty
completions". The first implementation read a summary column named `mean_tokens` that
`selection_scaling.py` has never written, so `.get(...) or 0` returned 0.0 and marked EVERY anchor
FAIL -- including the audited one, whose result is the paper's headline. The second half of the
gate was not implemented at all. Both are pinned here, along with the empty-completion sensitivity
check that running the gate properly forced.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_breadth import (ANCHORS, GATE_EMPTY, GATE_TOKENS,  # noqa: E402
                                        SCORING_JUDGE)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "selection_breadth.csv")


def rows():
    with open(CSV, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_the_gate_thresholds_are_the_registered_ones():
    assert GATE_TOKENS == 20.0
    assert GATE_EMPTY == 0.05


def test_the_gate_reads_a_column_the_producer_actually_writes():
    """The original bug in one line: the gate's input must exist in the CSV it claims to read.
    `mean_tokens` does not; `mean_words` does. The gate now measures the generations instead, so
    what this guards is that it never goes back to a summary column that is silently absent."""
    src = open(os.path.join(ROOT, "analysis", "selection_breadth.py"), encoding="utf-8").read()
    assert 'one.get("mean_tokens")' not in src, "the gate is reading the column that never existed"
    assert "generation_length_tokens" in src, "the gate must measure the generations"
    prod = open(os.path.join(ROOT, "analysis", "selection_scaling.py"), encoding="utf-8").read()
    assert "mean_tokens" not in prod, "if selection_scaling.py starts writing it, revisit the gate"


def test_at_least_one_anchor_passes_the_gate():
    """A gate that fails everything is not a gate. This is the assertion whose absence let the
    broken version through a whole run."""
    r = rows()
    assert r, "no breadth rows yet"
    assert any(x["entry_gate"] == "PASS" for x in r), [x["entry_gate"] for x in r]


def test_the_gate_is_computed_from_both_halves():
    """Every row carries both quantities, and the verdict is their conjunction."""
    for x in rows():
        tok, emp = float(x["mean_tokens_n1"]), float(x["empty_frac_n1"])
        assert tok > 0, x["anchor"]
        want = "PASS" if (tok >= GATE_TOKENS and emp < GATE_EMPTY) else "FAIL"
        assert x["entry_gate"] == want, (x["anchor"], tok, emp, x["entry_gate"])


def test_the_audited_anchor_fails_the_empty_half_and_is_reported_not_dropped():
    """6.8% empty at n=1 against a registered 5%. It is the paper's own anchor, so the failure is
    recorded rather than exempted -- and the sensitivity check below is why it is survivable."""
    aud = [x for x in rows() if "audited" in x["anchor"]]
    assert aud, "the audited anchor must stay in the table"
    for x in aud:
        assert float(x["empty_frac_n1"]) > GATE_EMPTY
        assert float(x["mean_tokens_n1"]) >= GATE_TOKENS   # it can write; it sometimes writes nothing
        assert x["entry_gate"] == "FAIL"


def test_the_gain_is_not_a_degeneracy_filter():
    """Best-of-n never picks an empty candidate, so a gain could be nothing but a filter on the
    empties. The worry is DEFLATION: if the gain is a filter artefact, removing the prompts whose
    n=1 completion is empty makes it collapse. At a gate-passing anchor it must barely move; at
    one that fails the gate on empties -- Comma-1T at 16.6% -- it may move, and what must stay
    true is that it does not shrink. It grows there (judge C, +0.113 -> +0.143), which is the
    opposite of a filter artefact and is why the strict band is scoped rather than loosened."""
    strict = loose = 0
    for x in rows():
        if x["gain_nonempty"] == "":
            continue
        g, gn = float(x["gain"]), float(x["gain_nonempty"])
        if x["entry_gate"] == "PASS":
            strict += 1
            assert abs(g - gn) < 0.01, x
        else:
            loose += 1
            assert gn >= 0.5 * g, (x["anchor"], x["judge"], g, gn,
                                   "the gain collapses without the empty-completion prompts")
    assert strict >= 2, "the sensitivity check produced nothing to check"
    assert loose >= 1, "no gate-failing anchor is being checked; scope the band again"


def test_b1_is_read_off_the_registered_scorer_only():
    """Excluded alternative 3 forbids changing the judge between anchors, and taking 'either judge
    excludes zero' would be two chances at one band."""
    src = open(os.path.join(ROOT, "analysis", "selection_breadth.py"), encoding="utf-8").read()
    assert SCORING_JUDGE == "Phi-3.5-mini-instruct"
    assert 'r["judge"] == SCORING_JUDGE' in src
    assert "secondary judge, reported not scored" in src


def test_every_registered_anchor_has_a_generation_directory_declared():
    """The gate needs the artefacts, so an anchor with no directory is a configuration error, not
    a silent PASS."""
    assert len(ANCHORS) >= 4, ANCHORS
    for label, tag, model, gen_dir in ANCHORS:
        assert gen_dir.startswith("output/phase5/"), (label, gen_dir)
        assert (tag == "") == ("audited" in label)
        assert model.count("/") == 1, model
    # An anchor added by copy-paste that forgets to change the tag or the directory would score the
    # same generations twice under two names, which is the failure this replaced a literal 4 with.
    for field, i in (("label", 0), ("tag", 1), ("model", 2), ("gen_dir", 3)):
        seen = [a[i] for a in ANCHORS]
        assert len(set(seen)) == len(seen), (field, seen)


def test_section6_quotes_the_four_anchor_gains_from_the_breadth_csv():
    """Section 6's breadth paragraph names three gains. They round from selection_breadth.csv on
    the registered scorer, once, and the claim 'two of three exclude zero' has to be true of it."""
    from tests.manuscript import tex
    body = open(tex("sections/experiments.tex"), encoding="utf-8").read().replace("\n", " ")
    # B1 was scored at FOUR anchors and the six-anchor pre-registration's excluded alternatives
    # close it, so the sentence names the three non-audited anchors it was scored on -- not every
    # anchor later added to the CSV for C1-C3.
    b1 = {"Pleias-1.2B", "KL3M-1.7B", "Comma-7B"}
    new = [r for r in rows() if r["anchor"] in b1 and r["judge"] == SCORING_JUDGE]
    assert len(new) == 3, [r["anchor"] for r in new]
    # These three moved out of the prose and into Figure 1's forest plot on 2026-09-17. The band
    # is still the paper's claim and still has to round from the CSV once; what changed is which
    # surface prints it, so the check reads both (tests/manuscript.py::carries_band).
    from tests.manuscript import carries_band
    for r in new:
        g, lo, hi = (float(r["gain"]), float(r["gain_lo95"]), float(r["gain_hi95"]))
        assert carries_band(g, lo, hi, "experiments.tex"), (r["anchor"], g, lo, hi)
    excl = sum(1 for r in new if float(r["gain_lo95"]) > 0)
    assert excl == 2, excl
    # the claim "two of three exclude zero" is now made by the plot, so assert the plot makes it
    assert "selection_breadth_forest" in body, "the figure carrying these bands is not included"
    plotted = {lbl.split(",")[0]: (lo_, hi_) for lbl, (_g, lo_, hi_), *_ in
               __import__("tests.manuscript", fromlist=["_forest"])._forest() if lo_ is not None}
    for r in new:
        assert r["anchor"] in plotted, (r["anchor"], sorted(plotted))
    assert sum(1 for r in new if plotted[r["anchor"]][0] > 0) == 2, plotted
    best = max(new, key=lambda r: float(r["gain"]))
    assert "Comma-7B" in best["anchor"], best["anchor"]
    # C2 of the six-anchor pre-registration read NO TREND (rho = +0.543 over six), and its
    # committed consequence is that the superlative comes OUT. It may only come back if the
    # correlation reaches the registered +0.6.
    import csv as _csv
    import math as _math
    per = {}
    for r in rows():
        per.setdefault(r["anchor"], []).append((float(r["u_n1"]), float(r["gain"])))
    xs = [sum(u for u, _ in v) / len(v) for v in per.values()]
    ys = [sum(g for _, g in v) / len(v) for v in per.values()]
    def _rank(z):
        s = sorted(range(len(z)), key=lambda i: z[i])
        out = [0.0] * len(z)
        for j, i in enumerate(s):
            out[i] = float(j)
        return out
    rx, ry = _rank(xs), _rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = _math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    rho = num / den
    if rho >= 0.6:
        assert "The strongest anchor gives the largest gain" in body, rho
    else:
        assert "The strongest anchor gives the largest gain" not in body, \
            f"C2 reads NO TREND at rho={rho:+.3f}; the withdrawn superlative is back"
        assert f"$\\rho = {rho:+.3f}$" in body, f"Section 6 does not quote rho={rho:+.3f}"


def test_the_row_for_the_strongest_anchor_matches_the_breadth_csv():
    """Was a Table 1 cell until 2026-09-17; Table 1's four headline rows became rows of the forest
    figure and the table came out, so the same band is now checked where the paper plots it."""
    from tests.manuscript import carries_band, _forest
    r = next(x for x in rows() if x["anchor"] == "Comma-7B" and x["judge"] == SCORING_JUDGE)
    g, lo, hi = float(r["gain"]), float(r["gain_lo95"]), float(r["gain_hi95"])
    assert carries_band(g, lo, hi, "experiments.tex"), (g, lo, hi)
    # and it is still the largest gain the anchor sweep produces, which is why it is quoted
    anchors = [x for x in _forest() if x[1][1] is not None and "$n=8$" in x[0]]
    assert max(anchors, key=lambda x: x[1][0])[0].startswith("Comma-7B,"), anchors


def test_the_abstract_claims_the_anchor_count_the_csv_supports():
    """The abstract says selection repeats "at N of M anchors in three families". The claim is
    about anchors that GAIN on the registered scorer -- six are measured and four gain -- and the
    families are the families of those four.

    The denominator is part of the claim. Until 2026-09-17 the abstract said "four anchors in three
    families ... though two intervals include zero", and the caveat attached to nothing a reader
    could see: all four quoted anchors exclude zero by construction, and the two that do not are
    the two the sentence never mentions. Stating M is both shorter and honest, so the guard pins
    it: writing the numerator alone, or either count wrong, fails here."""
    from tests.manuscript import tex
    abstract = open(tex("iclr_2027.tex"), encoding="utf-8").read().replace("\n", " ")
    fams = {"TinyComma-1.8B (audited)": "Comma", "Comma-7B": "Comma",
            "Comma-7B (1T tokens)": "Comma", "Pleias-1.2B": "Pleias",
            "Pleias-3B": "Pleias", "KL3M-1.7B": "KL3M"}
    gaining = {r["anchor"] for r in rows()
               if r["judge"] == SCORING_JUDGE and float(r["gain_lo95"]) > 0}
    assert set(fams) == {r["anchor"] for r in rows()}, "an anchor has no family declared"
    words = ["", "one", "two", "three", "four", "five", "six"]
    n, total = len(gaining), len(fams)
    claim = f"{words[n]} of {words[total]} anchors in three families"
    assert claim in abstract, (claim, sorted(gaining), sorted(fams))
    assert len({fams[a] for a in gaining}) == 3, sorted(gaining)
