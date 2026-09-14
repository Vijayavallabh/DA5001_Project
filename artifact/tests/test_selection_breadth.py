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
    6.8%. Dropping those prompts must leave the gain essentially unchanged, or the headline means
    something much weaker than the paper says."""
    n = 0
    for x in rows():
        if x["gain_nonempty"] == "":
            continue
        n += 1
        assert abs(float(x["gain"]) - float(x["gain_nonempty"])) < 0.01, x
    assert n >= 2, "the sensitivity check produced nothing to check"


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
    assert len(ANCHORS) == 4
    for label, tag, model, gen_dir in ANCHORS:
        assert gen_dir.startswith("output/phase5/"), (label, gen_dir)
        assert (tag == "") == ("audited" in label)
        assert model.count("/") == 1, model


def test_section6_quotes_the_four_anchor_gains_from_the_breadth_csv():
    """Section 6's breadth paragraph names three gains. They round from selection_breadth.csv on
    the registered scorer, once, and the claim 'two of three exclude zero' has to be true of it."""
    from tests.manuscript import tex
    body = open(tex("sections/experiments.tex"), encoding="utf-8").read().replace("\n", " ")
    new = [r for r in rows() if "audited" not in r["anchor"]
           and r["judge"] == SCORING_JUDGE]
    assert len(new) == 3, [r["anchor"] for r in new]
    for r in new:
        g, lo, hi = (float(r["gain"]), float(r["gain_lo95"]), float(r["gain_hi95"]))
        assert f"${g:+.3f}$ $[{lo:+.3f}, {hi:+.3f}]$" in body, (r["anchor"], g, lo, hi)
    excl = sum(1 for r in new if float(r["gain_lo95"]) > 0)
    assert excl == 2, excl
    assert "two of three exclude zero on the pre-registered scorer" in body
    best = max(new, key=lambda r: float(r["gain"]))
    assert "Comma-7B" in best["anchor"], best["anchor"]
    assert "The strongest anchor gives the largest gain" in body


def test_the_table_row_for_the_strongest_anchor_matches_the_breadth_csv():
    from tests.manuscript import tex
    body = open(tex("sections/experiments.tex"), encoding="utf-8").read().replace("\n", " ")
    r = next(x for x in rows() if x["anchor"] == "Comma-7B" and x["judge"] == SCORING_JUDGE)
    g, lo, hi = float(r["gain"]), float(r["gain_lo95"]), float(r["gain_hi95"])
    assert f"${g:+.3f}$ & $[{lo:+.3f}, {hi:+.3f}]$" in body, (g, lo, hi)


def test_the_abstract_claims_the_anchor_count_the_csv_supports():
    from tests.manuscript import tex
    abstract = open(tex("iclr_2027.tex"), encoding="utf-8").read().replace("\n", " ")
    n = len({r["anchor"] for r in rows()})
    word = ["", "one", "two", "three", "four", "five"][n]
    assert f"{word} anchors in three families" in abstract, (n, word)
    fams = {"TinyComma-1.8B (audited)": "Comma", "Comma-7B": "Comma",
            "Pleias-1.2B": "Pleias", "KL3M-1.7B": "KL3M"}
    assert len({fams[r["anchor"]] for r in rows()}) == 3
