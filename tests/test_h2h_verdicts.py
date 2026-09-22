"""No committed head-to-head verdict may disagree with its own interval.

Caution (av): a verdict is a string that outlives the number under it. `order_averaged_h2h.py`
called ANY negative point estimate REVERSAL REFUTED regardless of whether the interval straddled
zero, two committed rows were mislabelled for weeks, and the label reached manuscript claims --- a
scoring log had even caught it in prose while the script went on producing it.

The repair was one line. This is the invariant that stops it coming back anywhere: every D3 and D5
row in every committed pass is re-derived from its own `lo95`/`hi95` and compared with the
`reading` it carries. Both verdict vocabularies are covered, because a second scheme is exactly
where the next drift will live.
"""
import csv
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (quantity prefix) -> (label when lo > 0, when hi < 0, when it straddles)
SCHEMES = {
    "D3": ("REVERSAL CONFIRMED", "REVERSAL REFUTED", "REVERSAL UNRESOLVED"),
    "D5": ("INCUMBENT WINS", "CHALLENGER WINS", "TIE"),
}


def _rows():
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "order_averaged_h2h*.csv"))):
        if "per_prompt" in os.path.basename(f):
            continue
        for r in csv.DictReader(open(f, encoding="utf-8")):
            yield os.path.basename(f), r


def test_every_committed_verdict_agrees_with_its_own_interval():
    seen = 0
    for name, r in _rows():
        key = r["quantity"][:2]
        if key not in SCHEMES:
            continue
        try:
            lo, hi = float(r["lo95"]), float(r["hi95"])
        except ValueError:
            continue
        pos, neg, mid = SCHEMES[key]
        want = pos if lo > 0 else neg if hi < 0 else mid
        got = r["reading"].strip()
        assert got == want, (
            f"{name} {key}: interval [{lo}, {hi}] is {want}, the file says {got!r}")
        seen += 1
    assert seen >= 40, f"only {seen} verdict rows found; the glob stopped matching"


def test_the_two_vocabularies_do_not_overlap():
    """A row must be unambiguously in one scheme. If a label ever appears under both prefixes the
    check above would pass a D5 row carrying a D3 verdict."""
    d3, d5 = set(SCHEMES["D3"]), set(SCHEMES["D5"])
    assert not (d3 & d5)
    for name, r in _rows():
        got = r["reading"].strip()
        key = r["quantity"][:2]
        if key == "D3":
            assert got not in d5, f"{name}: a D3 row carries a D5 verdict"
        elif key == "D5":
            assert got not in d3, f"{name}: a D5 row carries a D3 verdict"
