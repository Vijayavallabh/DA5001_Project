"""feat-130: PARTIAL -- the climb to n=64 holds at three of five anchors, not universally.

Read on the paired g(64)-g(8) under judge B WITHIN each new pass, which never touches the committed
breadth table. Two of the three committed arms ran before the launcher that passes --batch-size 32
existed, so they took h1.py's default of 8; batch size is part of the seed (caution (u)), which
makes their registered reproduction check inapplicable rather than failed. Pleias-3B's committed arm
did use 32 and reproduces bit-exactly -- the positive control for the pipeline.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import body  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECT = {                       # anchor -> (diff, lo, hi, verdict, warrant)
    "pleias12b": (0.036, -0.004, 0.074, "SATURATED BY 8", "REDUCED"),
    "kl3m17b": (0.065, 0.027, 0.103, "CLIMBS", "REDUCED"),
    "pleias3b": (0.003, -0.038, 0.046, "SATURATED BY 8", "FULL"),
}


def _scored():
    p = os.path.join(ROOT, "results", "breadth64_scoring.csv")
    return {r["name"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}


def test_every_band_rounds_from_its_csv_and_the_verdict_follows_from_it():
    rows = _scored()
    assert set(rows) == set(EXPECT), sorted(rows)
    for name, (d, lo, hi, verdict, warrant) in EXPECT.items():
        r = rows[name]
        assert abs(float(r["gain_diff"]) - d) < 5e-4, (name, r["gain_diff"])
        assert abs(float(r["lo95"]) - lo) < 5e-4, (name, r["lo95"])
        assert abs(float(r["hi95"]) - hi) < 5e-4, (name, r["hi95"])
        assert r["verdict"] == verdict, (name, r["verdict"])
        assert r["warrant"] == warrant, (name, r["warrant"])
        # the verdict must be what the interval says, not a label someone typed
        climbs = float(r["lo95"]) > 0
        assert climbs == (verdict == "CLIMBS"), (name, r["lo95"], verdict)
        assert r["overall"] == "PARTIAL"


def test_exactly_one_anchor_climbs_which_is_why_the_reading_is_partial():
    rows = _scored()
    climbing = [n for n, r in rows.items() if float(r["lo95"]) > 0]
    assert climbing == ["kl3m17b"], climbing
    turning = [n for n, r in rows.items() if float(r["hi95"]) < 0]
    assert not turning, turning


def test_the_waiver_is_recorded_as_a_waiver_and_the_control_is_not():
    rows = _scored()
    assert rows["pleias3b"]["reproduction"] == "PASS", \
        "the positive control must actually reproduce, or the whole waiver argument fails"
    for n in ("pleias12b", "kl3m17b"):
        assert rows[n]["reproduction"].startswith("WAIVED"), (n, rows[n]["reproduction"])


def test_the_appendix_states_the_claim_over_exactly_the_anchors_that_carry_it():
    """NARROWED 2026-09-18 by feat-131. The one anchor that climbed did not survive a fresh draw,
    so the committed consequence of DOES NOT REPLICATE applies: the climb to n=64 is established at
    TinyComma and Comma-7B and nowhere else. This test previously asserted the three-anchor
    wording; it now asserts the two-anchor one, and that the three-anchor claim is gone."""
    txt = body("appendix_selection.tex")
    assert "three --- TinyComma, Comma-7B and KL3M-1.7B" not in txt, \
        "the three-anchor claim is back, and feat-131 refuted its third anchor"
    assert "no anchor outside the two already on record\nclimbs reproducibly" in txt.replace(" ", " ") \
        or "no anchor outside the two already on record climbs reproducibly" in txt, \
        "the committed consequence of DOES NOT REPLICATE was softened"
    assert "established at \\textbf{TinyComma and Comma-7B}" in txt, \
        "the claim must name exactly the two anchors that carry it"
    assert "\\textsc{partial}" in txt, "the committed reading was dropped"
    assert "\\textsc{does not replicate}" in txt, "the replication verdict was dropped"
    assert "reduced warrant" in txt, "the warrant concession was trimmed"
    assert "inapplicable" in txt, "the reproduction defect was trimmed"
    assert "$\\mathbf{+0.0650}$ $[+0.0270, +0.1030]$" in txt, "the one climbing band was trimmed"
    assert "$+0.0040$ $[-0.0330, +0.0400]$" in txt, "the replication band was trimmed"


def test_the_paired_read_never_quotes_the_committed_breadth_table():
    """Cross-grid comparison is the error caution (ap) forbids.

    The window is bounded by the NEXT \\paragraph, not by a character count. It was `i : i + 3000`
    until 2026-09-18, when adding two rows to the table pushed the disclaimer past 3000 and the guard
    fired on prose that had not changed. A fixed-width window around a claim is caution (an)'s
    complaint in another dress: it retires itself the first time the paragraph grows.
    """
    txt = body("appendix_selection.tex")
    i = txt.find("Does the climb to the headline")
    assert i != -1, "the paragraph this guard is about is gone"
    j = txt.find("\\paragraph", i + 1)
    seg = txt[i: j if j != -1 else len(txt)]
    assert "none of these numbers is set" in seg, \
        "the appendix must say these numbers are not compared with the committed breadth table"


def test_the_two_retained_anchors_are_read_on_the_same_paired_statistic():
    """The n=64 table rejects three anchors on a paired g(64)-g(8) and keeps two.

    Added 2026-09-18. Until then it kept them on a different statistic -- monotonicity for Comma-7B,
    the n=128 arm for TinyComma -- so 'climbs' meant one thing in the rows that were rejected and
    another in the rows that survived. A reviewer checking why KL3M's +0.0650 is not enough while
    Comma-7B's climb is would have found no comparable number printed. Both are now in the table and
    both are rebuilt here from their own per-prompt CSVs, so neither can drift from its source.
    """
    import csv as _csv
    import random
    import sys as _sys
    _sys.path.insert(0, ROOT)
    from analysis.selection_decoding import boot_mean

    def paired(tag):
        p = os.path.join(ROOT, "results", f"selection_scaling_per_prompt{tag}.csv")
        rows = [r for r in _csv.DictReader(open(p, encoding="utf-8"))
                if "Phi-3.5-mini-instruct" in r["judge"]]
        d = [float(r["u_n64"]) - float(r["u_n8"]) for r in rows]
        g = sum(d) / len(d)
        lo, hi = boot_mean(d, random.Random(20260918))
        return g, lo, hi

    txt = body("appendix_selection.tex")
    for tag, want in (("", (0.0880, 0.0470, 0.1300)), ("_comma7b64", (0.1010, 0.0590, 0.1420))):
        g, lo, hi = paired(tag)
        assert abs(g - want[0]) < 5e-4 and abs(lo - want[1]) < 5e-4 and abs(hi - want[2]) < 5e-4, \
            (tag, g, lo, hi, want)
        printed = f"${want[0]:+.4f}$ $[{want[1]:+.4f}, {want[2]:+.4f}]$"
        assert printed in txt, f"{tag}: the table no longer prints {printed}"
        # and both must exclude zero, or 'climbs' is the wrong word in that row
        assert lo > 0, (tag, lo)
