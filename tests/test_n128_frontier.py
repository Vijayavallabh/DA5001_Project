"""feat-129 Arm A: SATURATED BY 64, and the two instrument facts that reading it depended on.

The registered band is the PAIRED g(128) - g(64) within one pass: +0.0140 [-0.0180, +0.0460],
interval containing zero. Three things this pins.

1. The withdrawn claim stays withdrawn. "still climbing at n=64" was the committed consequence of
   an interval containing zero, and caution (ag) says a concession is the first thing a length edit
   deletes.

2. The reproduction that matters is at the REWARD level, not the judged level. 32,000 of 32,000
   rewards at ranks 0-63 are bit-identical between the committed pool and the 128-draw pool. The
   first version of the gate compared judged gains across passes and could never have passed --
   cautions (e) and (m) both say why -- so this test pins the check that is actually sound.

3. The judged level of an arm depends on WHICH OTHER ARMS are in the sweep. `distinct` is the sorted
   set of served candidates across the grid and one rng.random() is drawn per item in that order, so
   adding the n=128 arm (1,954 -> 2,219 items) re-rolls the presentation order of nearly every
   shared item -- and this judge is position-dominated. Identical text, different judged level
   (0.435 -> 0.478 at n=1). No number from one pass may be set against a number from the other.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import body  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _csv(name):
    return list(csv.DictReader(open(os.path.join(ROOT, "results", name), encoding="utf-8")))


def test_the_band_rounds_from_its_csv():
    row = next(r for r in _csv("n128_frontier.csv") if r["arm"].startswith("A "))
    assert abs(float(row["gain_diff"]) - 0.014) < 5e-4, row
    assert abs(float(row["lo95"]) - (-0.018)) < 5e-4, row
    assert abs(float(row["hi95"]) - 0.046) < 5e-4, row
    assert float(row["lo95"]) < 0 < float(row["hi95"]), "the interval must contain zero to read SATURATED"
    assert row["verdict"] == "SATURATED BY 64", row
    assert row["reproduction"] == "PASS", row


def test_ranks_0_to_63_really_are_bit_identical():
    """The gate that is sound. If this ever fails the arm is invalid, not merely surprising."""
    k = lambda r: (r["prompt_id"], int(r["rank"]))              # noqa: E731
    old = {k(r): float(r["reward"]) for r in _csv("selection_rewards64.csv")}
    new = {k(r): float(r["reward"]) for r in _csv("selection_rewards128.csv") if int(r["rank"]) < 64}
    assert len(old) == 32000, len(old)
    assert set(old) <= set(new), f"{len(set(old) - set(new))} committed keys absent from the new pool"
    bad = [x for x in old if old[x] != new[x]]
    assert not bad, f"{len(bad)} rewards differ at ranks 0-63"


def test_the_judged_level_moved_on_identical_text_which_is_why_passes_are_not_comparable():
    """Documents the effect so that quoting across passes cannot creep back in unnoticed."""
    def by(name):
        return {(r["judge"], int(float(r["n"]))): r for r in _csv(name)}
    old, new = by("selection_scaling.csv"), by("selection_scaling_n128.csv")
    shared = sorted(set(old) & set(new))
    assert shared, "no shared arms"
    # the generations are the same: mean_words is identical everywhere
    for key in shared:
        assert abs(float(old[key]["mean_words"]) - float(new[key]["mean_words"])) < 5e-4, key
    # and the judged levels are not
    moves = [abs(float(old[key]["u"]) - float(new[key]["u"])) for key in shared]
    assert max(moves) > 0.02, (max(moves), "the judged level no longer moves; revisit the caution")


def test_the_appendix_withdrew_the_still_climbing_claim_and_says_where_it_stops():
    txt = body("appendix_selection.tex")
    assert "still climbing at $n=64$" not in txt, \
        "the withdrawn claim is back; the committed consequence of an interval containing zero"
    assert "still rising where its grid stops" in txt, "the replacement wording was lost"
    assert "$+0.0140$ $[-0.0180, +0.0460]$" in txt, "the measured band was trimmed"
    assert "\\textsc{saturated by 64}" in txt, "the verdict was trimmed"
    assert "where the strongest anchor's ceiling sits is open" in txt, \
        "the scope concession -- Arm A tested TinyComma, not Comma-7B -- was trimmed"
