"""The opponent-strength ladder, checked against its own CSV and its own claims.

feat-175 replaced a two-point anecdote in `app:h2hrepeat` with a five-point measured axis. The
number guards are the easy half; what this also pins is the two claims ABOUT the set (caution
(ai)), the verdict words (caution (av)), and the fact that the registered test went UNTESTED --- a
result that is easy to quietly upgrade into the exploratory one beside it.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import ROOT, body, carries_band  # noqa: E402

SEC = "appendix_selection.tex"


def rows():
    return list(csv.DictReader(open(os.path.join(ROOT, "results", "opponent_ladder.csv"),
                                    encoding="utf-8")))


def _ladder_block():
    """The tabular only. Every check below is scoped to it, because the prose around it repeats
    several of the same literals -- a guard satisfied by a different occurrence of its number is
    not guarding its row (caution (an)), and the first version of this file had three."""
    txt = body(SEC)
    i = txt.find("opponent & strength & paired")
    assert i > 0, "the opponent-strength table is gone"
    j = txt.find("\\end{tabular}", i)
    return txt[i:j]


def test_every_rung_is_printed_with_its_band_and_its_strength():
    r = rows()
    assert len(r) == 5, f"the ladder has {len(r)} rungs; the paragraph prints five"
    block = _ladder_block()
    for x in r:
        g, lo, hi = float(x["d3"]), float(x["d3_lo95"]), float(x["d3_hi95"])
        assert carries_band(g, lo, hi, SEC), f"{x['opponent']}'s band left the appendix"
        assert f"${float(x['strength']):.3f}$" in block, \
            f"{x['opponent']}'s strength {x['strength']} left the TABLE"


def test_the_verdict_words_match_the_intervals():
    """Caution (av): a verdict is a string that outlives its number. Two rungs are confirmed and
    three unresolved; if an interval moves, the word beside it has to."""
    r = rows()
    conf = [x for x in r if float(x["d3_lo95"]) > 0]
    unres = [x for x in r if float(x["d3_lo95"]) <= 0 <= float(x["d3_hi95"])]
    assert len(conf) == 2 and len(unres) == 3, \
        f"the ladder now reads {len(conf)} confirmed / {len(unres)} unresolved; revisit the table"
    block = _ladder_block()
    assert block.count("& confirmed \\\\") == 2 and block.count("& unresolved \\\\") == 3, \
        "the table's verdict column no longer matches its own intervals"


def test_the_crossing_is_stated_between_the_two_rungs_that_bracket_it():
    """A claim ABOUT the set, so it is rebuilt from the set (caution (ai))."""
    r = sorted(rows(), key=lambda x: float(x["strength"]))
    pos = [x for x in r if float(x["d3"]) > 0]
    neg = [x for x in r if float(x["d3"]) < 0]
    assert pos and neg, "the series no longer changes sign; the crossing claim is stale"
    lo_s, hi_s = float(pos[-1]["strength"]), float(neg[0]["strength"])
    txt = body(SEC)
    assert f"between ${lo_s:.3f}$ and ${hi_s:.3f}$" in txt, \
        f"the paragraph does not say the crossing is between {lo_s:.3f} and {hi_s:.3f}"


def test_the_registered_test_is_reported_as_untested():
    """H1 required an opponent WEAKER than the committed one and none was. The exploratory decay
    sits in the same paragraph and is the easy thing to promote; the paper must not."""
    r = rows()
    ref = next(x for x in r if x["status"] == "committed")
    below = [x for x in r if x["status"] == "new"
             and float(x["strength"]) < float(ref["strength"])]
    txt = body(SEC)
    if below:
        assert "went untested" not in txt, \
            f"{[x['opponent'] for x in below]} IS below the threshold; H1 was testable after all"
    else:
        assert "went untested" in txt, \
            "no opponent is weaker than the committed one, so the registered test went untested " \
            "and the appendix must say so rather than leaning on the exploratory decay"
        # scoped: `not a mechanism` also appears in the workload paragraph of the same file
        i = txt.find("went untested")
        assert "not a mechanism" in txt[i:i + 2600], \
            "the paragraph no longer says the decay is a shape and not a shown mechanism"


def test_the_not_by_length_claim_is_true_of_the_csv():
    """The surprise -- a 0.5B opponent stronger than an 8B one -- is only interesting because it
    is not the usual LLM-judge length effect, so that has to keep being true."""
    r = rows()
    ref = next(x for x in r if x["status"] == "committed")
    others = [x for x in r if x["status"] != "committed"]
    assert all(float(ref["mean_words"]) > float(x["mean_words"]) for x in others), \
        "the committed opponent is no longer the longest; the not-by-length sentence is stale"
    assert all(float(ref["strength"]) < float(x["strength"]) for x in others), \
        "the committed opponent is no longer the weakest; the H1 NOT TESTED reading is stale"
    txt = body(SEC)
    assert f"${float(ref['mean_words']):.1f}$ words" in txt, \
        "the length figure that rules out the obvious explanation was cut"
    # and the RANGE it is set against must round from the same column. It was typed from a partial
    # run and read $114.6$--$133.3$ where the CSV says 109.45--133.25 (caution (j)).
    lo = min(float(x["mean_words"]) for x in others)
    hi = max(float(x["mean_words"]) for x in others)
    assert f"${lo:.1f}$--${hi:.1f}$" in txt, \
        f"the appendix's length range does not round from the CSV (${lo:.1f}$--${hi:.1f}$)"


def test_the_generator_disclosure_is_made():
    """Caution (at): two arms compared must have come from the same pipeline. They did not, the
    CSV records which is which, and the paper says so."""
    r = rows()
    gens = {x["generator"] for x in r}
    txt = body(SEC)
    # scoped to the disclosure sentence: `blocklist` also names the MemFree baseline elsewhere
    if len(gens) > 1:
        i = txt.find("One disclosure the ladder surfaced")
        assert i > 0, "the generator disclosure was deleted"
        near = txt[i:i + 700]
        assert "blocklist" in near and "h1.py" in near and "one prompt at a time" in near, \
            f"the ladder mixes generators {sorted(gens)} and the disclosure no longer names them"
    else:
        assert "One disclosure the ladder surfaced" not in txt, \
            "the generators now agree; the disclosure paragraph should be revisited"


def test_the_decay_is_attributed_to_one_judge_because_the_other_does_not_see_it():
    """The post-hoc judge refuted the shape, and the appendix paragraph had already been written
    from the judge-B ladder. This is caution (af) in the small: a claim corrected in one place has
    to stay corrected, and the tempting edit later is to drop the qualifier and keep the decay.

    Conditioned on the second judge's own CSV, so if judge C ever does reproduce the ordering the
    guard fails and says to revisit the wording rather than quietly permitting the stronger claim
    (caution (ao))."""
    import csv as _csv
    import os as _os
    f = _os.path.join(ROOT, "results", "opponent_ladder_judgeC.csv")
    assert _os.path.exists(f), "the post-hoc judge's ladder is gone; the correction is unsourced"
    jc = list(_csv.DictReader(open(f, encoding="utf-8")))
    assert len(jc) >= 3
    conf_c = [r for r in jc if float(r["lo_judgeC"]) > 0]
    conf_b = [r for r in jc if float(r["lo_judgeB"]) > 0]
    txt = body(SEC)
    if conf_c or len(conf_c) >= len(conf_b):
        assert "the decay is one judge's" not in txt.lower(), \
            "judge C now confirms a rung; the attribution sentence must be revisited"
    else:
        assert "the decay is one judge's" in txt.lower(), \
            "the appendix dropped the fact that a second judge does not reproduce the decay"
        assert f"${len(conf_c)}$ of ${len(jc)}$ confirmed" in txt, \
            f"the appendix no longer prints judge C's {len(conf_c)} of {len(jc)} confirmed"
        for r in jc:
            g, lo, hi = (float(r["d3_judgeC"]), float(r["lo_judgeC"]), float(r["hi_judgeC"]))
            assert carries_band(g, lo, hi, SEC), \
                f"judge C's band at strength {r['strength']} left the appendix"


def test_the_pre_stated_direction_argument_is_reported_as_only_half_right():
    """We wrote before the run that judge C would make the decay look STRONGER. It did shift every
    rung down and the shift was strength-dependent, which flattens instead. A paper that keeps the
    prediction and drops the correction is worse than one that never predicted."""
    import csv as _csv
    import os as _os
    jc = list(_csv.DictReader(open(_os.path.join(
        ROOT, "results", "opponent_ladder_judgeC.csv"), encoding="utf-8")))
    jc.sort(key=lambda r: float(r["strength"]))
    shifts = [float(r["shift"]) for r in jc]
    assert all(x < 0 for x in shifts), \
        "judge C no longer shifts every rung down; the direction sentence is stale"
    assert abs(shifts[0]) > abs(shifts[-1]), \
        "the shift is no longer largest at the weakest opponent, which is what flattens the series"
    txt = body(SEC)
    assert "flattens a decaying series rather than steepening it" in txt, \
        "the correction to our own pre-stated direction argument was cut"
    for x in shifts:
        assert f"${x:+.3f}$" in txt, f"the shift {x:+.3f} left the appendix"
