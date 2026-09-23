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
        # scoped: `not a mechanism` also appears in the workload paragraph of the same file. Scoped
        # to the paragraph's own end, not to a character count: a fixed 2,600-character window
        # failed on 2026-09-23 when a registered sentence added to this paragraph pushed the
        # concession 21 characters past it, with the concession untouched (caution (ar)).
        i = txt.find("went untested")
        end = txt.find(r"\textbf{One disclosure the ladder surfaced", i)
        assert end > i, "the paragraph's end marker moved; rescope this guard"
        assert "not a mechanism" in txt[i:end], \
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


def test_the_two_judges_are_reported_as_agreeing_on_shape_and_differing_on_the_crossing():
    """The first version of this paragraph said the decay was one judge's, read off three rungs
    that all lie past the second judge's own crossing. Adding the rung it was still confirming on
    inverted that. What is guarded is the corrected claim, rebuilt from the second judge's CSV:
    both judges confirm the weakest opponent, neither confirms the strongest, and their crossings
    differ.

    If a future pass makes them agree on the crossing, or makes one stop confirming everywhere,
    the guard fails and says to revisit the wording rather than quietly keeping the stronger
    sentence (caution (ao))."""
    import csv as _csv
    import os as _os
    f = _os.path.join(ROOT, "results", "opponent_ladder_judgeC.csv")
    assert _os.path.exists(f), "the second judge's ladder is gone; the correction is unsourced"
    jc = sorted(_csv.DictReader(open(f, encoding="utf-8")), key=lambda r: float(r["strength"]))
    assert len(jc) >= 4, f"{len(jc)} rungs; the four-rung reading needs four"

    def cross(side):
        conf = [r for r in jc if float(r[f"lo_judge{side}"]) > 0]
        return max((float(r["strength"]) for r in conf), default=None)

    cb, cc = cross("B"), cross("C")
    assert cb is not None and cc is not None, \
        "a judge now confirms nothing; the agreement sentence must be revisited"
    assert cc < cb, \
        f"judge C's crossing ({cc}) is no longer earlier than judge B's ({cb}); revisit"
    # both must confirm the weakest rung, which is what 'agree on the shape' means here
    assert float(jc[0]["lo_judgeB"]) > 0 and float(jc[0]["lo_judgeC"]) > 0, \
        "the two judges no longer agree at the weakest opponent"
    assert float(jc[-1]["lo_judgeB"]) <= 0 and float(jc[-1]["lo_judgeC"]) <= 0, \
        "a judge now confirms the strongest opponent; the shape claim is stale"

    txt = body(SEC)
    assert "agrees on the shape and crosses earlier" in txt, \
        "the appendix no longer reports the two judges as agreeing on shape"
    assert "that was wrong" in txt, (
        "the appendix dropped the record that we first read this on three rungs and got it "
        "backwards -- the correction is the part a length edit would take")
    assert f"up to ${cb}$" in txt, f"judge B's crossing {cb} left the paragraph"
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
    assert "rather than the uniform one we had\npredicted".replace("\n", " ") in " ".join(txt.split()), \
        "the correction to our own pre-stated direction argument was cut"
    for x in shifts:
        assert f"${x:+.3f}$" in txt, f"the shift {x:+.3f} left the appendix"
