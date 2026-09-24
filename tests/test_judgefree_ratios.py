"""Ratios between the two judge-free rules move with n, so every quoted one must name its n.

The closing said "the scorer binds, gaining 3.4x less than majority vote on GSM8K" with no n. The
ratio over the grid is 3.00, 3.32, 5.00, 3.77, 3.36 at n = 4, 8, 16, 32, 64 -- so a reviewer
recomputing it at the matched n=32 the appendix uses gets 3.77 and at n=16 gets 5.00. Same class as
the rho defect: the number is right and a reader cannot tell which quantity it is.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import body  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _gains(name):
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results", name), encoding="utf-8")))
    mv = {int(float(r["n"])): float(r["gain"]) for r in rows if r["arm"].startswith("majority")}
    pw = {int(float(r["n"])): float(r["gain"]) for r in rows if r["arm"].startswith("pointwise")}
    return mv, pw


def test_the_gsm8k_ratio_is_quoted_at_the_n_it_is_computed_at():
    mv, pw = _gains("selection_verifiable_comma7b.csv")
    assert abs(mv[64] / pw[64] - 3.4) < 0.05, (mv[64] / pw[64], "n=64 ratio moved")
    assert abs(mv[32] / pw[32] - 3.77) < 0.01, (mv[32] / pw[32], "the appendix's matched n=32 ratio moved")
    txt = body("iclr_closing.tex")
    assert "$3.4\\times$ less than majority vote on GSM8K at $n=64$" in txt, \
        "the closing's GSM8K ratio lost the n it is computed at; it reads 3.77 at n=32 and 5.00 at n=16"


def test_the_ratio_really_does_move_with_n_which_is_why_the_n_is_required():
    mv, pw = _gains("selection_verifiable_comma7b.csv")
    rs = [mv[n] / pw[n] for n in (4, 8, 16, 32, 64)]
    assert max(rs) - min(rs) > 1.0, (rs, "the ratio is now flat in n; the qualifier could be revisited")


def test_the_closing_levels_reproduce():
    """TriviaQA: selection 0.190 at n=64, metered 0.618 only at k=20, certified 480 nats."""
    sel = list(csv.DictReader(open(os.path.join(ROOT, "results", "verifiable_metered_tqa.csv"),
                                   encoding="utf-8")))
    s64 = next(r for r in sel if r["mechanism"].startswith("selection") and r["arm"] == "n=64")
    assert abs(float(s64["acc"]) - 0.190) < 5e-4, s64["acc"]
    m20 = next(r for r in sel if r["mechanism"].startswith("metered") and r["arm"] == "k=20")
    assert abs(float(m20["acc"]) - 0.618) < 5e-4, m20["acc"]
    assert abs(float(m20["certificate_nats"]) - 480.0) < 1e-9, m20["certificate_nats"]
    # it IS the risky model there -- identical accuracy to the k=-1 baseline
    mm1 = next(r for r in sel if r["mechanism"].startswith("metered") and r["arm"] == "k=-1")
    assert float(m20["acc"]) == float(mm1["acc"]), (m20["acc"], mm1["acc"])
    txt = body("iclr_closing.tex")
    for v in ("$0.618$", "$0.190$", "$480$ nats", "$24$-token"):
        assert v in txt, f"the closing lost {v}"


def test_the_majority_vote_dominance_claim_matches_its_whole_grid():
    """'four draws of it beat all 28 reward cells on either task' -- checked against every cell.

    Added 2026-09-18, replacing 'beats every reward cell at any price'. That phrasing had a natural
    strong reading under which it is FALSE: majority vote over n=2 gains +0.000 on GSM8K, below the
    best reward cell's +0.066, because two samples have no majority. The claim that survives is
    sharper and states its denominator (caution (ag)): the n=4 cell already clears the maximum over
    all 28 reward cells -- four scorers by seven n -- on both checkable tasks.

    Caution (ai): the claim ABOUT a set of numbers is what nothing checks. This rebuilds both sets.
    """
    import csv as _csv
    import glob as _glob
    import os as _os
    from tests.manuscript import body
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    cells = {"GSM8K": {"majority": {}, "reward": [], "reward_by": []},
             "TriviaQA": {"majority": {}, "reward": [], "reward_by": []}}
    # SCOPED to the arm the sentence is about (caution (an)/(ao)). The 28 cells are FOUR SCORERS
    # by seven n at the audited 7B anchor -- comma7b plus its three qwen scorer scales -- on the two
    # open-ended tasks. This was a GLOB WITH EXCLUSIONS twice and was broken twice by arms that had
    # nothing to do with the claim: feat-137's second ANCHOR (selection_verifiable_comma1t.csv) made
    # the denominator 35 on 2026-09-20, and feat-157's CoTaEval arm at a 14B scorer
    # (selection_verifiable_cta14_comma7b.csv) made it 9 the same evening. A denylist has to be
    # extended by every future arm and silently admits the one nobody thought of, so the membership
    # is now ENUMERATED: any new file is out by default and a real addition to the grid has to say so
    # here. (feat-149's MMLU is a four-way forced choice with a 0.25 floor and its own guard in
    # test_incumbents.py; it is a different grid, not a fifth scorer.)
    # feat-158 added a FIFTH scorer (14B) on both tasks. Its registration fixed, before any number
    # was seen, that this claim is re-derived over all five and the prose updated to the new
    # denominator whatever the answer -- and WITHDRAWN rather than restricted back to four if it
    # no longer holds. Extended in the same commit as the prose, as registered.
    SCORERS = ("", "_qwen05b", "_qwen15b", "_qwen3b", "_qwen14b")
    files = [_os.path.join(root, "results", f"selection_verifiable_{pre}comma7b{sc}.csv")
             for pre in ("", "tqa_") for sc in SCORERS]
    missing = [_os.path.basename(f) for f in files if not _os.path.exists(f)]
    assert not missing, f"the 4x2 grid this claim quotes is incomplete: {missing}"
    assert len(files) == 10, f"expected 5 scorers x 2 tasks, got {[_os.path.basename(f) for f in files]}"
    for f in files:
        task = "TriviaQA" if "_tqa" in _os.path.basename(f) else "GSM8K"
        for r in _csv.DictReader(open(f, encoding="utf-8")):
            if not (r.get("gain") or "").strip():
                continue
            g, n = float(r["gain"]), int(float(r["n"]))
            if "major" in r["arm"].lower():
                cells[task]["majority"][n] = g
            else:
                cells[task]["reward"].append(g)
                # remember WHICH scorer produced each cell, so the replacement claim ("dominates
                # every scorer up to 7.6B, not 14B") can be re-derived rather than asserted.
                base = _os.path.basename(f)
                sc = next((x for x in SCORERS if x and base.endswith(x + ".csv")), "")
                cells[task]["reward_by"].append((g, sc, n))

    # WITHDRAWN 2026-09-21, exactly as feat-158 registered BEFORE any number was seen. With the
    # fifth scorer the "four draws beat all 28 reward cells" claim is false on BOTH tasks:
    # majority vote at n=4 gives +0.0720 on GSM8K against a best reward cell of +0.2140 (14B,
    # n=64), and +0.0200 on TriviaQA against +0.0780. The registration forbade restricting it back
    # to the four scorers that happen to support it, so the sentence is gone from selection.tex
    # and this guard now pins what replaced it.
    for task, c in cells.items():
        assert len(c["reward"]) == 35, (task, len(c["reward"]), "the denominator is 35")
        best_reward = max(c["reward"])
        assert c["majority"][4] <= best_reward, (
            task, "majority vote at n=4 clears the whole 35-cell grid again, so the withdrawn "
                  "sentence may be reinstatable -- re-derive it and update selection.tex in the "
                  "same commit rather than leaving the weaker claim standing")
        # what the prose says instead, and it must keep being true of the four smaller scorers
        sub = [g for g, sc, _ in c["reward_by"] if sc != "_qwen14b"]
        assert len(sub) == 28, (task, len(sub))
        assert c["majority"][4] > max(sub), (
            task, c["majority"][4], max(sub),
            "majority vote no longer dominates the scorers up to 7.6B either; the replacement "
            "sentence in selection.tex is wrong too")
        # and the overtaking that the new sentence asserts is real
        top14 = max(g for g, sc, _ in c["reward_by"] if sc == "_qwen14b")
        assert top14 > max(c["majority"].values()) or task == "GSM8K", (task, top14)

    # v10 (2026-09-24) moved v9 Section 2's cost paragraph, which carried this sentence, into
    # Section 4.5 (experiments.tex); both files are read so the guard follows the paragraph.
    txt = body("selection.tex", "experiments.tex")
    assert "four draws of it beat all $28$ reward cells" not in txt, \
        "the withdrawn dominance claim is back in the body; feat-158 registered its withdrawal"
    assert "up to $7.6$B --- but not at $14$B" in txt, \
        "the replacement judge-free dominance claim was reworded or trimmed"


def test_the_four_draws_rule_is_ANCHOR_SPECIFIC_and_the_body_says_so():
    """feat-137 measured the same two rules at a second anchor, and 'four draws' does not hold there.

    At Comma-7B-1T majority vote gains +0.050 at n=4 against its scorer's best cell of +0.060, so
    four draws do NOT beat the reward arm; eight do (+0.124). The body's sentence was written before
    any second anchor existed and generalised over none, which is caution (ao)'s shape exactly: an
    unscoped claim about a set of numbers that a later measurement falsifies in general while
    leaving it true of the arm it was drawn from.

    Guarded both ways (caution (ao)): if the 1T anchor ever does clear at n=4, this fails and says to
    revisit the wording rather than quietly permitting the stronger claim again.
    """
    import csv as _csv
    import os as _os
    from tests.manuscript import body
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    maj, rew = {}, []
    with open(_os.path.join(root, "results", "selection_verifiable_comma1t.csv"),
              encoding="utf-8") as fh:
        for r in _csv.DictReader(fh):
            if not (r.get("gain") or "").strip():
                continue
            g, n = float(r["gain"]), int(float(r["n"]))
            (maj.__setitem__(n, g) if "major" in r["arm"].lower() else rew.append(g))
    best = max(rew)
    assert maj[4] <= best, (
        f"at the 1T anchor n=4 majority ({maj[4]}) now clears its scorer's best ({best}); the "
        f"body's scoping may be revisited, deliberately")
    assert maj[8] > best, (maj[8], best)

    # The prose half of this guard is WITHDRAWN DELIBERATELY, 2026-09-21, which is what its own
    # message demanded. feat-158 added a fifth scorer and its registration -- fixed before any
    # number was seen -- required the "four draws beat all 28 reward cells" sentence to be
    # withdrawn rather than restricted once it no longer held. It does not hold: on GSM8K majority
    # vote at n=4 gives +0.0720 against a best reward cell of +0.2140. So there is no longer a
    # sentence to scope, and the data above is what survives and is still worth pinning.
    txt = " ".join(body("selection.tex").split())
    assert "four draws" not in txt, (
        "the withdrawn 'four draws' claim is back in the body. feat-158 registered its withdrawal "
        "and the 35-cell grid still refutes it -- re-derive before reinstating anything.")
