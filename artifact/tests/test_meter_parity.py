"""feat-159: the Program Chairs' evaluation-parity objection, answered by measurement.

Every claim the scoring log and the manuscript make about this arm is rebuilt here from
results/meter_parity.csv. The headline is a NEGATIVE result for the objection and a positive one
for us, which is exactly the shape caution (ag) says gets quietly strengthened later, so the guards
pin the concessions too: two arms are INVALID, the meter wins on level, and the win is at a budget
this paper's own measurement calls vacuous.
"""
import csv
import math
import os

from tests.manuscript import ROOT

CSV = os.path.join(ROOT, "results", "meter_parity.csv")
VAC = os.path.join(ROOT, "results", "tqa_vacuity_summary.csv")
T_MAX = 24


def rows():
    assert os.path.exists(CSV), f"{CSV} missing; this guard must not pass by never running"
    return list(csv.DictReader(open(CSV, encoding="utf-8")))


def cell(arm, n):
    for r in rows():
        if r["arm"] == arm and r["n"] and int(r["n"]) == n:
            return r
    return None


def test_the_reward_model_does_not_transfer_to_the_meter():
    """B1, and the whole answer to the objection. The same scorer, same call, same pass: the
    anchor's draws gain and the meter's do not."""
    for arm in ("metered k=3", "metered k=20"):
        r = cell(arm, 16)
        assert r is not None, f"{arm} has no n=16 row"
        lo, hi = float(r["gain_lo95"]), float(r["gain_hi95"])
        assert lo <= 0 <= hi, f"{arm} no longer reads NO EFFECT: [{lo}, {hi}]"
    s = cell("selection (reward)", 16)
    assert float(s["gain_lo95"]) > 0, (
        f"selection's gain no longer excludes zero: {s['gain_lo95']}")
    assert float(s["gain"]) > 0.05, s["gain"]


def test_the_two_invalid_arms_stay_recorded_as_invalid():
    """They were the near-controls. A later pass must not quietly drop the rows that say so."""
    failed = {r["arm"] for r in rows() if r["gate"] == "FAIL-G3"}
    assert failed == {"metered k=0.5", "metered k=1"}, failed
    for r in rows():
        if r["gate"] == "FAIL-G3":
            assert not (r.get("gain") or "").strip(), (
                f"{r['arm']} failed its gate and carries a band anyway; the registration "
                f"forbids computing one")


def test_the_meter_wins_on_level_and_only_at_a_vacuous_budget():
    """B2 both ways. The concession (it wins) and the qualifier (the budget is vacuous) are one
    sentence in the paper and both halves are pinned, so neither can be dropped alone."""
    met = [r for r in rows() if r["arm"].startswith("metered") and (r.get("acc") or "").strip()]
    sel = [r for r in rows() if r["arm"] == "selection (reward)" and (r.get("acc") or "").strip()]
    bm = max(met, key=lambda r: float(r["acc"]))
    bs = max(sel, key=lambda r: float(r["acc"]))
    assert float(bm["acc"]) > float(bs["acc"]), "the meter no longer wins on level"
    wins = [r for r in met if float(r["acc"]) > float(bs["acc"])]
    vac = {float(r["k"]): float(r["vacuous_frac"]) for r in csv.DictReader(open(VAC, encoding="utf-8"))}
    for r in wins:
        assert vac.get(float(r["k"]), 0.0) >= 0.5, (
            f"the meter now wins at k={r['k']}, which is NOT a vacuous budget; the reading is "
            f"PARITY MATTERS and the appendix must say so")


def test_the_budgets_add():
    """The arm's whole point: best-of-n over a decoder with budget K certifies K + log n, not
    log n. If this stops holding the sentence about composition is wrong."""
    r = cell("metered k=20", 2)
    assert abs((20 * T_MAX + math.log(2)) - 480.693) < 1e-3
    assert float(r["acc"]) > 0, r


def test_the_refuted_clause_is_recorded_as_withdrawn():
    """caution (w): a defect in our own specification is recorded, not quietly repaired. Branch
    one's consequence claims a reward model helps the meter too, which B1 refutes."""
    log = open(os.path.join(ROOT, "results", "onset_prediction_meter_parity.md"),
              encoding="utf-8").read()
    body = log.split("\n## Scoring log", 1)[1]
    flat = " ".join(body.split())
    assert "withdrawn" in flat, "the scoring log no longer records the withdrawn clause"
    assert "PARITY AT A VACUOUS BUDGET" in flat and "PARITY MATTERS" in flat, (
        "both labels must stay in the log so the softer one is not a post-hoc choice")


def test_the_appendix_table_is_the_csv():
    """caution (j)/(al): every cell the paper prints must round from the CSV, once, and a number
    that moved into a table keeps a guard that can read it."""
    from tests.manuscript import body
    t = body("appendix_selection.tex")
    i = t.find("Giving the \\emph{metered} decoder the same reward model")
    assert i > 0, "the parity paragraph is gone from the appendix"
    para = t[i:i + 3000]
    for arm, n in (("metered k=3", 16), ("metered k=20", 16), ("selection (reward)", 16)):
        r = cell(arm, n)
        for key, fmt in (("gain", "{:+.3f}"), ("gain_lo95", "{:+.3f}"), ("gain_hi95", "{:+.3f}")):
            v = fmt.format(float(r[key]))
            assert f"${v}$" in para or v in para, (
                f"{arm} {key} = {v} is not printed in the parity paragraph")
        assert f"${float(r['acc']):.3f}$" in para, f"{arm} acc {r['acc']} missing"


def test_the_concession_and_its_qualifier_are_both_in_the_paper():
    """The meter wins; it wins only at a vacuous budget. Neither half may be dropped alone --
    caution (ag) says a length edit takes the concession first, and caution (an) says a guard on a
    phrase must be scoped to the sentence it is about."""
    from tests.manuscript import body
    t = body("appendix_selection.tex")
    i = t.find("Giving the \\emph{metered} decoder the same reward model")
    para = t[i:i + 3000]
    assert "meter does win on level" in para, "the concession that the meter wins is gone"
    assert "vacuous" in para, "the qualifier that its winning budget is vacuous is gone"
    met = [r for r in rows() if r["arm"].startswith("metered") and (r.get("acc") or "").strip()]
    sel = [r for r in rows() if r["arm"] == "selection (reward)" and (r.get("acc") or "").strip()]
    bm = max(met, key=lambda r: float(r["acc"]))
    bs = max(sel, key=lambda r: float(r["acc"]))
    assert f"${float(bm['acc']):.3f}$" in para and f"${float(bs['acc']):.3f}$" in para, (
        "the two levels the concession compares are not both printed")
