"""feat-161: the judged scorer ladder does NOT saturate away -- it does not lift either.

Reads scorer_scale_6rung*.csv, NOT scorer_scale*.csv. feat-161's run wrote the canonical names and
silently overwrote the closed feat-117 four-rung arm, whose six committed values the appendix still
quotes; three guards caught it. Two passes of a judged sweep must never share a file -- caution (ap)
forbids comparing their levels, and one file invites exactly that.

Two rungs above the paper's operating point, judged in one six-scorer pass. The result is a
NEGATIVE one for scorer spending and a confirmation of the saturation sentence, so the guards pin
the shape (non-monotone, nothing lifts) AND the honesty (all three readings marginal, the cost).
"""
import csv
import os
import random
import sys

from tests.manuscript import ROOT, body

sys.path.insert(0, ROOT)
CSV = os.path.join(ROOT, "results", "scorer_scale_6rung.csv")
PER = os.path.join(ROOT, "results", "scorer_scale_6rung_per_prompt.csv")
SEC = "appendix_selection.tex"


def level(arm):
    assert os.path.exists(CSV), f"{CSV} missing; this guard must not pass by never running"
    for r in csv.DictReader(open(CSV, encoding="utf-8")):
        if r["arm"] == arm:
            return r
    raise AssertionError(f"{arm} not in {CSV}")


def paired(a, b, seed=11703):
    """The registered B2 difference, with the same bootstrap the scorer uses."""
    from analysis.order_averaged_h2h import paired_boot
    rs = list(csv.DictReader(open(PER, encoding="utf-8")))
    d = [float(r[f"gain_{a}_n64"]) - float(r[f"gain_{b}_n64"]) for r in rs]
    lo, hi = paired_boot(d, random.Random(seed))
    return sum(d) / len(d), lo, hi


def test_neither_new_rung_lifts_the_judged_gain():
    """The finding, and the registered prediction. If a re-run reverses it, the saturation
    sentence and the Limitations clause are no longer supported."""
    for tag in ("sel14b", "sel72b"):
        m, lo, hi = paired(tag, "sel7b")
        assert lo <= 0, f"{tag} now lifts the judged gain ({m:+.4f} [{lo:+.4f}, {hi:+.4f}]); the " \
                        f"saturation reading and Limitations must be revisited"


def test_the_ladder_is_non_monotone_in_scorer_size():
    """7.6B is the peak. That is the shape the appendix claims, and caution (ai) says the claim
    ABOUT a set of numbers is what nothing checks -- so it is rebuilt here."""
    g = {t: float(level(f"sel{t}_n64")["gain"])
         for t in ("05b", "15b", "3b", "7b", "14b", "72b")}
    assert g["7b"] == max(g.values()), f"7.6B is no longer the peak: {g}"
    assert g["14b"] < g["7b"] and g["72b"] < g["7b"], f"the ladder is monotone again: {g}"


def test_all_three_readings_are_reported_as_marginal():
    """Honesty half: every difference this arm quotes sits inside the 2.0-half-width rule, and the
    appendix must not present the 14B dip as a result we build on."""
    for tag in ("sel14b", "sel72b"):
        m, lo, hi = paired(tag, "sel7b")
        hw = (hi - lo) / 2
        assert abs(m) / hw < 2.0, f"{tag} is no longer marginal ({abs(m)/hw:.2f} hw); the " \
                                  f"appendix's 'both marginal' clause is now wrong"
    t = body(SEC)
    i = t.find("Two further rungs")
    assert i > 0, "the two new rungs are gone from the saturation paragraph"
    assert "marginality rule" in t[i:i + 1400], "the marginality caveat was dropped"


def test_the_appendix_prints_the_csv():
    t = body(SEC)
    i = t.find("Two further rungs")
    para = t[i:i + 1400]
    for tag in ("7b", "14b", "72b"):
        v = f"{float(level(f'sel{tag}_n64')['gain']):+.4f}".replace("+", "")
        assert v in para, f"sel{tag}_n64's gain {v} is not printed"
    cost = float(level("sel72b_n64")["cost_vs_metered"])
    assert f"{cost:.1f}" in para, f"the 72B cost {cost:.1f}x is not printed"


def test_the_judged_disclaimer_became_a_measurement():
    """The 72B judge-free paragraph used to say this was unmeasured. It is measured now, and the
    reconciliation (task, not mechanism) is the registered consequence."""
    t = body(SEC)
    assert "does not\ntransfer to the judged workload".replace("\n", " ") in " ".join(t.split()) \
        or "not transfer to the judged workload" in " ".join(t.split()), \
        "the judge-free paragraph no longer points at the judged measurement"
    assert "statement about the \\emph{task}, not the mechanism" in " ".join(t.split()), \
        "the reconciliation the registration fixed as the consequence was dropped"


def test_limitations_no_longer_claims_it_is_unmeasured():
    t = " ".join(body("appendix_limitations.tex").split())
    assert "not known to improve" not in t, (
        "Limitations still says the judged workload is unmeasured; feat-161 measured it")
    assert "buys nothing by spending more on the reward model" in t, (
        "the deployer-facing consequence of feat-161 was dropped from Limitations")
