"""feat-133: Comma-7B's climb to n=64 SURVIVES a disjoint draw, and the criterion held out of sample.

Paired g(64)-g(8) reads +0.0880 [+0.0460, +0.1290] at seeds 52 53 54 against +0.1010 [+0.0590,
+0.1420] at 42 43 44 -- a move of 0.013, against 0.0000 at the audited anchor and 0.0610 at
KL3M-1.7B. Both anchors the breadth-at-n=64 claim rests on have now been re-drawn and both held.

The refinement this arm forced is the interesting half: the stability criterion predicts the VERDICT
and not the size of the shift. Ratios 1.71 / 2.12 / 2.43 map to distances 0.061 / 0.000 / 0.013, so
they do not order the distances, and TinyComma's agreement to four decimals is a fortunate draw
rather than what a stable paired difference owes.
"""
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.selection_decoding import boot_mean  # noqa: E402
from tests.manuscript import body  # noqa: E402

JUDGE_B = "Phi-3.5-mini-instruct"


def _csvrows(name):
    return list(csv.DictReader(open(os.path.join(ROOT, "results", name), encoding="utf-8")))


def _paired(tag):
    rows = [r for r in _csvrows(f"selection_scaling_per_prompt{tag}.csv") if JUDGE_B in r["judge"]]
    d = [float(r["u_n64"]) - float(r["u_n8"]) for r in rows]
    g = sum(d) / len(d)
    lo, hi = boot_mean(d, random.Random(20260918))
    return g, lo, hi, len(d)


def test_the_band_rounds_from_its_csv_and_the_verdict_follows_from_it():
    r = _csvrows("comma7b_seed8_scoring.csv")[0]
    assert r["seeds"] == "52 53 54", r["seeds"]
    assert abs(float(r["gain_diff"]) - 0.088) < 5e-4, r
    assert abs(float(r["lo95"]) - 0.046) < 5e-4, r
    assert abs(float(r["hi95"]) - 0.129) < 5e-4, r
    assert float(r["lo95"]) > 0, "REPLICATES requires the interval to exclude zero"
    assert r["verdict"] == "REPLICATES", r["verdict"]
    assert r["integrity"] == "PASS" and int(r["n_prompts"]) == 500
    assert abs(float(r["distance"]) - 0.013) < 1e-3, r["distance"]
    g, lo, hi, n = _paired("_comma7bseed52b8")
    assert (round(g, 4), round(lo, 4), round(hi, 4)) == (0.088, 0.046, 0.129), (g, lo, hi)
    assert n == 500


def test_only_one_thing_changed_from_the_arm_on_record():
    """The whole reason feat-132 is INVALID and this arm is not."""
    sh = open(os.path.join(ROOT, "scripts", "run_comma7bseed8_gen.sh"), encoding="utf-8").read()
    live = "\n".join(l for l in sh.splitlines() if not l.lstrip().startswith("#"))
    assert "--batch-size" not in live, "feat-133's launcher must pass no --batch-size at all"
    assert "--seeds 52 53 54" in live and "common-pile/comma-v0.1-2t" in live
    from dap.stats import build_trajectory_seeds
    a = set(build_trajectory_seeds("x", (42, 43, 44), 64))
    b = set(build_trajectory_seeds("x", (52, 53, 54), 64))
    assert len(a) == len(b) == 64 and not (a & b), f"{len(a & b)} of 64 seeds collide"


def test_the_criterion_predicts_the_verdict_and_not_the_shift():
    """Ratios 1.71 / 2.12 / 2.43 against distances 0.061 / 0.000 / 0.013.

    Guard the SHAPE claim, both ways (caution (ao)): the ratio must separate the verdicts, and it
    must NOT order the distances. If it ever does order them, this fails and says to revisit the
    wording rather than quietly permitting the stronger rule.
    """
    def ratio_and_distance(orig_tag, rep_tag):
        go, lo, hi, _ = _paired(orig_tag)
        gr, _rlo, _rhi, _ = _paired(rep_tag)
        return go / ((hi - lo) / 2), abs(gr - go)

    kl = ratio_and_distance("_kl3m17b64", "_kl3m17bseed52")
    tc = ratio_and_distance("", "_seed52")
    c7 = ratio_and_distance("_comma7b64", "_comma7bseed52b8")
    for got, want in ((kl, (1.71, 0.061)), (tc, (2.12, 0.000)), (c7, (2.43, 0.013))):
        assert abs(got[0] - want[0]) < 0.05 and abs(got[1] - want[1]) < 1e-3, (got, want)
    # the ratio separates the verdicts
    assert kl[0] < 2.0 < tc[0] and tc[0] < c7[0]
    # and does NOT order the distances -- the larger ratio moved further
    assert c7[1] > tc[1], \
        ("the ratio now orders the distances too; the appendix says it does not, so revisit that "
         "sentence deliberately rather than leaving the paper weaker than its evidence", tc, c7)


def test_the_appendix_records_both_replications_and_the_refinement():
    txt = body("appendix_selection.tex")
    assert "Both of the two were then re-drawn" in txt, "the replication of both anchors was trimmed"
    assert "$+0.0880$ $[+0.0460, +0.1290]$" in txt, "Comma-7B's replication band was trimmed"
    assert "predicts the verdict rather than the shift" in txt, \
        "the out-of-sample refinement of the stability criterion was trimmed"
    assert "$1.71 \\to 0.061$, $2.12 \\to 0.000$, $2.43 \\to 0.013$" in txt, \
        "the three ratio/distance pairs that make the refinement checkable were trimmed"
    assert "We do not pool the two draws" in txt, "the refusal to pool was trimmed"
