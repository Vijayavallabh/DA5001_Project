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
    """v10 (2026-09-24) compressed this into the 'climb to n=64 at other anchors' paragraph of
    appendix_selection.tex; the wording changed and every number stayed. The band and the three
    ratio/distance pairs are now rebuilt from the per-prompt CSVs rather than typed, so the prose
    cannot drift from the draws it reports (caution (j))."""
    txt = body("appendix_selection.tex")
    assert "Both of those were re-drawn the same way and held" in txt, \
        "the replication of both anchors was trimmed"
    g, lo, hi, _ = _paired("_comma7bseed52b8")
    assert f"${g:+.4f}$ $[{lo:+.4f}, {hi:+.4f}]$" in txt, "Comma-7B's replication band was trimmed"
    assert "the ratio predicts whether a reading survives, not how far it moves" in txt, \
        "the out-of-sample refinement of the stability criterion was trimmed"

    def ratio_and_distance(orig_tag, rep_tag):
        go, olo, ohi, _ = _paired(orig_tag)
        gr, _rlo, _rhi, _ = _paired(rep_tag)
        return go / ((ohi - olo) / 2), abs(gr - go)

    (r1, d1), (r2, d2), (r3, d3) = (ratio_and_distance("_kl3m17b64", "_kl3m17bseed52"),
                                    ratio_and_distance("", "_seed52"),
                                    ratio_and_distance("_comma7b64", "_comma7bseed52b8"))
    pairs = (f"at ${r1:.2f}$, ${r2:.2f}$ and ${r3:.2f}$ half-widths the three re-draws moved by "
             f"${d1:.3f}$, ${d2:.3f}$ and ${d3:.3f}$")
    assert pairs in txt, ("the three ratio/distance pairs that make the refinement checkable were "
                          "trimmed", pairs)
    assert "We do not pool a reading with its own failed replication" in txt, \
        "the refusal to pool was trimmed"


def test_the_merge_launchers_let_the_odometer_see_their_waiting():
    """A sentinel-wait shell holds no GPU, and analysis/compute_hours.py already knows that.

    Its traced_sleep() subtracts every `+ sleep N` line a `set -x` trace recorded, precisely so a
    shell waiting hours on a file is not billed. These three launchers did not trace, so 7.69 idle
    hours went into the odometer on 2026-09-18 (2.07 h and 5.62 h in two merge shells). The total
    stayed a true upper bound -- the manuscript says "at most" -- it was just that much looser than
    the work cost.

    A bare `set -x` does not fix it: xtrace writes to stderr and these are launched with stderr
    discarded, so the trace would land nowhere the odometer reads. The trace needs its own
    descriptor on the log, which is what this asserts.
    """
    for name in ("run_kl3mseed_merge.sh", "run_comma7bseed_merge.sh", "run_comma7bseed8_merge.sh"):
        sh = open(os.path.join(ROOT, "scripts", name), encoding="utf-8").read()
        live = "\n".join(l for l in sh.splitlines() if not l.lstrip().startswith("#"))
        assert 'exec 9>>"$LOG"' in live and "BASH_XTRACEFD=9" in live, \
            (name, "the wait is not traced to the log, so the odometer will bill it as GPU time")
        assert "set -x" in live and "set +x" in live, (name, "trace is never turned off")
        assert live.index("set -x") < live.index("sleep 60") < live.index("set +x"), \
            (name, "the sleep loop is outside the traced region")
