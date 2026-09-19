"""feat-134: Comma-7B past n=64, registered before it runs.

The arm closes the appendix's open sentence, "We did not take Comma-7B past 64, so where the
strongest anchor's ceiling sits is open". These guards pin the design, the reproduction gate and the
one premise the gate exists to test; the verdict assertion arrives with the scoring commit, because
there is nothing to assert about a number that does not exist yet.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.score_n128 import ANCHORS, reward_gate  # noqa: E402

LOG = os.path.join(ROOT, "results", "onset_prediction_comma7b_n128.md")


def _live(name):
    sh = open(os.path.join(ROOT, "scripts", name), encoding="utf-8").read()
    return "\n".join(l for l in sh.splitlines() if not l.lstrip().startswith("#"))


def test_the_launchers_pass_no_batch_size_and_extend_the_arm_on_record():
    """h1.py's default of 8 IS the protocol, because output/phase5/sel_comma7b_64 passed no flag.

    Asserting the flag's ABSENCE rather than its value is deliberate: feat-132 was made unreadable by
    adding --batch-size 32 to a re-draw, and here a mismatch would break the reproduction gate that
    is the whole point of the arm.
    """
    for name in ("run_comma7b128_card1.sh", "run_comma7b128_card2.sh"):
        live = _live(name)
        assert "--batch-size" not in live, (name, "passes --batch-size; it must not")
        assert "--seeds 42 43 44" in live, (name, "the seeds must match the arm being extended")
        assert "--trajectories-per-prompt 128" in live, name
        assert "common-pile/comma-v0.1-2t" in live and "--max-new-tokens 200" in live, name
    # the two cards must cover the same 500 prompts between them, and neither twice
    c1, c2 = _live("run_comma7b128_card1.sh"), _live("run_comma7b128_card2.sh")
    assert "--cap-neutral 200 --cap-creative 0 --cap-factual 0" in c1, c1
    assert "--cap-neutral 0 --cap-creative 150 --cap-factual 150" in c2, c2


def test_the_gate_is_on_the_reward_cache_and_the_committed_one_exists():
    """32,000 floats compared with ==, not 28 summary cells at 5e-4 (feat-129's correction)."""
    A = ANCHORS["comma7b"]
    old = list(csv.DictReader(open(os.path.join(ROOT, "results", A["old_cache"]),
                                   encoding="utf-8")))
    assert len(old) == 32000, (len(old), "the committed n=64 reward cache is not 500 x 64")
    assert {int(r["rank"]) for r in old} == set(range(64))
    assert len({r["prompt_id"] for r in old}) == 500
    # the gate is exact: a single float moved by 1e-6 must fail it
    bad = [dict(r) for r in old]
    bad[0]["reward"] = f"{float(bad[0]['reward']) + 1e-6:.12g}"
    ok, _ = reward_gate(bad, old, top=64)
    assert not ok, "the reproduction gate no longer detects a 1e-6 perturbation"
    ok, _ = reward_gate(old, old, top=64)
    assert ok, "the gate fails on an identical cache"


def test_the_new_cache_name_the_merge_writes_is_the_one_the_scorer_reads():
    """A scorer reading a file the launcher never writes is a gate that can only be 'not scoreable'."""
    A = ANCHORS["comma7b"]
    expected = A["old_cache"].replace("64", str(A["top"]), 1)
    assert expected == "selection_rewards128_comma7b.csv", expected
    c2 = _live("run_comma7b128_card2.sh")
    assert f"results/{expected}" in c2, (expected, "the merge launcher writes a different cache")
    assert f"--tag {A['tag']}" in c2, (A["tag"], "the merge launcher writes a different tag")
    assert "--max-n 128" in c2


def test_the_two_anchors_are_read_by_one_gate():
    a, b = ANCHORS["audited"], ANCHORS["comma7b"]
    for key in ("tag", "old_cache", "old_scaling", "out_csv", "log"):
        assert a[key] != b[key], key
    src = open(os.path.join(ROOT, "analysis", "score_n128.py"), encoding="utf-8").read()
    assert src.count("def reward_gate(") == 1, "two gates means the anchors are not read by one rule"
    assert b["with_arm_b"] is False, "extraction is feat-129's arm, not this one"


def test_the_pre_registration_commits_the_band_and_the_consequence():
    txt = open(LOG, encoding="utf-8").read()
    head, _sep, scored = txt.partition("\n## Scoring log")
    assert head, "the pre-registration has no committed half"
    for phrase in ("STILL CLIMBING", "SATURATED BY 64", "TURNS OVER",
                   "bit-identical", "No $n>64$ number is read until it clears",
                   "where the strongest anchor's ceiling sits is open"):
        assert phrase in head, f"the pre-registration does not commit: {phrase}"
    # the compute is over the escalation threshold and the file must say so
    assert "24-gpu-hour escalation threshold" in head, \
        "an arm over the threshold must say so where the next reader will see it"
