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
    cards = ("run_comma7b128_card1.sh", "run_comma7b128_card2b.sh", "run_comma7b128_card3.sh")
    for name in cards:
        live = _live(name)
        assert "--batch-size" not in live, (name, "passes --batch-size; it must not")
        assert "--seeds 42 43 44" in live, (name, "the seeds must match the arm being extended")
        assert "--trajectories-per-prompt 128" in live, name
        assert "common-pile/comma-v0.1-2t" in live and "--max-new-tokens 200" in live, name

    # The three cards must cover the 500 prompts exactly once between them. Rebuild the caps from
    # the launchers and add them up rather than matching one string per card: the registered
    # two-card plan was re-dealt to three when GPU 1 came free, and a re-deal that double-counted a
    # class would pass a per-card string check while silently judging 650 prompts.
    import re as _re
    total = {"neutral": 0, "creative": 0, "factual": 0}
    for name in cards:
        live = _live(name)
        for cls in total:
            m = _re.search(rf"--cap-{cls} (\d+)", live)
            assert m, (name, cls)
            total[cls] += int(m.group(1))
    assert total == {"neutral": 200, "creative": 150, "factual": 150}, \
        (total, "the cards do not cover the 500 prompts exactly once")

    # and the superseded two-card launcher must not come back: running it would re-do factual
    assert not os.path.exists(os.path.join(ROOT, "scripts", "run_comma7b128_card2.sh")), \
        "the superseded card2 launcher is back; it would double-count the factual class"


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
    c2 = _live("run_comma7b128_card2b.sh")
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


def test_the_merge_shell_owns_the_scoring_and_waits_safely():
    """feat-134 was half-killed by another session's GPU job, so the merge moved out of card2b.

    First it moved into run_comma7b128_requeue.sh, which had to QUEUE factual behind creative
    because only one card was free; GPU 2 then came free, factual was started directly, and the
    queueing half was deleted. What survives is a merge-only shell, and the thing to guard is that
    it still cannot become a second experiment: it owns no generation command at all.
    """
    live = _live("run_comma7b128_merge.sh")
    assert "--batch-size" not in live, "the merge shell passes --batch-size; it must not"
    assert "h1.py" not in live, "the merge shell must not carry a generation command"
    import os as _os
    assert not _os.path.exists(_os.path.join(ROOT, "scripts", "run_comma7b128_requeue.sh")), \
        "the superseded requeue shell is back; it would launch factual a second time"

    # caution (c), eight incidents: wait on files and on a string the CURRENT script writes
    assert "pgrep" not in live and "pkill" not in live, "the merge shell waits on a pattern match"
    assert "GEN_DONE" in live, "the merge shell does not wait on the success sentinels"
    assert "generation rc=0" in live, "it does not wait on a string card2b actually writes"
    assert "BASH_XTRACEFD" in live, "its sleeps are untraced and will bill as GPU time"

    # card2b is still alive and may reach the merge first; exactly one of them may score
    assert "START scoring" in live, "the merge shell does not check whether card2b already scored"

    A = ANCHORS["comma7b"]
    expected = A["old_cache"].replace("64", str(A["top"]), 1)
    assert f"results/{expected}" in live and f"--tag {A['tag']}" in live, \
        "the merge shell writes a different cache or tag than the scorer reads"
    assert "--max-n 128" in live


def test_the_incident_is_recorded_below_the_committed_line():
    """A pre-registration's value is that nothing above `## Scoring log` changes after the fact."""
    txt = open(LOG, encoding="utf-8").read()
    head, _sep, scored = txt.partition("\n## Scoring log")
    assert "OOM" not in head and "agenticls" not in head, \
        "the incident was written into the committed half of the pre-registration"
    assert "OutOfMemoryError" in scored, "the OOM incident is not recorded in the scoring log"
    assert "has not been computed or looked at" in scored, \
        "the log must state that the band was never read, or the rerun is a second attempt"
