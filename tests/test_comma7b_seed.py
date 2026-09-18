"""feat-132: the Comma-7B seed replication, written BEFORE the arm produced a number.

After feat-131 the breadth-at-n=64 claim rests on two anchors, TinyComma and Comma-7B. Comma-7B has
never been re-drawn, and at 2.43 interval half-widths it sits above TinyComma's 2.12 -- the one ratio
that replicated -- so this arm is simultaneously the only outstanding check on a claim the paper
makes and the only OUT-OF-SAMPLE test of the stability criterion the paper states.

These guards pin the pre-registration's own arithmetic and the scorer's decision rule, so that
neither can drift between the commit that registered the bands and the commit that reads them. The
verdict assertion deliberately lives in test_comma7b_seed_scored (added when the arm is scored) and
not here: there is nothing to assert about a number that does not exist yet.
"""
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.score_kl3m_seed import ANCHORS  # noqa: E402
from analysis.selection_decoding import boot_mean  # noqa: E402

JUDGE_B = "Phi-3.5-mini-instruct"
LOG = os.path.join(ROOT, "results", "onset_prediction_comma7b_seed.md")


def _paired(tag):
    p = os.path.join(ROOT, "results", f"selection_scaling_per_prompt{tag}.csv")
    rows = [r for r in csv.DictReader(open(p, encoding="utf-8")) if JUDGE_B in r["judge"]]
    d = [float(r["u_n64"]) - float(r["u_n8"]) for r in rows]
    g = sum(d) / len(d)
    lo, hi = boot_mean(d, random.Random(20260918))
    return g, lo, hi


def test_the_committed_reference_band_rounds_from_the_arm_on_record():
    """caution (j): a paper number rounds from the CSV, once. A pre-registration number too --
    otherwise the arm is compared against a figure nobody can reproduce."""
    g, lo, hi = _paired(ANCHORS["comma7b"]["orig_tag"])
    want = ANCHORS["comma7b"]["committed_orig"]
    assert abs(g - want[0]) < 5e-4 and abs(lo - want[1]) < 5e-4 and abs(hi - want[2]) < 5e-4, \
        (g, lo, hi, want)
    txt = open(LOG, encoding="utf-8").read()
    assert "`+0.1010 [+0.0590, +0.1420]`" in txt, "the pre-registration's own table drifted"


def test_the_criterion_ratios_are_what_the_pre_registration_says_they_are():
    """The arm exists because 2.43 > 2.12 > 1.71. If those ever move, the reason for running it
    moves with them and the log must be re-read rather than quietly kept."""
    ratios = {}
    for name, tag in (("tinycomma", ""), ("kl3m17b", "_kl3m17b64"), ("comma7b", "_comma7b64")):
        g, lo, hi = _paired(tag)
        ratios[name] = g / ((hi - lo) / 2)
    assert abs(ratios["tinycomma"] - 2.12) < 0.05, ratios
    assert abs(ratios["kl3m17b"] - 1.71) < 0.05, ratios
    assert abs(ratios["comma7b"] - 2.43) < 0.05, ratios
    assert ratios["comma7b"] > ratios["tinycomma"] > ratios["kl3m17b"], ratios


def test_the_integrity_reference_is_measured_on_the_arm_being_replicated():
    """The defect the mutation test caught before the run, pinned so it cannot come back.

    The first draft took 0.030 from selection_breadth.csv's n=8 column. The n=64 arm being
    replicated reads 0.094, and 0.064 against a 0.03 tolerance would have failed a good arm --
    caution (v), a reference number carries its protocol. feat-131's own reference is left exactly
    as it was committed, which is why the two entries disagree about where the number comes from.
    """
    assert abs(ANCHORS["comma7b"]["committed_empty_frac"] - 0.094) < 1e-9, \
        "the empty-fraction reference must be the n=64 arm's own, not the n=8 column's 0.030"
    assert ANCHORS["kl3m17b"]["committed_empty_frac"] == 0.002, \
        "feat-131's committed value must not be edited after the fact"


def test_the_two_arms_are_read_by_one_rule_and_address_different_data():
    a, b = ANCHORS["kl3m17b"], ANCHORS["comma7b"]
    for key in ("orig_tag", "rep_tag", "gen_dir", "out_csv", "log"):
        assert a[key] != b[key], key
    # one rule: the scorer has exactly one verdict expression, and it is the pre-registered one
    src = open(os.path.join(ROOT, "analysis", "score_kl3m_seed.py"), encoding="utf-8").read()
    assert src.count('"REPLICATES" if lo > 0') == 1, \
        "two verdict expressions means the two arms are no longer read by one rule"
    assert 'INVERTS" if hi < 0' in src


def test_the_seed_sets_of_the_replication_and_the_arm_on_record_are_disjoint():
    from dap.stats import build_trajectory_seeds
    a = set(build_trajectory_seeds("x", (42, 43, 44), 64))
    b = set(build_trajectory_seeds("x", (52, 53, 54), 64))
    assert len(a) == len(b) == 64 and not (a & b), f"{len(a & b)} of 64 seeds collide"


def test_the_launcher_runs_the_anchor_and_seeds_the_pre_registration_names():
    sh = open(os.path.join(ROOT, "scripts", "run_comma7bseed_gen.sh"), encoding="utf-8").read()
    assert "--seeds 52 53 54" in sh and "common-pile/comma-v0.1-2t" in sh
    assert "--trajectories-per-prompt 64" in sh and "--batch-size 32" in sh
    assert "--max-new-tokens 200" in sh
    # the marker is written only on rc=0, so a failed class cannot be merged (caution (c))
    assert "[ $RC -eq 0 ] && date +%s" in sh


def test_feat132_is_recorded_invalid_and_its_band_was_never_read():
    """The strongest fact about feat-132 is a negative one: its band does not exist.

    A replication that fails its integrity check and then has its band computed 'just to see' is a
    replication whose re-run can be tuned to the answer. The scorer returns before computing it, and
    no results file carries it -- which is what makes feat-133 an honest re-run rather than a second
    attempt at a number already seen.
    """
    import glob
    assert not glob.glob(os.path.join(ROOT, "results", "comma7b_seed_scoring.csv")), \
        "feat-132's band was computed; it must not be, and feat-133 is now compromised"
    log = open(os.path.join(ROOT, "results", "onset_prediction_comma7b_seed.md"),
               encoding="utf-8").read()
    _head, _sep, scored = log.partition("\n## Scoring, ")
    assert scored, "feat-132 has no scoring section"
    assert "INVALID" in scored and "The band was not read" in scored


def test_the_corrected_gate_is_scale_free_and_stratified():
    """Both defects feat-132's gate had, pinned so neither can return.

    An absolute tolerance on a rate is unfalsifiable where the rate is near zero and tighter than the
    quantity's own spread where it is not; and an aggregate gate on a stratified rate gates the wrong
    quantity -- feat-132's total would have passed the corrected test while its neutral class failed.
    """
    from analysis.score_kl3m_seed import two_proportion_z, Z_CRIT, ANCHORS
    A = ANCHORS["comma7b8"]
    assert A["gate"] == "z" and "batch" not in str(A.get("committed_empty_frac", ""))
    assert set(A["committed_counts"]) == {"neutral", "total"}, \
        "the gate must be stratified: the aggregate hid feat-132's defect"
    assert A["committed_counts"] == {"neutral": (45, 200), "total": (47, 500)}
    # it refuses feat-132's data on neutral and would have let it through on the total alone
    assert abs(two_proportion_z(45, 200, 24, 200)) > Z_CRIT, "the gate no longer refuses feat-132"
    assert abs(two_proportion_z(47, 500, 28, 500)) < Z_CRIT, \
        "if the total alone now fires, the point about stratification is lost -- recheck the claim"
    # and it does not fire against the arm it references
    assert two_proportion_z(45, 200, 45, 200) == 0.0


def test_the_corrected_launcher_passes_no_batch_size_at_all():
    """h1.py's default of 8 IS the protocol, because the arm on record passed no flag either.

    Adding --batch-size 32 is precisely what made feat-132 unreadable, so this asserts the flag's
    absence rather than its value.
    """
    sh = open(os.path.join(ROOT, "scripts", "run_comma7bseed8_gen.sh"), encoding="utf-8").read()
    body = "\n".join(l for l in sh.splitlines() if not l.lstrip().startswith("#"))
    assert "--batch-size" not in body, \
        "feat-133's launcher passes --batch-size; the whole point is that it must not"
    assert "--seeds 52 53 54" in body and "common-pile/comma-v0.1-2t" in body
    assert "--trajectories-per-prompt 64" in body and "--max-new-tokens 200" in body
    assert "[ $RC -eq 0 ] && date +%s" in body
