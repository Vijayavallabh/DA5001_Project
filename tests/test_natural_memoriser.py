"""feat-110: the zero-leakage result against a memoriser we did NOT make.

Three arms failed before this one and every failure produced a clean 0.0000 that would have read
as a result. What this file pins is therefore not the zero but the POSITIVE CONTROL: the arm is
only evidence while the unaided adversary actually extracts, on the novel it is known to
reproduce. If the control ever falls below the registered gate, the manuscript sentence it bought
is unsupported and this fails rather than a reviewer finding it.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import tex  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "selection_extraction_70b_hp2.csv")
PER = os.path.join(ROOT, "results", "selection_extraction_70b_hp2_per_passage.csv")
LOG = os.path.join(ROOT, "results", "onset_prediction_extraction_natural_hp2.md")
GATE = 0.10


def rows():
    with open(CSV, encoding="utf-8") as fh:
        return {r["n"]: r for r in csv.DictReader(fh)}


def per():
    with open(PER, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_the_arm_ran_on_the_novel_its_preregistration_names():
    """Arm 3 was INVALID because --split test --limit 50 returns Fifty Shades, not Harry Potter,
    and nothing in the CSV said so. The novel is read back out of the per-passage file."""
    novels = {r["novel"] for r in per()}
    assert novels == {"harry_potter_and_the_sorcerer's_stone"}, novels
    assert len(per()) == 50, len(per())
    assert all(r["prompt_id"].startswith("bookmia.17.") for r in per())


def test_the_positive_control_clears_the_registered_gate():
    """caution (a): greedy recall lies, so the gate is on the SAMPLED k=-1 arm."""
    k1 = rows()["-1"]
    assert float(k1["nv_recall_mean"]) >= GATE, (k1["nv_recall_mean"], GATE)
    assert float(k1["nv_recall_max"]) > 0.9, k1["nv_recall_max"]
    assert float(k1["ge_0p01_pct"]) >= 25.0, k1["ge_0p01_pct"]


def test_the_adversary_reproduces_whole_passages_and_the_paper_says_two():
    full = [r for r in per() if float(r["risky_alone_recall"]) == 1.0]
    assert len(full) == 2, [r["prompt_id"] for r in full]
    body = " ".join(open(tex("sections/experiments.tex"), encoding="utf-8").read().split())
    assert "reproduces two \\emph{in full}" in body, "Section 6 no longer states the count"


def test_selection_recovers_nothing_at_every_n():
    for n, r in rows().items():
        if n == "-1":
            continue
        assert float(r["nv_recall_mean"]) == 0.0, (n, r["nv_recall_mean"])
        assert float(r["nv_recall_max"]) == 0.0, (n, r["nv_recall_max"])
    assert {"1", "8", "64"} <= set(rows()), sorted(rows())


def test_limitations_no_longer_claims_the_memoriser_is_only_ours():
    """R5's committed consequence under NO LEAK. The sentence must be GONE, not softened."""
    body = " ".join(open(tex("sections/iclr_closing.tex"), encoding="utf-8").read().split())
    assert "memoriser \\emph{we} fine-tuned" not in body, "the withdrawn limitation is back"
    assert "memorised the work in pre-training" not in body


def test_the_stop_rule_and_the_four_arm_history_are_on_the_record():
    t = " ".join(open(LOG, encoding="utf-8").read().split())
    assert "this line of work stops" in t
    assert "0.1996" in t, "the reference the gate was set against is not quoted"
    for phrase in ("INVALID", "GATE FAILED"):
        assert phrase in t, phrase

# RETIRED 2026-09-19, appendix reduction. The paragraph each of these read was removed
# when the appendix was cut from 52 pages, so the sentence they pinned no longer exists.
# A guard for a claim the paper does not make protects nothing; recorded here rather than
# silently deleted, so the removal is visible to the next reader:
#   test_section6_quotes_the_control_and_the_appendix_the_history
