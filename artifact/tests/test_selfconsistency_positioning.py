"""Self-consistency is the abstract's positioning claim, so Related Work must carry it.

The abstract and intro both say the field's most deployed inference-time method already carries
this certificate. Related Work is where a reviewer checks a positioning claim against prior art,
and wang2023selfconsistency appeared in neither related_work_v4.tex nor appendix_related.tex --
it was cited only in the sections that USE it. Same class as caution (af): a claim that lives in
one part of the paper and not in the part a reader goes to for it.

The main text carries only the citation (adding the prose spilled the body onto page 10, caution
(n)); the uncounted appendix carries the argument.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import body  # noqa: E402

KEY = "wang2023selfconsistency"


def test_related_work_cites_it():
    assert KEY in body("related_work_v4.tex"), \
        "Related Work does not cite self-consistency, which the abstract calls the deployed instance"


def test_the_appendix_makes_the_argument_and_concedes_the_mechanism_is_not_ours():
    txt = body("appendix_related.tex")
    assert KEY in txt, "the appendix lost the self-consistency paragraph"
    assert "a mode is a score" in txt, "the reason it is an instance was trimmed"
    # v10 (2026-09-24) made it its own sentence, "The mechanism is theirs." -- capital T only.
    assert "the mechanism is theirs" in txt.lower(), \
        "the concession that we add the reading and not the mechanism was trimmed"
    assert "$\\log n$" in txt


def test_the_claim_is_consistent_wherever_it_appears():
    """It is an instance because Prop 1 assumes nothing about the score -- that must be why,
    everywhere, since any other reason would be a different (and weaker) claim."""
    for sec in ("iclr_intro.tex", "selection.tex", "appendix_related.tex"):
        txt = body(sec)
        if KEY not in txt:
            continue
        i = txt.find(KEY)
        window = txt[max(0, i - 400):i + 700]
        assert "instance" in window or "mode is a score" in window, (sec, window[:200])
