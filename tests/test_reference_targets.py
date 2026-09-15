"""Two \\refs in one parenthetical must not render as the same number.

A \\label attached to a \\paragraph captures no counter of its own, so \\ref resolves it to the
enclosing \\section. That is fine and often intended -- sec:onset is a paragraph inside the section
that \\inputs it, and 23 references to it render correctly. It is NOT fine when two such refs sit
side by side: "(Appendices~\\ref{app:saturation},~\\ref{app:judgefreescale})" rendered as
"Appendices I, I" and "(Appendices~\\ref{app:saturation}--\\ref{app:scorerfree})" as
"Appendices I--I", both in the compiled PDF, for as long as those labels existed.

Nothing in the build sees it: tectonic exits 0, the overfull count is 0, `??` is 0, and the page
shows a real appendix letter -- just the same one twice. Same class as cautions (y) and (z): a
source-level defect only a reader of the rendered page would catch.
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import tex  # noqa: E402

SECTIONS = ("iclr_intro", "selection", "experiments", "frontier", "orders", "onset",
            "related_work_v4", "iclr_closing", "appendix_proofs", "appendix_opening",
            "appendix_onset", "appendix_robustness", "appendix_seed", "appendix_second_anchor",
            "appendix_selection", "appendix_limitations", "appendix_related")


def _files():
    out = [tex("iclr_2027.tex")]
    for s in SECTIONS:
        p = tex(f"sections/{s}.tex")
        if os.path.exists(p):
            out.append(p)
    return out


def _resolve():
    """label -> the label whose NUMBER it will render as.

    A paragraph label renders as its enclosing section's number; everything else renders as itself.
    """
    target = {}
    for f in _files():
        section = None
        for line in open(f, encoding="utf-8"):
            m = re.search(r"\\section\*?\{[^}]*\}\\label\{([^}]*)\}", line)
            if m:
                section = m.group(1)
                target[section] = section
                continue
            m = re.search(r"\\paragraph\{[^}]*\}\\label\{([^}]*)\}", line)
            if m:
                target[m.group(1)] = section          # None if before any \section
    return target


def test_no_parenthetical_renders_the_same_number_twice():
    target = _resolve()
    bad = []
    for f in _files():
        body = open(f, encoding="utf-8").read().replace("\n", " ")
        for paren in re.findall(r"\(([^()]*)\)", body):
            refs = re.findall(r"\\ref\{([^}]*)\}", paren)
            if len(refs) < 2:
                continue
            nums = [target.get(r, r) for r in refs]
            for n in set(nums):
                if n is not None and nums.count(n) > 1:
                    bad.append((os.path.basename(f), refs, n))
    assert not bad, ("these render the same number twice: "
                     + "; ".join(f"{f}: {refs} all -> {n}" for f, refs, n in bad))


def test_every_referenced_label_exists_somewhere():
    """A \\ref to a label declared only in a kept-verbatim _v<n> predecessor renders as ?? --
    caught by the ?? check -- but one declared nowhere at all is the same defect, found earlier."""
    declared = set()
    for f in glob.glob(tex("sections/*.tex")) + [tex("iclr_2027.tex")]:
        declared |= set(re.findall(r"\\label\{([^}]*)\}", open(f, encoding="utf-8").read()))
    used = set()
    for f in _files():
        used |= set(re.findall(r"\\ref\{([^}]*)\}", open(f, encoding="utf-8").read()))
    assert not (used - declared), sorted(used - declared)
