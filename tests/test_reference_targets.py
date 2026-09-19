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


def _doc_lines():
    r"""The document in \input order, with \input{sections/X} spliced in.

    Reading the section files as a flat list loses exactly the fact that matters: onset.tex declares
    no \section of its own because it is \input INSIDE orders.tex, so a per-file walk resolves
    sec:onset to None and every check that depends on it silently passes.
    """
    def expand(path, depth=0):
        if depth > 4 or not os.path.exists(path):
            return
        for line in open(path, encoding="utf-8"):
            m = re.match(r"\s*\\input\{([^}]*)\}", line)
            if m:
                child = m.group(1)
                yield from expand(tex(child if child.endswith(".tex") else child + ".tex"), depth + 1)
            else:
                yield line
    return list(expand(tex("iclr_2027.tex")))


def _resolve():
    """label -> the label whose NUMBER it will render as.

    A paragraph label renders as its enclosing section's number; everything else renders as itself.
    """
    target = {}
    section = None
    for line in _doc_lines():
        m = re.search(r"\\section\*?\{[^}]*\}\\label\{([^}]*)\}", line)
        if m:
            section = m.group(1)
            target[section] = section
            continue
        m = re.search(r"\\paragraph\{[^}]*\}\\label\{([^}]*)\}", line)
        if m:
            target[m.group(1)] = section          # None only before the first \section
    return target


def _section_ordinals():
    """label -> 1-based position of its \\section in the MAIN TEXT, in document order.

    Counted over the spliced document rather than per file, because the numbers a reader compares
    are the document's, not each file's.
    """
    order, n = {}, 0
    for line in _doc_lines():
        if re.match(r"\s*\\appendix\b", line):
            break
        m = re.search(r"\\section\*?\{[^}]*\}\\label\{([^}]*)\}", line)
        if m:
            n += 1
            order[m.group(1)] = n
    return order


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


def test_no_sentence_names_the_same_number_twice():
    """The parenthetical check above missed the same defect in running prose.

    appendix_onset.tex opened with "Section~\\ref{sec:onset} states the nine-pair result in a
    paragraph and Section~\\ref{sec:orders} the four-order comparison in two." onset.tex is
    \\input INSIDE orders.tex, so both labels resolve to the same section and the sentence rendered
    "Section 5 states the nine-pair result in a paragraph and Section 5 the four-order comparison in
    two." Exit 0, 0 overfull, 0 ?? -- a real section number, just the same one twice, in a sentence
    whose whole point is that they are different places.
    """
    target = _resolve()
    bad = []
    for f in _files():
        # Strip comments BEFORE joining lines: once the newlines are gone, "^\s*%.*$" matches from
        # the first comment to the end of the file and silently empties the document, which is how
        # this check passed on the very defect it is named for.
        lines = [ln for ln in open(f, encoding="utf-8").read().split("\n")
                 if not ln.lstrip().startswith("%")]
        body = " ".join(lines)
        # A caption ends ".}" and the period is INSIDE the brace, so a plain sentence split
        # ran the caption on into the next \paragraph heading and reported the heading's
        # \ref as a duplicate of the caption's. Float ends are sentence boundaries too
        # (exposed 2026-09-19 when the block between a caption and a heading was cut).
        body = re.sub(r"\\end\{(figure|table)\}", ". ", body)
        for sentence in re.split(r"(?<=[.!?])\s+|(?<=\.\})\s+", body):
            refs = re.findall(r"\\ref\{([^}]*)\}", sentence)
            if len(refs) < 2:
                continue
            nums = [target.get(r, r) for r in refs]
            for n in set(nums):
                if n is not None and nums.count(n) > 1:
                    bad.append((os.path.basename(f), refs, n))
    assert not bad, ("these render the same number twice in one sentence: "
                     + "; ".join(f"{f}: {refs} all -> {n}" for f, refs, n in bad))


def test_reference_pairs_are_not_written_in_descending_order():
    """"Sections~\\ref{sec:orders} and~\\ref{sec:experiments}" rendered as "Sections 5 and 3" in the
    LLM Usage statement after the v8 reorder renumbered both. Nothing catches a backwards pair: both
    numbers are real and both targets are right, and only a reader notices the order."""
    order = _section_ordinals()
    target = _resolve()
    bad = []
    for f in _files():
        body = open(f, encoding="utf-8").read().replace("\n", " ")
        for m in re.finditer(r"Sections~\\ref\{([^}]*)\}\s*and~?\\ref\{([^}]*)\}", body):
            a, b = (order.get(target.get(r, r)) for r in m.groups())
            if a and b and a > b:
                bad.append((os.path.basename(f), m.groups(), a, b))
    assert not bad, ("these render as a descending pair: "
                     + "; ".join(f"{f}: {g} -> {a} and {b}" for f, g, a, b in bad))


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
