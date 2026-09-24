"""Where the manuscript lives.

On this box `~` is a service account and is NOT the project home, so `expanduser("~/sub/satml")` names a
directory that does not exist. Three tests guarded on that path and therefore **skipped silently**
for as long as they existed: the 72-cell appendix table check, the compute-hours check against the
LLM-usage sentence, and this session's Section 4 table check. Use the `$SATML_DIR` convention the
scripts already use, and fail loudly if the manuscript is genuinely absent rather than returning.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.environ.get("SATML_DIR") or os.path.normpath(os.path.join(ROOT, os.pardir, "sub", "satml"))


def tex(name):
    return os.path.join(DIR, name)


def body(*names):
    """One whitespace-normalised string of the named live section files."""
    return " ".join(" ".join(open(tex(os.path.join("sections", n)),
                                  encoding="utf-8").read().split()) for n in names)


def caption_of(label):
    """Whitespace-normalised caption of the float carrying \\label{<label>}, in whichever live
    section file holds it. A float that moves between the body and an appendix keeps its guard
    (caution (al)): the guard follows the label, not the file it used to be in."""
    import re
    # only the files the manuscript actually \\inputs (recursively): sections/ also holds retired
    # SaTML sections that are never compiled, and a label there is not a label in the paper.
    live, todo = [], ["iclr_2027.tex"]
    while todo:
        f = todo.pop()
        live.append(f)
        src = re.sub(r"(?<!\\)%.*", "", open(tex(f), encoding="utf-8").read())  # not commented-out ones
        todo += [m + ".tex" for m in re.findall(r"\\input\{([^}]*)\}", src)]
    hits = []
    for f in live:
        t = " ".join(open(tex(f), encoding="utf-8").read().split())
        i = t.find("\\label{%s}" % label)
        if i >= 0:
            hits.append(t[t.rindex("\\caption{", 0, i): i])
    assert len(hits) == 1, f"\\label{{{label}}} is in {len(hits)} live section files"
    return hits[0]


def _forest():
    """The rows Figure~\\ref{fig:breadth} plots, from figures/make_figures_v4.py."""
    import sys
    fig = os.path.join(ROOT, "figures")
    if fig not in sys.path:
        sys.path.insert(0, fig)
    from make_figures_v4 import selection_forest_rows  # noqa: E402
    return selection_forest_rows()[0]


def carries_band(g, lo, hi, *sections, tol=5e-4):
    """Does the paper print this band ANYWHERE it prints bands -- prose, or the forest figure?

    Written 2026-09-17, when converting Section 3's breadth paragraph into a figure silently
    retired six guards. Each one asserted 'the prose quotes this number'; the numbers had not been
    deleted, they had moved into a plot the guards could not see, and a guard that cannot see the
    claim is a guard that passes by never running (caution (aj)). A band is guarded wherever the
    paper chooses to print it, so check both surfaces.
    """
    if sections:
        # MATCH NUMERICALLY, NOT BY SPELLING. This used to build one f-string at 3 decimals, so a
        # band the paper prints at 4 -- which is the appendix's own convention -- did not count as
        # printed, and a guard that cannot see the claim passes by never running (caution (aj)).
        # Reading the triples out and comparing them as numbers is spelling-independent, which is
        # caution (an)'s rule: guard the property, not the spelling.
        import re
        txt = body(*sections)
        pat = re.compile(r"\$([+-]?\d*\.\d+)\$ ?\$\[([+-]?\d*\.\d+), ?([+-]?\d*\.\d+)\]\$")

        def rounds_to(printed, value):
            """Caution (j): a paper number rounds from the CSV, ONCE -- so compare at the precision
            the paper printed, not against a fixed tolerance. A flat 5e-4 is exactly wrong at the
            boundary: a CSV 0.0665 printed as $+0.067$ differs by 5e-4 and is a correct rounding,
            and a `< tol` test rejects it."""
            d = len(printed.split(".")[1])
            return f"{value:+.{d}f}" == printed or abs(float(printed) - value) < tol

        for m in pat.finditer(txt):
            a, b, c = m.groups()
            if rounds_to(a, g) and rounds_to(b, lo) and rounds_to(c, hi):
                return True
    for _label, (fg, flo, fhi), _cost, _cert, _gate in _forest():
        if flo is None:
            continue
        if (abs(fg - g) < tol and abs(flo - lo) < tol and abs(fhi - hi) < tol):
            return True
    return False
