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
        txt = body(*sections)
        if f"${g:+.3f}$ $[{lo:+.3f}, {hi:+.3f}]$" in txt:
            return True
    for _label, (fg, flo, fhi), _cost, _cert, _gate in _forest():
        if flo is None:
            continue
        if (abs(fg - g) < tol and abs(flo - lo) < tol and abs(fhi - hi) < tol):
            return True
    return False
