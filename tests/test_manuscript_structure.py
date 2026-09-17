"""Structural checks over the files the manuscript actually \\input.

Three defects this catches, all invisible to the compiler -- tectonic exits 0, the overfull count
is 0 and `??` is 0 for every one of them:

  * a label declared TWICE silently redirects every \\ref to the second one. caution (z): the
    paper's own theorem rendered as "Proposition 6" and pointed at an appendix restatement for
    three drafts, because prop:sparse was declared in the body and again in appendix_proofs.
  * a \\ref to a label that exists in the repo but in a file the build does not \\input.
  * a float NOTHING references. Nine of them existed on 2026-09-17 -- seven figures and two
    tables, each with a real caption, none of which the prose ever sent the reader to. A reviewer
    scoring presentation reads an unreferenced float as filler.

The build graph is walked from iclr_2027.tex rather than globbed, because sections/ also holds
retired SaTML-era files and every kept `_v<n>` predecessor; globbing them in reports duplicate
labels that are not in the document at all.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import DIR  # noqa: E402

LABEL = re.compile(r"\\label\{([^}]+)\}")
REF = re.compile(r"\\(?:eq|c|C)?ref\{([^}]+)\}")


def _build_graph():
    """Every .tex the manuscript reaches, in order, resolved relative to the manuscript dir."""
    seen, out = set(), []

    def walk(rel):
        path = os.path.join(DIR, rel)
        if rel in seen or not os.path.exists(path):
            return
        seen.add(rel)
        out.append(rel)
        txt = open(path, encoding="utf-8").read()
        for m in re.finditer(r"\\input\{([^}]+)\}", txt):
            n = m.group(1)
            walk(n if n.endswith(".tex") else n + ".tex")

    walk("iclr_2027.tex")
    return out


def _labels_and_refs():
    labels, refs = {}, {}
    for rel in _build_graph():
        txt = open(os.path.join(DIR, rel), encoding="utf-8").read()
        for m in LABEL.finditer(txt):
            labels.setdefault(m.group(1), []).append(rel)
        for m in REF.finditer(txt):
            refs.setdefault(m.group(1), []).append(rel)
    return labels, refs


def test_the_build_graph_is_not_empty():
    """A guard that resolves to no files is a guard that passes by never running (caution (j))."""
    g = _build_graph()
    assert len(g) > 10, g
    assert "sections/experiments.tex" in g, g


def test_no_label_is_declared_twice_in_the_build():
    labels, _ = _labels_and_refs()
    dups = {k: v for k, v in labels.items() if len(v) > 1}
    assert not dups, f"a \\ref to these silently resolves to the LAST declaration: {dups}"


def test_every_ref_resolves_inside_the_build():
    labels, refs = _labels_and_refs()
    missing = sorted(k for k in refs if k not in labels)
    assert not missing, f"referenced but never declared in an \\input file: {missing}"


def test_every_float_is_referenced_somewhere():
    labels, refs = _labels_and_refs()
    floats = {k for k in labels if k.split(":")[0] in ("fig", "tab")}
    assert len(floats) >= 15, sorted(floats)
    orphans = sorted(floats - set(refs))
    assert not orphans, f"floats the prose never sends the reader to: {orphans}"
