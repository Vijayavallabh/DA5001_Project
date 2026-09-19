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
    # Floor lowered 15 -> 9 on 2026-09-19: the appendix reduction retired appendix_seed (three
    # figures) and cut eight floats whose content is carried by a table, a proof or the paragraph
    # beside them. The floor exists only so a broken build graph cannot make this guard vacuous;
    # the assertion that matters is the next one, that nothing printed is left unreferenced.
    assert len(floats) >= 9, sorted(floats)
    orphans = sorted(floats - set(refs))
    assert not orphans, f"floats the prose never sends the reader to: {orphans}"


def test_the_propositions_number_the_way_the_paper_talks_about_them():
    """Numbering is by declaration order in the build, and the prose depends on it everywhere --
    "Proposition 3 then leaves a per-token meter vacuous or trivial", "Proposition 1 assumes
    nothing about the score". Moving a statement between sections renumbers every later one, and
    nothing reports it: the \\ref still resolves and prints a real number, just a different one.
    That is caution (z) without the duplicate label -- the same silent-renumber failure by a
    different route.
    """
    order, kinds = [], {"proposition": 0, "theorem": 0}
    pat = re.compile(r"\\begin\{(proposition|theorem)\}(?:\[[^\]]*\])?\s*\\label\{([^}]+)\}")
    for rel in _build_graph():
        txt = open(os.path.join(DIR, rel), encoding="utf-8").read()
        for m in pat.finditer(txt):
            kinds[m.group(1)] += 1
            order.append((m.group(1), kinds[m.group(1)], m.group(2)))
    got = {lab: (kind, n) for kind, n, lab in order}
    want = {
        "prop:selection": ("proposition", 1),
        "prop:threshold": ("proposition", 2),
        "prop:sparse": ("proposition", 3),
        "prop:imitation": ("proposition", 4),
        "prop:outrun": ("proposition", 5),
        "thm:nfl": ("theorem", 1),
    }
    assert got == want, (got, want)


def test_every_repo_path_the_manuscript_cites_exists():
    r"""A \texttt{results/...} naming a file that is not there is a reproducibility defect an
    artifact reviewer hits on their first click, and nothing in the toolchain looks: tectonic does
    not resolve paths, and analysis/audit_numbers.py checks numbers, not filenames.

    The extraction is the subtle part. Caution (af) put \allowbreak after every separator inside
    long \texttt arguments so they can break across a line, and warns that any consumer greping
    paths out of the manuscript must strip it. Stripping the macro alone is not enough -- it
    leaves the space that followed it, turning results/onset_prediction_x.md into
    "results/ onset_ prediction_ x.md", which then "does not exist" for every path at once. That
    happened on the first run of this check (2026-09-18) and reported 34 false positives.
    """
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prefixes = ("results/", "analysis/", "scripts/", "tests/", "dap/", "a_patch/",
                "figures/", "recipes/", "data/", "output/")
    cited = {}
    for rel in _build_graph():
        txt = " ".join(open(os.path.join(DIR, rel), encoding="utf-8").read().split())
        for m in re.finditer(r"\\texttt\{([^}]*)\}", txt):
            raw = re.sub(r"\\allowbreak\s*", "", m.group(1))
            raw = raw.replace("\\_", "_").replace("\\", "").replace(" ", "")
            if raw.startswith(prefixes):
                cited.setdefault(raw, set()).add(rel)
    # The manuscript no longer cites repo paths: they were internal bookkeeping and were
    # removed on 2026-09-19. There is nothing to require a minimum of; what still matters
    # is that any path it DOES cite resolves, which the assertion below checks.
    missing = {p: sorted(w) for p, w in cited.items()
               if not os.path.exists(os.path.join(repo, p))}
    assert not missing, f"cited but absent from the repo: {missing}"


def test_every_citation_key_resolves_to_a_bib_entry():
    """An undefined citation renders as [?], not as ??, so the check the project runs after every
    edit ("0 unresolved references") does not see it -- and the CFP makes a non-existent reference
    a desk-reject item. 74 keys against 164 entries as of 2026-09-18."""
    keys = set()
    for rel in _build_graph():
        txt = open(os.path.join(DIR, rel), encoding="utf-8").read()
        for m in re.finditer(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}", txt):
            keys |= {k.strip() for k in m.group(1).split(",")}
    assert len(keys) > 40  # threshold lowered 2026-09-19: the appendix was cut from 52 to ~37 pages, (len(keys), "the cite extractor matched almost nothing")
    bib = os.path.join(DIR, "references.bib")
    assert os.path.exists(bib), bib
    entries = set(re.findall(r"@\w+\{([^,]+),", open(bib, encoding="utf-8").read()))
    missing = sorted(keys - entries)
    assert not missing, f"cited but not in references.bib: {missing}"
