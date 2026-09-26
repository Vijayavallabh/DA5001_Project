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
    order, kinds = [], {"proposition": 0, "theorem": 0, "lemma": 0}
    pat = re.compile(r"\\begin\{(proposition|theorem|lemma)\}(?:\[[^\]]*\])?\s*\\label\{([^}]+)\}")
    for rel in _build_graph():
        txt = open(os.path.join(DIR, rel), encoding="utf-8").read()
        for m in pat.finditer(txt):
            kinds[m.group(1)] += 1
            order.append((m.group(1), kinds[m.group(1)], m.group(2)))
    got = {lab: (kind, n) for kind, n, lab in order}
    # v13 (2026-09-25, seventh review round): Section 2 now states the vacuity threshold before the
    # certificate that does not grow, so the threshold is Proposition 1 and selection Proposition 2.
    # Every in-text reference goes through \ref (checked: no hard-coded "Proposition~<n>" outside
    # comments), so only this map had to move.
    want = {
        "prop:threshold": ("proposition", 1),
        "prop:selection": ("proposition", 2),
        "prop:sparse": ("proposition", 3),
        "prop:imitation": ("proposition", 4),
        "prop:outrun": ("proposition", 5),
        "thm:nfl": ("lemma", 1),  # v16: the Donsker-Varadhan principle, relabelled a lemma at review 8's request
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
    # v10 (2026-09-24) cites a FAMILY of files, results/onset_prediction_*.md, for the
    # registrations. A pattern resolves when it matches at least one file; a literal path must exist.
    import glob
    missing = {p: sorted(w) for p, w in cited.items()
               if not (glob.glob(os.path.join(repo, p)) if "*" in p
                       else os.path.exists(os.path.join(repo, p)))}
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


def test_no_ladder_sentence_cites_the_appendix_that_has_no_ladder_in_it():
    r"""Caution (aj): a \ref that resolves to the WRONG content. Tectonic exits 0, `??` is 0 and a
    real appendix letter renders -- only a reader who follows the click sees it.

    Two sentences, one of them in the main text, sent the reader to `app:scaling` for the
    fine-tuning-length and seed ladders. That appendix measures the margin `s(x)/c_use` over ten safe
    models across three corpora and contains no ladder at all; the ladders are in `app:collapse`.
    Found by the read-through after the 2026-09-19 appendix reduction.

    The guard is stated as a property of the target rather than as a list of citations: whichever
    section the ladder sentences point at, `app:scaling` is not allowed to be it while its own text
    reports no ladder. If a ladder is ever moved into that appendix this fails and says to re-check
    the wording, which is the safe direction.
    """
    import re as _re
    # v10 (2026-09-24): app:scaling and app:collapse are PARAGRAPH labels inside Appendix G, so a
    # label's target is the text from the label to the next \paragraph or \section, not a section.
    body_of = {}
    for rel in _build_graph():
        txt = " ".join(open(os.path.join(DIR, rel), encoding="utf-8").read().split())
        for m in _re.finditer(r"\\label\{(app:scaling|app:collapse)\}", txt):
            ends = [e for e in (txt.find("\\paragraph{", m.end()), txt.find("\\section{", m.end())) if e > 0]
            body_of[m.group(1)] = txt[m.end(): min(ends) if ends else len(txt)]
    assert "app:scaling" in body_of and "app:collapse" in body_of, sorted(body_of)

    # the premise: the scaling target reports no ladder, the collapse one does (its endpoints)
    assert "$0.8756$" not in body_of["app:scaling"] and "ladder" not in body_of["app:scaling"], \
        "a ladder moved into app:scaling; re-check every citation before relaxing this"
    assert "$0.8756$" in body_of["app:collapse"] and "$1.3477$" in body_of["app:collapse"], \
        "the ladders are no longer reported at app:collapse; the citations need re-pointing"

    # and no sentence sends a reader there for one
    bad = []
    for rel in _build_graph():
        flat = " ".join(open(os.path.join(DIR, rel), encoding="utf-8").read().split())
        for m in _re.finditer(r"Appendix~\\ref\{app:scaling\}", flat):
            window = flat[max(0, m.start() - 220): m.end() + 80]
            if "ladder" in window:
                bad.append(f"{rel}: a ladder sentence cites app:scaling, which has no ladder in it")
    assert not bad, bad


def test_no_numbered_appendix_heading_is_unreachable():
    """A \\section or \\subsection in the appendix that nothing \\refs is a whole appendix the
    reader never learns exists. app:second-anchor was exactly that until 2026-09-20.

    WHY ONLY NUMBERED HEADINGS. The first version of this guard required a \\ref for EVERY app:
    label and found five more -- but almost every app: label in this paper sits on a
    \\paragraph, which LaTeX does not number, so \\ref to one renders the enclosing SECTION's
    letter. That is this paper's established convention (app:vetting, app:saturation,
    app:frontier and app:blocklist all work that way and are cited throughout): the reader is sent
    to the right appendix, not the exact paragraph. Requiring a ref for those produced three
    sentences naming the same appendix letter twice, which
    test_no_sentence_names_the_same_number_twice caught and which is a worse defect than the one
    being fixed. A paragraph label is reachable through its section; a section is reachable only
    through a \\ref.
    """
    import glob as _glob
    import os as _os
    import re as _re
    from tests.manuscript import DIR
    files = [f for f in sorted(_glob.glob(_os.path.join(DIR, "sections", "*.tex")))
             + [_os.path.join(DIR, "iclr_2027.tex")]
             if not _re.search(r"_v\d+_\d{4}-\d{2}-\d{2}\.tex$", f)]
    assert len(files) > 10, "section files not found; this guard must not pass by never running"
    numbered, refs = set(), set()
    for f in files:
        lines = [ln for ln in open(f, encoding="utf-8").read().split("\n")
                 if not ln.lstrip().startswith("%")]
        text = "\n".join(lines)
        for m in _re.finditer(r"\\(sub)?section\{[^}]*\}\s*\\label\{(app:[^}]+)\}", text):
            numbered.add(m.group(2))
        refs |= set(_re.findall(r"\\ref\{(app:[^}]+)\}", text))
    assert numbered, "no numbered appendix headings found; the guard would be vacuous"
    orphans = sorted(numbered - refs)
    assert not orphans, f"numbered appendix headings nothing points at: {orphans}"
