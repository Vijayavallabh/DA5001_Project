"""Guards for the AC report's points that were still open on 2026-09-23.

The original instruction for that report was "address all of them at the top priority". An audit
against the manuscript on 2026-09-23 found most done and several not: one mathematical statement
still wrong (Proposition 5, guarded in test_review_revisions.py, whose first guard passed on a
different sentence), three tables with no number or caption, forty-five run-in headings with no
terminal punctuation, and four points only partly met. Each guard below is written against the
structure or the data rather than a phrasing, because every one of these had already been "fixed"
once in a way a later edit undid or never reached.

The live files are read off iclr_2027.tex's own \\input lines, so a section added later is covered
without anyone remembering to list it here.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tests.manuscript import tex  # noqa: E402


def _live():
    main = open(tex("iclr_2027.tex"), encoding="utf-8").read()
    names = re.findall(r"^\\input\{sections/([^}]+)\}", main, re.M)
    assert len(names) >= 10, f"only {len(names)} \\input lines found; the parser is wrong"
    return {"iclr_2027.tex": main,
            **{n: open(tex(f"sections/{n}.tex"), encoding="utf-8").read() for n in names}}


def _balanced(s, start):
    """Index just past the brace group opening at s[start-1] == '{'."""
    depth, j = 1, start
    while depth:
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        j += 1
    return j


def test_every_table_is_a_numbered_captioned_float():
    """AC: 'Every standalone table should be assigned a number and a caption.' Three were not --
    each added after the first revision, each a `center` block after a colon. A table a reader
    cannot cite by number is one a reviewer cannot either."""
    bad = []
    for name, s in _live().items():
        floats = [(m.start(), m.end()) for m in
                  re.finditer(r"\\begin\{(table\*?)\}.*?\\end\{\1\}", s, re.S)]
        for m in re.finditer(r"\\begin\{tabular\*?\}", s):
            if not any(a <= m.start() < b for a, b in floats):
                bad.append(f"{name}:{s[:m.start()].count(chr(10)) + 1} (not inside a table float)")
        for a, b in floats:
            body = s[a:b]
            if "\\caption" not in body or "\\label" not in body:
                bad.append(f"{name}:{s[:a].count(chr(10)) + 1} (float without caption or label)")
    assert not bad, "unnumbered or uncaptioned tables:\n  " + "\n  ".join(bad)


def test_every_run_in_heading_ends_in_punctuation():
    """AC: the intro's 'Why this was available...' heading 'lacks terminal punctuation or
    consistent formatting compared to other similar assertions'. It had forty-four siblings."""
    bad = []
    for name, s in _live().items():
        for m in re.finditer(r"\\paragraph\{", s):
            head = s[m.end():_balanced(s, m.end()) - 1].rstrip()
            last = head.rstrip("}")[-1:]
            if last not in ".?!:":
                bad.append(f"{name}: {head[-60:]!r}")
    assert not bad, "run-in headings without terminal punctuation:\n  " + "\n  ".join(bad)


def test_the_scope_is_delimited_early_and_in_related_work():
    """AC: the opening 'Inference-time copyright defences charge, per decoded token...' reads as a
    description of the whole class, when the widely deployed defences (blocklists, filtering)
    certify nothing. Section 2 had a Scope paragraph after the first revision; the abstract's first
    sentence -- the one the AC quoted -- still said it of every defence."""
    live = _live()
    abstract = " ".join(live["iclr_2027.tex"].split(r"\begin{abstract}")[1]
                        .split(r"\end{abstract}")[0].split())
    first = abstract.split(". ")[0]
    assert "inference-time copyright defences" in first.lower(), first
    assert first.lower().startswith("certified"), (
        "the abstract's first sentence again describes every inference-time defence: " + first)
    intro = " ".join(live["iclr_intro"].split())
    assert "Certified decoders offer something cheaper" in intro, \
        "the introduction's opening no longer scopes the mechanisms it describes"
    rw = " ".join(live["related_work_v4"].split())
    i = rw.find(r"\textbf{Scope.}")
    assert i != -1, "Section 2's Scope paragraph is gone"
    scope = rw[i:i + 900]
    for key in ("ippolito2023preventing", "wei2024cotaeval", "divergence certificate"):
        assert key in scope, f"the Scope paragraph no longer names {key}"


def _csv(name):
    import csv
    return list(csv.DictReader(open(os.path.join(ROOT, "results", name), encoding="utf-8")))


def _table(src, label):
    """The tabular body of the float carrying `label`, whitespace-normalised."""
    i = src.index(r"\label{%s}" % label)
    a, b = src.rfind(r"\begin{table}", 0, i), src.index(r"\end{table}", i)
    return " ".join(src[a:b].split())


def test_the_judged_workload_is_described_and_every_cell_is_the_csvs():
    """AC: 'The manuscript lacks details regarding the composition, length distribution,
    complexity, and source domains of this prompt set.' Composition was stated; the rest is now a
    table, built from results/prompt_set_profile.csv, which counts EXACTLY the judged ids."""
    rows = {r["cls"]: r for r in _csv("prompt_set_profile.csv")}
    judged = _csv("order_averaged_h2h_per_prompt.csv")
    assert sum(int(r["n"]) for r in rows.values()) == len(judged) == 500
    assert {c: int(r["n"]) for c, r in rows.items()} == {"neutral": 200, "factual": 150,
                                                        "creative": 150}, \
        "the class counts no longer match what Section 4.1 states"
    # the profile's anchor win rates must reproduce the committed opponent strength, 0.555
    mean = sum(int(r["n"]) * float(r["anchor_win"]) for r in rows.values()) / 500
    assert abs(mean - 0.445) < 5e-4, f"weighted anchor win {mean:.4f} is not the committed 0.445"
    t = _table(open(tex("sections/appendix_selection.tex"), encoding="utf-8").read(),
               "tab:promptset")
    for r in rows.values():
        med = float(r["prompt_words_median"])
        cell = (f"${int(med) if med == int(med) else med}$ "
                f"$[{float(r['prompt_words_q1']):.0f}, {float(r['prompt_words_q3']):.0f}]$, "
                f"${r['prompt_words_min']}$--${r['prompt_words_max']}$")
        assert cell in t, f"{r['cls']}: prompt-length cell {cell} is not the CSV's"
        assert f"${float(r['anchor_win']):.4f}$" in t, f"{r['cls']}: anchor win moved"
        assert f"& ${r['served_with_header']}$ &" in t, f"{r['cls']}: header count moved"
    exp = " ".join(open(tex("sections/experiments.tex"), encoding="utf-8").read().split())
    assert r"\ref{tab:promptset}" in exp, "Section 4.1 no longer points at the workload's profile"


def test_the_latency_table_carries_the_clock_shares_it_is_cited_for():
    """AC: 'the reward pass is 9.3% ... while the draws are 90.7%' was bold prose beside a table
    that did not show it. The shares are now a column, computed from the table's own CSV."""
    import csv
    q = {r[0]: r for r in csv.reader(open(os.path.join(ROOT, "results", "serving_latency.csv"),
                                          encoding="utf-8")) if r}
    tot = float(q["selection n=64, total"][1])
    t = _table(open(tex("sections/appendix_selection.tex"), encoding="utf-8").read(),
               "tab:latency")
    assert "& share" in t, "the share column is gone"
    d = float(q["selection n=64, anchor draws"][1]) / tot * 100
    r = float(q["selection n=64, reward pass"][1]) / tot * 100
    assert f"{d:.1f}\\%" in t and f"{r:.1f}\\%" in t, (d, r)


def test_the_main_text_answers_the_evaluation_asymmetry_where_it_arises():
    """AC: selection ranks with a reward model the metered decoder never gets, so the gain may be
    the scorer's. The answer is an arm (Table tab:parity: the same reward, handed to the meter,
    moves it by nothing), and a reader of Section 4.2 must be sent to it -- scoped to the task it
    was measured on, since it is one task."""
    exp = " ".join(open(tex("sections/experiments.tex"), encoding="utf-8").read().split())
    i = exp.find(r"\ref{tab:parity}")
    assert i != -1, "Section 4.2 no longer points at the parity arm"
    assert "on TriviaQA" in exp[max(0, i - 160):i], "the parity pointer lost its task scope"
    # 'gains nothing' is a claim about every admitted metered cell, so it is checked on all of
    # them rather than on the one the table prints. Gate-failed budgets carry no gain and are the
    # registration's own exclusions (k=0.5 and k=1, FAIL-G3).
    met = [r for r in _csv("meter_parity.csv")
           if r["arm"].startswith("metered") and r["gate"] == "PASS" and int(r["n"]) > 1]
    assert len(met) == 8, f"expected 2 admitted budgets x 4 values of n, found {len(met)}"
    for r in met:
        assert float(r["gain_lo95"]) < 0 < float(r["gain_hi95"]), (
            f"{r['arm']} n={r['n']}: the reward now moves the meter clear of zero; "
            "'the meter gains nothing' is stale")


def test_the_conclusions_price_sentence_is_not_dash_interrupted():
    """AC: 'the price is not, the scorer being 9.3% of the clock, so the bar we have not cleared
    is the cost of drawing' was 'syntactically complex and slightly difficult to parse'. The
    committed concession it carries (where that bar sits is open) is guarded elsewhere."""
    close = " ".join(open(tex("sections/iclr_closing.tex"), encoding="utf-8").read().split())
    assert r"The \emph{price} is not ---" not in close
    assert r"The \emph{price} is the drawing, not the scorer's" in close
