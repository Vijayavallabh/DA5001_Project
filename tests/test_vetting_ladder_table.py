"""feat-180 as the paper prints it (results/onset_prediction_vetting_ladder.md, scored 2026-09-23).

Every cell of tab:vetladder is the leaking count in results/vetting_ladder.csv; the recommendation the
Ethics Statement gives is the one V1's reading licenses; the licensed anchors' pass is claimed only as
far as the rungs that measured it; and every number the prose quotes about the ladder is rebuilt from
the CSV, because prose that cites a table is not thereby taken from it (caution (ai)).
"""
import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis import vetting_ladder as v  # noqa: E402
from tests.manuscript import body, tex  # noqa: E402

NAMES = {"Llama-3.1-70B": "llama70b", "OLMo-2-13B": "olmo2_13b", "OLMo-2-7B": "olmo2_7b",
         "TinyComma-1.8B": "tinycomma", "Comma-7B": "comma7b", "Comma-1T": "comma1t",
         "KL3M-1.7B": "kl3m17b", "Pleias-1.2B": "pleias12b", "Pleias-3B": "pleias3b"}
RUNGS = (20, 35, 50, 75, 100, 150, 200)
WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}


def _rows():
    with open(os.path.join(ROOT, "results", "vetting_ladder.csv"), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _cells():
    """{(tag, L): (count or None, dagger)} read off the tabular"""
    app = open(tex("sections/appendix_selection.tex"), encoding="utf-8").read()
    tab = app[app.index(r"\label{tab:vetladder}"):]
    tab = tab[:tab.index(r"\end{tabular}")]
    out = {}
    for line in tab.splitlines():
        parts = [p.strip() for p in line.rstrip("\\ ").split("&")]
        if parts[0] not in NAMES:
            continue
        assert len(parts) == 2 + len(RUNGS), line
        for L, c in zip(RUNGS, parts[2:]):
            m = re.fullmatch(r"\$(\d+)(\^\\dagger)?\$", c)
            out[(NAMES[parts[0]], L)] = (int(m.group(1)), bool(m.group(2))) if m else (None, False)
            assert m or c == "---", (parts[0], L, c)
    return out


def _ethics():
    t = " ".join(open(tex("iclr_2027.tex"), encoding="utf-8").read().split())
    i = t.index(r"\section*{Ethics Statement}")
    return t[i: t.index(r"\section*", i + 10)]


def _vetting():
    app = body("appendix_selection.tex")
    i = app.index(r"\label{app:vetting}")
    return app[i: app.index(r"\paragraph", i)]


def test_every_cell_of_the_table_is_the_csv():
    rows, cells = _rows(), _cells()
    have = {(r["model"], int(r["prefix_tokens"])): r for r in rows}
    assert {t for t, _ in cells} == {r["model"] for r in rows}, "a model is missing from one side"
    for (t, L), (count, dagger) in cells.items():
        r = have.get((t, L))
        if r is None:
            assert count is None, f"{t} at L={L} was never run, yet the table prints {count}"
            continue
        assert count == int(r["leaking"]), (t, L, count, r["leaking"])
        assert dagger == (r["source"] == "this arm, host B"), (t, L, "dagger must mark host B")


def test_the_recommendation_is_the_one_v1_licenses():
    rows = _rows()
    falls = []
    for t in {r["model"] for r in rows if r["role"] != "openly licensed"}:
        c = sorted((int(r["prefix_tokens"]), int(r["leaking"])) for r in rows if r["model"] == t)
        falls += [(t, a, b) for (a, ka), (b, kb) in zip(c, c[1:]) if ka - kb >= v.DROP]
    eth, vet = _ethics(), _vetting()
    if falls:                                           # NON-MONOTONE: a schedule, not a length
        assert r"\emph{every} prefix length" in eth, "the Ethics Statement must recommend a schedule"
        assert "attack surface: run the check at the longest genuine prefix" not in eth, \
            "the single-length rule V1 refuted is back"
        assert "not monotone" in vet and r"\emph{unmeasured}" in vet, \
            "the licensed anchors' short rungs must be named as unmeasured"
    else:
        assert "not monotone" not in eth + vet, "V1 now reads MONOTONE; the schedule is not licensed"


def test_the_licensed_pass_is_claimed_only_as_far_as_it_was_measured():
    rows = _rows()
    lic = sorted({r["model"] for r in rows if r["role"] == "openly licensed"})
    past = [r for r in rows if r["role"] == "openly licensed" and int(r["prefix_tokens"]) > 100]
    leaks = [r["model"] for r in past if int(r["leaking"]) > 0]
    six = WORD[len(lic)]
    claim = f"all {six} read $0$ of $50$ at $150$ and at $200$ tokens"
    if leaks:
        assert claim not in _vetting(), f"PASS BREAKS at {leaks}, and the appendix says it holds"
    else:
        assert {int(r["prefix_tokens"]) for r in past} == {150, 200} and claim in _vetting()
        assert f"The {six} anchors used here still read $0$ at $150$ and $200$" in _ethics()
    # the count the paper gives is the count screened: TinyComma sits in the 70B control's anchor slot
    assert f"each of the {six} openly licensed anchors" in _ethics()
    assert f"all {six} anchors used here" in _vetting()
    assert f"at all {six} licensed anchors" in body("experiments.tex")
    live = _ethics() + body("appendix_selection.tex", "experiments.tex", "selection.tex",
                            "iclr_intro.tex")
    for stale in ("five openly licensed", "all five anchors", "five licensed"):
        assert stale not in live, stale


def test_every_ladder_number_in_the_prose_is_the_csv():
    k = {(r["model"], int(r["prefix_tokens"])): int(r["leaking"]) for r in _rows()}
    b = lambda L: k[("llama70b", L)]  # noqa: E731
    vet, eth = _vetting(), _ethics()
    assert f"(${b(20)}$ and ${b(35)}$ of $50$ passages)" in vet
    assert f"is seen from $50$ (${b(50)}$)" in vet and b(50) >= v.SEES > b(35)
    assert f"leaks on ${b(150)}$ at $150$, where the screen at a hundred finds ${b(100)}$" in vet
    o = lambda L: k[("olmo2_13b", L)]  # noqa: E731
    assert o(100) == o(150), "the prose says 'at 100 and at 150' with one count"
    assert (f"\\texttt{{OLMo-2-13B}} leaks on ${o(100)}$ passages at $100$ and at $150$ and on "
            f"${o(200)}$ at $200$") in vet
    with open(os.path.join(ROOT, "results", "vetladder_L200_olmo2_13b_per_passage.csv"),
              encoding="utf-8") as fh:
        full = sum(float(r["anchor_max_recall"]) >= 1.0 for r in csv.DictReader(fh))
    assert f"It still reproduces {WORD[full]} passage in full there" in vet
    assert (f"leaks on ${b(20)}$ passage of $50$ at twenty raw tokens, ${b(50)}$ at fifty and "
            f"${b(150)}$ at $150$") in eth


def test_the_host_check_is_quoted_as_the_scorer_reads_it():
    word, d = v.host_check(os.path.join(ROOT, "results"))
    assert word == "PASS", (word, d)
    assert (f"reaches recall $0.01$ on ${d['host_b']}$ of ${d['n']}$ passages against "
            f"${d['local']}$ here ($z = {d['z']:.2f}$)") in _vetting()
