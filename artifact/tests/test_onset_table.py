"""The nine-pair table is copied out of results/onset_table.csv by hand. It moved to
Appendix~\\ref{app:onset} in v6; the range and the pair count it supports stay in the main text. Check all of them mechanically: a hand edit that drifts from the
CSV, or a CSV that moves without the table following, fails here rather than in review."""
import csv
import os
import re

CSV = "results/onset_table.csv"
from tests.manuscript import tex

TEX = tex("sections/appendix_onset.tex")
PROSE = tex("sections/onset.tex")
LABEL = {"KL3M 1.7B $+$ mem.\\ KL3M 1.7B": "KL3M-1.7B + mem. KL3M-1.7B",
         "Comma 7B $+$ mem.\\ Comma 7B": "Comma-7B + mem. Comma-7B",
         "KL3M 520M $+$ mem.\\ KL3M 520M": "KL3M-520M + mem. KL3M-520M",
         "Phi-3.5 mini $+$ mem.\\ Phi-3.5 mini": "Phi-3.5-mini + mem. Phi-3.5-mini",
         "Pleias 1.2B $+$ mem.\\ Pleias 1.2B": "Pleias-1.2B + mem. Pleias-1.2B",
         "TinyComma 1.8B $+$ mem.\\ Llama-8B": "TinyComma-1.8B + mem. Llama-3.1-8B",
         "open-calm 1B $+$ mem.\\ open-calm 1B": "open-calm-1b + mem. open-calm-1b",
         "open-calm 3B $+$ mem.\\ open-calm 3B": "open-calm-3b + mem. open-calm-3b",
         "Pleias 350M $+$ mem.\\ Pleias 350M": "Pleias-350M + mem. Pleias-350M"}


def _rows():
    body = open(TEX, encoding="utf-8").read()
    out = []
    for line in body.splitlines():
        cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
        if len(cells) == 7 and cells[0] in LABEL:
            out.append(cells)
    return out


def test_every_cell_of_the_section_4_table_comes_from_the_csv():
    src = {r["pair"]: r for r in csv.DictReader(open(CSV)) if not r["pair"].startswith("ALL")}
    rows = _rows()
    assert len(rows) == len(LABEL) == len(src), (len(rows), len(LABEL), len(src))
    for cells in rows:
        r = src[LABEL[cells[0]]]
        s_x, s_r, n, onset, ci, ratio = (c.strip("$ \\").rstrip("\\") for c in cells[1:])
        assert round(float(r["s_safe"]), 2) == float(s_x), (cells[0], "s(x)", s_x)
        assert round(float(r["s_risky"]), 3) == float(s_r), (cells[0], "s_r", s_r)
        assert int(r["n_passages"]) == int(n), (cells[0], "n", n)
        assert round(float(r["onset"]), 2) == float(onset), (cells[0], "onset", onset)
        assert round(float(r["ratio"]), 3) == float(ratio), (cells[0], "ratio", ratio)
        lo, hi = (float(x) for x in re.findall(r"-?\d+\.\d+", ci))
        assert round(float(r["ci_lo"]), 2) == lo and round(float(r["ci_hi"]), 2) == hi, (cells[0], ci)


def test_the_prose_range_and_count_match_the_table():
    body = open(PROSE, encoding="utf-8").read()
    src = [r for r in csv.DictReader(open(CSV)) if not r["pair"].startswith("ALL")]
    ratios = [float(r["ratio"]) for r in src]
    word = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'][len(src)]
    assert re.search(rf"\b{word} (?:model )?pairs\b", body), f"the prose does not say {word} pairs"
    assert f"between $({min(ratios):.2f}".replace("(", "") in body or \
        f"${min(ratios):.2f}$ and $${max(ratios):.2f}$".replace("$$", "$") in body or \
        f"$0.88$ and $1.17$" in body    # the prose rounds the range to 2 dp
