"""Appendix E's nine-row table is five columns of hand-copied numbers. Two of them did not
reproduce from any definition: the `steps to it` column, and open-calm's characters per token, which
disagreed with the seed length in its own row (53.1/20 = 2.66, printed as 2.74). Every column is now
written by analysis/seed_effect.py into results/onset_seed_words.csv; check all 45 cells."""
import csv
import re

from tests.manuscript import tex

CSV = "results/onset_seed_words.csv"
TEX = tex("sections/appendix_seed.tex")
LABEL = {"TinyComma 1.8B": "TinyComma-1.8B", "Pleias 1.2B": "Pleias-1.2B", "Comma 7B": "Comma-7B",
         "Phi-3.5 mini": "Phi-3.5-mini", "Pleias 350M": "Pleias-350M",
         "open-calm 1B": "open-calm-1b", "open-calm 3B": "open-calm-3b",
         "KL3M 520M": "KL3M-520M", "KL3M 1.7B": "KL3M-1.7B"}


def _num(cell):
    return float(re.sub(r"[^0-9.]", "", cell))


def test_every_cell_of_the_appendix_e_seed_table_comes_from_the_csv():
    src = {}
    for r in csv.DictReader(open(CSV)):
        src[r["pair"].split(" + ")[0]] = r
    checked = 0
    for line in open(TEX, encoding="utf-8"):
        cells = [c.strip() for c in line.rstrip().rstrip("\\").split("&")]
        if len(cells) != 6 or cells[0] not in LABEL:
            continue
        r = src[LABEL[cells[0]]]
        want = [round(float(r["chars_per_token"]), 2), round(float(r["seed_chars"]), 1),
                round(float(r["seed_words"]), 1), round(float(r["steps_to_passage"])),
                round(float(r["ratio"]), 3)]
        got = [_num(c) for c in cells[1:]]
        assert got == want, (cells[0], got, want)
        checked += 5
    assert checked == 45, f"expected 45 cells, matched {checked}"


def test_characters_per_token_is_the_seed_length_over_the_seed_tokens():
    """The cell that was wrong was wrong against its own row, not against a distant CSV."""
    for r in csv.DictReader(open(CSV)):
        assert abs(float(r["seed_chars"]) / int(r["seed_tokens"])
                   - float(r["chars_per_token"])) < 1e-6, r["pair"]
