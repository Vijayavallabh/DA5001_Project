"""The appendix's twelve-pair table is 72 numbers copied out of results/order_law.csv by hand.

Six of them were one off in the last digit, all from double rounding: window_factor was stored at
3 significant figures and the paper rounded that rounded value again to 2 or 3. The column is now
written at 6 s.f.; this test checks the invariant that made the error detectable in the first place
(window_factor is exp(nats_per_window)) and then checks every cell of the table itself, so a hand
edit that drifts from the CSV fails here rather than in review.
"""
import collections
import csv
import math
import os
import re

CSV = "results/order_law.csv"
TEX = os.path.expanduser("~/sub/satml/sections/appendix_robustness.tex")
LABEL = {"Pleias 350M": "Pleias-350M", "Pleias 1.2B": "Pleias-1.2B", "Comma 7B": "Comma-7B",
         "KL3M 170M": "KL3M-170M", "TinyComma 1.8B": "TinyComma-1.8B", "KL3M 3.7B": "KL3M-3.7B",
         "KL3M 520M": "KL3M-520M", "Llama-3.2 3B": "Llama-3.2-3B", "KL3M 1.7B": "KL3M-1.7B",
         "Llama-3.2 1B": "Llama-3.2-1B", "Qwen2.5 7B": "Qwen2.5-7B", "Phi-3.5 mini": "Phi-3.5-mini"}
CELLS = [(1.0, 2.0), (1.0, 4.0), (1.0, 8.0), (3.0, 2.0), (3.0, 4.0), (3.0, 8.0)]


def _parse(cell):
    """(value, significant figures) for a table cell like `$2.5{\\cdot}10^{7}$` or `$47.5$`."""
    s = cell.strip().strip("$").replace(" ", "")
    m = re.match(r"^([\d.]+)\{?\\cdot\}?10\^\{(-?\d+)\}$", s)
    mant, exp = (m.group(1), int(m.group(2))) if m else (s, 0)
    return float(mant) * 10 ** exp, len(mant.replace(".", "").lstrip("0")) or 1


def _sigfig(x, n):
    return float("%.*e" % (n - 1, x)) if x else 0.0


def test_window_factor_is_exp_of_nats_per_window():
    if not os.path.exists(CSV):
        return
    rows = list(csv.DictReader(open(CSV)))
    assert rows
    for r in rows:
        want = math.exp(float(r["nats_per_window"]))
        # nats_per_window is stored to 4 dp, so the factor carries ~5e-5 of relative slack
        assert abs(float(r["window_factor"]) - want) <= 2e-4 * max(1.0, want), r


def test_every_cell_of_the_appendix_table_rounds_from_the_csv():
    if not (os.path.exists(CSV) and os.path.exists(TEX)):
        return
    t = collections.defaultdict(dict)
    for r in csv.DictReader(open(CSV)):
        t[r["pair"]][(float(r["published_k"]), float(r["alpha"]))] = float(r["window_factor"])
    checked = 0
    for line in open(TEX, encoding="utf-8"):
        line = line.rstrip()
        if "&" not in line or not line.endswith(r"\\"):
            continue
        c = [x.strip() for x in line.rstrip("\\").split("&")]
        if c[0] not in LABEL:
            continue
        for (k, a), cell in zip(CELLS, c[2:8]):
            paper, sig = _parse(cell)
            want = _sigfig(t[LABEL[c[0]]][(k, a)], sig)
            assert paper == want, f"{c[0]} k={k} alpha={a}: paper {cell} != {want} from {CSV}"
            checked += 1
    assert checked == 72, f"expected 72 cells, matched {checked}"
