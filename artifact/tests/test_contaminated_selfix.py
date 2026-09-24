"""feat-179 Part A in the manuscript: every cell of tab:contam and every range the text around it
quotes is rebuilt from results/selector_n256.csv (caution (j), (ai)). The defective-selector numbers
must not come back anywhere (the registration excludes quoting them beside the corrected ones)."""
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import DIR, body  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROWS = [r for r in csv.DictReader(open(os.path.join(ROOT, "results", "selector_n256.csv"), encoding="utf-8"))]
E8 = {}
for r in ROWS:
    if r["event"] == "E_08":
        E8.setdefault(r["anchor"], {})[int(r["n"])] = r
# feat-182's independent re-draw: the registration requires the factor per draw, both draws
E8R = {}
for r in csv.DictReader(open(os.path.join(ROOT, "results", "selector_redraw.csv"), encoding="utf-8")):
    if r["event"] == "E_08":
        E8R.setdefault(r["anchor"], {})[int(r["n"])] = r
NAME = {"llama32_1b": "Llama-3.2-1B", "llama32_3b": "Llama-3.2-3B", "phi35mini": "Phi-3.5-mini",
        "pleias12b": "Pleias-1.2B", "pleias350m": "Pleias-350M", "qwen25_7b": "Qwen2.5-7B",
        "kl3m170m": "KL3M-170M", "kl3m520m": "KL3M-520M", "kl3m17b": "KL3M-1.7B", "kl3m37b": "KL3M-3.7B",
        "opencalm1b": "OpenCALM-1B", "opencalm3b": "OpenCALM-3B"}


def _table():
    apx = open(os.path.join(DIR, "sections", "appendix_selection.tex"), encoding="utf-8").read()
    t = apx[apx.index("\\label{tab:contam}"):]
    t = t[:t.index("\\bottomrule")]
    out = {}
    for line in t.splitlines():
        cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
        if cells[0] in NAME.values():
            out[cells[0]] = cells[1:]
    return out


def _num(cell):
    return None if cell == "---" else float(cell.strip("$"))


def test_every_cell_of_the_contamination_table_rounds_from_the_csv():
    tab = _table()
    assert set(tab) == set(NAME.values()), sorted(tab)
    for tag, d in E8.items():
        cells = tab[NAME[tag]]
        for k, n in enumerate((1, 8, 64, 256)):
            assert abs(_num(cells[k]) - float(d[n]["rate"])) < 5e-3, (tag, n)
        assert abs(_num(cells[4]) - float(d[256]["per_draw_rate"])) < 5e-5, tag
        assert abs(_num(cells[5]) - float(d[256]["cover"])) < 5e-3, tag
        for k, n in ((6, 64), (7, 256)):
            first, second = d[n]["amplification_vs_n1"], E8R[tag][n]["amplification_vs_n1"]
            if not first and not second:
                assert cells[k] == "---", (tag, n, cells[k])
                continue
            got = [c.strip() for c in cells[k].split("\\,/\\,")]
            assert len(got) == 2, (tag, n, cells[k], "both draws, per draw")
            for c, a in zip(got, (first, second)):
                assert (c == "---") == (a == ""), (tag, n, c, a)
                if a:
                    assert abs(_num(c) - float(a)) < 5e-3, (tag, n, c, a)


def test_the_ranges_the_text_quotes_are_the_csvs():
    _ranges(E8, first=True)
    _ranges(E8R, first=False)


def _ranges(E8, first):
    apx = body("appendix_selection.tex")
    lead = [t for t in E8 if E8[t][1]["amplification_vs_n1"] or float(E8[t][256]["pool_events"]) > 0]
    a64 = [float(E8[t][64]["amplification_vs_n1"]) for t in lead if E8[t][64]["amplification_vs_n1"]]
    a256 = [float(E8[t][256]["amplification_vs_n1"]) for t in lead if E8[t][256]["amplification_vs_n1"]]
    pd64 = [float(E8[t][64]["amplification_vs_per_draw"]) for t in lead]
    pd256 = [float(E8[t][256]["amplification_vs_per_draw"]) for t in lead]
    np256 = [256 * int(E8[t][256]["pool_events"]) / int(E8[t][256]["pool_draws"]) for t in lead]
    zero = [t for t in E8 if int(E8[t][256]["pool_events"]) == 0]
    for lo, hi in ((min(a64), max(a64)), (min(pd64), max(pd64)), (min(pd256), max(pd256)),
                   (min(np256), max(np256))):
        assert re.search(rf"\${lo:.2f}\$ to \${hi:.2f}|\${lo:.1f}\$ to \${hi:.1f}", apx), (lo, hi)
    pat = "|".join(re.escape(f"${min(a256):{f}}$ to ${max(a256):.1f}\\times") for f in (".2f", ".1f"))
    assert re.search(pat, apx), (min(a256), max(a256))
    words = {6: "Six"}
    assert f"{words[len(zero)]} anchors never enter $E$" in apx, len(zero)
    if first:
        assert f"at most ${max(a64):.1f}/64$" in apx


def test_the_defective_selector_numbers_are_gone_everywhere():
    live = [f for f in glob.glob(os.path.join(DIR, "sections", "*.tex")) if not re.search(r"_v\d", f)]
    live.append(os.path.join(DIR, "iclr_2027.tex"))
    for f in live:
        t = " ".join(open(f, encoding="utf-8").read().split())
        # feat-182's re-draw reads 1.0 to 4.0 at n = 256 -- a different quantity from the defective
        # n = 64 range with the same endpoints. Exempt exactly that phrase, and only while the
        # re-draw's CSV really says so (caution (an): scope the exemption to its own property).
        a256 = [float(E8R[k][256]["amplification_vs_n1"]) for k in E8R if E8R[k][256]["amplification_vs_n1"]]
        if (min(a256), max(a256)) == (1.0, 4.0):
            t = t.replace("$1.0$ to $4.0\\times$ at $256$", "")
        for old in ("$1.0$ to $4.0", "$1.0$ and $4.0$", "about $6\\%$ of the allowed", "\\rho = -0.700",
                    "reads $0.010$ against a bound of $n \\times 0.000 = 0$"):
            assert old not in t, (os.path.basename(f), old)


def test_the_multilingual_paragraph_quotes_the_corrected_served_means():
    """feat-179 Part B: the multilingual arm's served means read the pick, so they moved with the
    selector repair (lcs 1.51-1.65 -> 1.51-1.53, ROUGE-L 0.082-0.094 -> 0.070-0.082). The paragraph
    must quote the AFTER columns of results/selector_n256_descriptive.csv, never the before."""
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "results", "selector_n256_descriptive.csv"),
                                           encoding="utf-8")) if r["arm"] == "multilingual"]
    assert {int(r["n"]) for r in rows} == {1, 8, 64}, rows
    lcs = [float(r["lcs_after"]) for r in rows]
    rg = [float(r["rouge_after"]) for r in rows]
    apx = body("appendix_selection.tex")
    sec = apx[apx.index("\\paragraph{Outside English prose.}"):]
    sec = sec[:sec.index("\\paragraph{", 20)]
    assert f"${min(lcs):.2f}$--${max(lcs):.2f}$ words" in sec, (min(lcs), max(lcs))
    assert f"ROUGE-L ${min(rg):.3f}$--${max(rg):.3f}$" in sec, (min(rg), max(rg))
    before = [float(r["rouge_before"]) for r in rows]
    assert f"${max(before):.3f}$" not in sec, "a defective-selector mean is back in the paragraph"


def test_the_extraction_tables_substring_row_is_the_corrected_selectors():
    """grid64 (declared addition to feat-179, read 2026-09-23): tab:extraction's substring row is a
    mean over the SERVED pick, so the padding defect moved it. Columns 1-64 come from the full-grid arm
    re-run on its identical pool (grid64), 256 from the n256 arm; every cell must be the 'after'
    column, and the claim that no other row could move is rebuilt from every draw in both pools."""
    desc = {(r["arm"], int(r["n"])): r for r in
            csv.DictReader(open(os.path.join(ROOT, "results", "selector_n256_descriptive.csv"),
                                encoding="utf-8"))}
    for arm in ("grid64", "n256"):
        assert all(desc[(a, n)]["g0"] == desc[(a, n)]["g1"] == "PASS" for a, n in desc if a == arm), arm
    want = [desc[("grid64", n)]["lcs_after"] for n in (1, 4, 8, 16, 32, 64)] + [desc[("n256", 256)]["lcs_after"]]
    apx = open(os.path.join(DIR, "sections", "appendix_selection.tex"), encoding="utf-8").read()
    t = apx[apx.index("\\label{tab:extraction}"):]
    row = next(l for l in t[:t.index("\\bottomrule")].splitlines() if l.startswith("longest substring"))
    cells = [c.strip() for c in row.rstrip("\\ ").split("&")][1:8]
    assert cells == [f"${float(w):.2f}$" for w in want], (cells, want)
    for f in ("selfix_clean_grid64", "selfix_clean_n256"):
        rows = list(csv.DictReader(open(os.path.join(ROOT, "results", f"{f}_per_passage.csv"),
                                        encoding="utf-8")))
        assert max(float(r["anchor_max_recall"]) for r in rows) == 0.0, f
        assert max(float(r["anchor_max_rouge"]) for r in rows) < 0.5, f
    cap = apx[apx.rfind("\\caption{", 0, apx.index("\\label{tab:extraction}")):apx.index("\\label{tab:extraction}")]
    assert "It is the only row the selector can move" in " ".join(cap.split())
