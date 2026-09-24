"""Guards for the v10 restructure (2026-09-24): the figure- and table-driven rewrite.

Caution (al): a number that moves from prose into a figure takes its guard with it, or the guard
passes by never running. v10 moved the headline robustness claims into Figure 3's forest, the meter's
two horns into Figure 2, and the per-response certificate into Table 1, so each is pinned here to the
function or CSV that draws it, and each sentence that summarises a figure is checked against the rows
the figure plots rather than against a string.
"""
import csv
import math
import os
import re
import sys

from manuscript import ROOT, body, caption_of, tex

sys.path.insert(0, os.path.join(ROOT, "figures"))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from make_figures_v4 import h2h_forest_rows, meter_horns_rows  # noqa: E402
from certificate_table import CAP, T_MAX, kl_bound  # noqa: E402

RES = os.path.join(ROOT, "results")


def _rows(name):
    return list(csv.DictReader(open(os.path.join(RES, name), encoding="utf-8")))


def _live_files():
    live, todo = [], ["iclr_2027.tex"]
    while todo:
        f = todo.pop()
        live.append(f)
        src = re.sub(r"(?<!\\)%.*", "", open(tex(f), encoding="utf-8").read())
        todo += [m + ".tex" for m in re.findall(r"\\input\{([^}]*)\}", src)]
    return live


def _tabular(label, name):
    """Rows of the tabular whose float carries \\label{label}, each a list of cleaned cells."""
    t = body(name)
    i = t.index("\\label{%s}" % label)
    a, b = t.index("\\midrule", i), t.index("\\bottomrule", i)
    out = []
    for line in t[a + len("\\midrule"):b].split("\\\\"):
        line = line.replace("\\midrule", "").strip()
        if line:
            out.append([c.strip() for c in line.split("&")])
    return out


def _num(cell):
    m = re.search(r"[-+]?\d+(?:\.\d+)?", cell.replace("{,}", ""))
    return m.group(0) if m else None


def _rounds_to(printed, value):
    """Caution (j): compare at the precision the paper printed, once."""
    d = len(printed.split(".")[1]) if "." in printed else 0
    return f"{value:.{d}f}" == f"{float(printed):.{d}f}"


# ---------------------------------------------------------------- Table 1, the certificate table

def test_the_certificate_csv_is_what_its_script_computes():
    """Recompute every column from the two CSVs the script reads, so a hand edit of the CSV fails."""
    s = float({r["event"]: r for r in _rows("window_vacuity.csv")}["50-token window"]["S_median_nats"])
    spend = {float(r["k"]): float(r["spend_nats"]) for r in _rows("imitation_cost.csv")
             if r["prompt_class"] == "ordinary"}
    got = {r["param"]: r for r in _rows("certificate_table.csv")}
    assert set(got) == {"k=0.5", "k=3", "k=10", "n=8", "n=64"}
    for k in (0.5, 3.0, 10.0):
        r, K = got[f"k={k:g}"], k * T_MAX
        assert float(r["certificate_nats"]) == K
        assert abs(float(r["K_over_Sw"]) - K / s) < 1e-6
        assert abs(float(r["window_bound"]) - kl_bound(K, s)) < 1e-6
        assert float(r["measured_spend_nats"]) == spend[k]
        assert int(r["responses_at_certificate"]) == math.floor(CAP / K)
        assert int(r["responses_at_spend"]) == math.floor(CAP / spend[k])
    for n in (8, 64):
        r, K = got[f"n={n}"], math.log(n)
        assert abs(float(r["certificate_nats"]) - K) < 1e-4
        assert abs(float(r["window_bound_log10"]) - (K - s) / math.log(10)) < 1e-4
        assert int(r["responses_at_certificate"]) == math.floor(CAP / K)


def test_table1_rounds_from_the_certificate_csv():
    rows = {r["param"]: r for r in _rows("certificate_table.csv")}
    table = _tabular("tab:certificate", "selection.tex")
    assert len(table) == 5, table
    for cells, key in zip(table, ["k=0.5", "k=3", "k=10", "n=8", "n=64"]):
        r = rows[key]
        assert key.split("=")[1] in cells[0].replace("$", ""), (cells[0], key)
        assert _rounds_to(_num(cells[2]), float(r["certificate_nats"])), (cells, r)
        assert _rounds_to(_num(cells[3]), float(r["K_over_Sw"])), (cells, r)
        # v11 (2026-09-24, sixth review round): the 400-nat response counts left the table for
        # Appendix A (a cap above S_w is itself vacuous for the window it is quoted against); the
        # table now prints the measured spend beside K, the queries before the composed certificate
        # is vacuous for a window, and the draws that certify a near-verbatim event at 1%.
        if r["mechanism"] == "metered":
            assert _rounds_to(_num(cells[4]), float(r["window_bound"])), (cells, r)
            assert _rounds_to(re.findall(r"\(\$?(\d+\.\d+)\$?\)", cells[2])[0],
                              float(r["measured_spend_nats"])), (cells, r)
            assert cells[6] == "---" and r["draws_to_certify_nv_1pct"] == "", (cells, r)
        else:
            assert _rounds_to(re.search(r"10\^\{(-[\d.]+)\}", cells[4]).group(1),
                              float(r["window_bound_log10"])), (cells, r)
            assert _num(cells[6]) == r["draws_to_certify_nv_1pct"], (cells, r)
        assert _num(cells[5]) == r["queries_before_vacuous"], (cells, r)


def test_the_composition_paragraph_quotes_table1():
    """Section 2's composition paragraph quotes the query horizon the table computes, and the draws
    that certify a near-verbatim event; the 400-nat response counts it quoted in v10 (192 against 2)
    are in Appendix A since v11 and are checked there (tests/test_selection_claims.py)."""
    rows = {r["param"]: r for r in _rows("certificate_table.csv")}
    t = body("selection.tex")
    para = t[t.index("\\label{sec:compose}"):][:2600]
    assert f"survives only ${rows['n=64']['queries_before_vacuous']}$ queries at $n=64$" in para
    assert "$0.75\\%$ at $n=64$" in para and "$3e^{K}$ draws" in para
    assert "$m\\log n$" in para
    for key in ("n=8", "k=3"):
        assert f"${rows[key]['responses_at_certificate']}$ responses" not in para


# ---------------------------------------------------------------- Figure 3, the head-to-head forest

def _forest():
    g = {}
    for grp, label, band in h2h_forest_rows():
        g.setdefault(grp, []).append((label, band))
    return g


def test_figure3_headline_is_the_deecho_csv_and_the_prose_quotes_it():
    (label, (d, lo, hi)), = _forest()["headline"]
    (r,) = [x for x in _rows("order_averaged_h2h_deecho.csv") if x["quantity"].startswith("D3")]
    assert (float(r["value"]), float(r["lo95"]), float(r["hi95"])) == (d, lo, hi)
    assert r["reading"] == "REVERSAL CONFIRMED" and lo > 0
    txt = body("experiments.tex", "iclr_intro.tex")
    assert f"$+{d:.4f}$ $[+{lo:.4f}, +{hi:.4f}]$" in txt


def test_figure3_prose_follows_the_plotted_rows():
    f = _forest()
    head = f["headline"][0][1][0]
    t = body("experiments.tex")
    # 'moves it by at most X' over the three fresh-draw rows
    move = max(abs(b[0] - head) for _, b in f["fresh draws, judge B"])
    assert f"moves it by at most ${move:.4f}$" in t
    # 'five of six judges exclude zero' counts the headline's judge B with C..G
    judges = [f["headline"][0][1]] + [b for _, b in f["same texts, other judges"]]
    words = {4: "four", 5: "five", 6: "six"}
    n_ex = sum(lo > 0 for _, lo, _ in judges)
    assert len(judges) == 6 and f"{words[n_ex]} of six judges exclude zero" in t
    (lab, (e, elo, ehi)), = [(l, b) for l, b in f["same texts, other judges"] if b[1] <= 0]
    assert "Mixtral" in lab and f"straddles it at ${e:+.4f}$".replace("+", "") in t.replace("+", "")
    # 'from Qwen2.5-1.5B-Instruct upward every interval covers zero', and not below it
    opp = dict(f["stronger fixed opponents, judge B"])
    assert opp["Qwen2.5-0.5B-Instruct"][1] > 0
    assert all(b[1] <= 0 <= b[2] for l, b in opp.items() if l != "Qwen2.5-0.5B-Instruct")
    assert re.search(r"[Ff]rom \\texttt\{Qwen2\.5-1\.5B-Instruct\} upward every interval covers zero", t)
    # workloads: holds on three, unresolved on two, reverses on AlpacaEval
    wl = dict(f["other workloads, binding budget, judge B"])
    holds = [l for l, b in wl.items() if b[1] > 0]
    assert len(holds) == 3 and all(("ours" in l or "books" in l) for l in holds), holds
    unres = [l for l, b in wl.items() if b[1] <= 0 <= b[2]]
    assert sorted(unres) == sorted(l for l in wl if "CoTaEval" in l or "MT-Bench" in l), unres
    (al, (a, alo, ahi)), = [(l, b) for l, b in wl.items() if b[2] < 0]
    assert "AlpacaEval" in al and f"the meter wins by ${-a:.4f}$" in t


# ---------------------------------------------------------------- Figure 2, the two horns

def test_figure2_text_follows_its_rows():
    pairs, (lo, hi), imit = meter_horns_rows()
    assert len(pairs) == 9
    band = f"between ${lo:.2f}$ and ${hi:.2f}$"
    cap = caption_of("fig:horns")
    # caution (an): the caption carries the same band in the same file, so the prose is checked with
    # the caption cut out -- otherwise the caption's copy satisfies the prose check (mutation-tested)
    t = body("frontier.tex").replace(cap, "")
    assert band in t
    assert band in cap
    k20 = [r for r in imit if r[0] == 20.0][0]
    assert f"imitation rate of ${k20[2]:.3f}$" in t
    assert f"spends ${k20[3]:.1f}$ nats" in t
    assert f"${100 * k20[3] / k20[4]:.1f}\\%$ of the ${k20[4]:.0f}$" in t
    # v11: the abstract states the vacuity theorem and the window exposure, not the measured onset
    # band, which lives in Section 3 (asserted above, prose and caption). It may not state a
    # DIFFERENT band: any "$a$ to $b$" pair of onset ratios in the abstract must be this one.
    abstract = " ".join(open(tex("iclr_2027.tex"), encoding="utf-8").read().split())
    abstract = abstract[abstract.index("begin{abstract}"):abstract.index("end{abstract}")]
    for a_, b_ in re.findall(r"\$(0\.\d\d)\$ to \$(1\.\d\d)\$", abstract):
        assert (a_, b_) == (f"{lo:.2f}", f"{hi:.2f}"), (a_, b_, lo, hi)


# ---------------------------------------------------------------- structure

def test_the_live_manuscript_inputs_exactly_the_v10_sections():
    live = {f for f in _live_files() if f != "iclr_2027.tex"}
    assert live == {f"sections/{n}.tex" for n in (
        "iclr_intro", "fig_overview", "selection", "frontier", "experiments", "related_work_v4",
        "iclr_closing", "appendix_proofs", "appendix_selection", "appendix_onset",
        "appendix_limitations", "appendix_related")}, live


def test_every_reference_resolves_to_exactly_one_live_label():
    """Tectonic prints '??' only for a missing label; a DUPLICATE label silently redirects every
    \\ref to the second copy (caution (z)). Check both over the live files only."""
    labels, refs = [], set()
    for f in _live_files():
        src = re.sub(r"(?<!\\)%.*", "", open(tex(f), encoding="utf-8").read())
        labels += re.findall(r"\\label\{([^}]*)\}", src)
        refs |= set(re.findall(r"\\(?:ref|eqref)\{([^}]*)\}", src))
    dup = sorted({x for x in labels if labels.count(x) > 1})
    assert not dup, dup
    assert not sorted(refs - set(labels)), sorted(refs - set(labels))
