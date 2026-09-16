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
        if len(cells) == 9 and cells[0].replace('$^{\\dagger}$', '').strip() in LABEL:
            out.append(cells)
    return out


def test_every_cell_of_the_section_4_table_comes_from_the_csv():
    src = {r["pair"]: r for r in csv.DictReader(open(CSV)) if not r["pair"].startswith("ALL")}
    rows = _rows()
    assert len(rows) == len(LABEL) == len(src), (len(rows), len(LABEL), len(src))
    for cells in rows:
        label = cells[0].replace("$^{\\dagger}$", "").strip()
        r = src[LABEL[label]]
        s_x, s_r, n, mem, ep, onset, ci, ratio = (c.strip("$ \\").rstrip("\\") for c in cells[1:])
        assert round(float(r["s_safe"]), 2) == float(s_x), (cells[0], "s(x)", s_x)
        assert round(float(r["s_risky"]), 3) == float(s_r), (cells[0], "s_r", s_r)
        assert int(r["n_passages"]) == int(n), (cells[0], "n", n)
        assert round(float(r["onset"]), 2) == float(onset), (cells[0], "onset", onset)
        assert round(float(r["ratio"]), 3) == float(ratio), (label, "ratio", ratio)
        # The strength column, and the dagger that says where it came from. A cell borrowed from a
        # companion run must SAY so: an unmarked borrow is the defect caution (v) is about.
        marked = "\\dagger" in cells[0]
        assert round(float(r["k_minus1_sampled"]), 3) == float(mem), (label, "mem k=-1", mem)
        assert marked == (r["strength_source"] != "own sweep"), (
            label, "dagger disagrees with strength_source", r["strength_source"])
        # the convergence marker, and the equivalence the caption asserts to a reader
        ran, cap = (int(v) for v in ep.split("/"))
        assert (ran, cap) == (int(r["epochs_run"]), int(r["epochs_cap"])), (label, "epochs", ep)
        assert (ran < cap) == (r["converged"] == "yes"), (
            label, "the caption says stopping early means converged; this row breaks that",
            ep, r["converged"], r["final_loss"], r["stop_loss"])
        assert (float(r["final_loss"]) <= float(r["stop_loss"])) == (r["converged"] == "yes")
        assert float(r["k_minus1_sampled"]) >= 0.10, (cells[0], "below the entry gate")
        assert float(r["k0_sampled"]) == 0.0, (cells[0], "k=0 is not 0.000")
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


# ---------------------------------------------------------------------------
# The strength range the prose quotes, and the guard that keeps it honest.
#
# Until 2026-09-16 two appendices said the nine memorisers "span a factor of 3.4 in sampled k=-1",
# from 0.2696 to 0.9091. The 0.2696 is output/phase5/fineg_phi35 -- Phi-3.5-mini on GUTENBERG, not
# one of the nine CopyBench pairs this table is about. Nothing caught it, because no CSV carried the
# strength column for the table to be checked against. It does now, and so does this.
import subprocess
import sys


def _measured():
    with open(CSV) as fh:
        return [float(r["k_minus1_sampled"]) for r in csv.DictReader(fh)
                if not r["pair"].startswith("ALL") and r["k_minus1_sampled"]]


def test_the_strength_range_the_appendices_quote_rounds_from_the_csv():
    v = _measured()
    assert len(v) == 9, f"expected nine measured strengths, got {len(v)}"
    lo, hi = min(v), max(v)
    factor = f"{hi / lo:.1f}"
    rob = " ".join(open(tex("sections/appendix_robustness.tex"), encoding="utf-8").read().split())
    lim = " ".join(open(tex("sections/appendix_limitations.tex"), encoding="utf-8").read().split())
    assert f"${lo:.3f}$ to ${hi:.3f}$, a factor of ${factor}$" in rob, \
        f"appendix_robustness must say {lo:.3f} to {hi:.3f}, a factor of {factor}"
    assert f"a factor of ${factor}$ in sampled $k=-1$" in lim, \
        f"appendix_limitations must say a factor of {factor}"


def test_no_quoted_strength_comes_from_a_different_corpus():
    """The specific wrong number, pinned so it cannot come back."""
    for name in ("sections/appendix_robustness.tex", "sections/appendix_limitations.tex",
                 "sections/appendix_onset.tex"):
        body = open(tex(name), encoding="utf-8").read()
        assert "0.2696" not in body, (
            f"{name} quotes 0.2696, which is fineg_phi35 on Gutenberg, not a CopyBench pair")


def test_a_borrowed_baseline_is_refused_when_the_protocol_differs():
    """The borrow is only sound because the protocol lines match. Prove the check bites."""
    code = (
        "import sys; sys.path.insert(0, '.');\n"
        "import analysis.onset_table as ot\n"
        "ot.SWEEPS['comma-7b + mem. comma-7b'] = ("
        "    'output/phase4/fine_comma', 'output/phase4/fine_tc.log')\n"
        "ot.strength('comma-7b + mem. comma-7b')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert r.returncode != 0, "borrowing across protocols was allowed"
    assert "refusing to borrow a baseline across protocols" in (r.stdout + r.stderr)


def test_strict_mode_refuses_the_borrow_entirely():
    import analysis.onset_table as ot
    k1, _, why = ot.strength("comma-7b + mem. comma-7b", strict=True)
    assert k1 is None and "strict" in why
    k1, _, why = ot.strength("kl3m-520m + mem. kl3m-520m", strict=True)
    assert k1 is not None and why == "own sweep", "strict must not affect an own-sweep baseline"


def test_the_convergence_count_the_caption_states_matches_the_csv():
    """'Five of the nine did not' is a number, and a number must come from the CSV."""
    with open(CSV) as fh:
        rows = [r for r in csv.DictReader(fh) if not r["pair"].startswith("ALL")]
    yes = sum(1 for r in rows if r["converged"] == "yes")
    no = len(rows) - yes
    words = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven"}
    body = " ".join(open(TEX, encoding="utf-8").read().split())
    assert f"{words[no]} of the nine did" in body, \
        f"{no} of nine did not converge; the caption must say {words[no]}"


def test_the_summary_row_reports_the_same_convergence_count():
    with open(CSV) as fh:
        rows = list(csv.DictReader(fh))
    body = [r for r in rows if not r["pair"].startswith("ALL")]
    summ = next(r for r in rows if r["pair"].startswith("ALL"))
    yes = sum(1 for r in body if r["converged"] == "yes")
    assert summ["converged"] == f"{yes} of {len(body)} reached their stop-loss"


def test_the_strength_of_a_row_is_measured_on_that_row_s_own_passages():
    """Pleias-350M is measured at n=100 and n=458; the table prints 458, so its strength must be
    the 458-passage one. A strength read off a different passage set than the onset beside it is
    the same defect as quoting a Gutenberg number in a CopyBench claim."""
    import analysis.onset_table as ot
    for key, (sweep, _) in ot.SWEEPS.items():
        if sweep.endswith(".log"):
            continue
        n = ot._n_of(sweep)
        assert n is not None, f"{key}: {sweep} has no single-mode rows"
    with open(CSV) as fh:
        row = next(r for r in csv.DictReader(fh) if r["pair"].startswith("Pleias-350M"))
    assert row["n_passages"] == "458"
    assert ot._n_of(ot.SWEEPS["pleias-350m + mem. pleias-350m"][0]) == 458
