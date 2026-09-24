"""The nine-pair table is copied out of results/onset_table.csv by hand. It moved to
Appendix~\\ref{app:onset} in v6; the range and the pair count it supports stay in the main text. Check all of them mechanically: a hand edit that drifts from the
CSV, or a CSV that moves without the table following, fails here rather than in review."""
import csv
import os
import re

import pytest

CSV = "results/onset_table.csv"
from tests.manuscript import tex

TEX = tex("sections/appendix_onset.tex")
# v10 (2026-09-24): onset.tex was retired (onset_v9_2026-09-24.tex, not compiled); the main-text
# onset prose and Figure horns (the nine pairs, the 0.88-1.17 range) live in frontier.tex, Section 3.
PROSE = tex("sections/frontier.tex")
# the section files the manuscript \inputs, for the scans that must cover every live page
LIVE = ("iclr_intro", "selection", "frontier", "experiments", "related_work_v4", "iclr_closing",
        "appendix_proofs", "appendix_selection", "appendix_onset", "appendix_limitations",
        "appendix_related")
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
    body = " ".join(open(PROSE, encoding="utf-8").read().split())   # caution (ar): not the wrap
    src = [r for r in csv.DictReader(open(CSV)) if not r["pair"].startswith("ALL")]
    ratios = [float(r["ratio"]) for r in src]
    word = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'][len(src)]
    assert re.search(rf"\b{word} (?:model )?pairs\b", body), f"the prose does not say {word} pairs"
    # DERIVED from the CSV, never written out. Until 2026-09-18 this was a disjunction whose last
    # branch was the literal `f"$0.88$ and $1.17$"` -- a constant, so it passed whatever the ratios
    # said, and a pair drifting to 1.25 would have left the prose unguarded. Caution (an) (a guard
    # written as a list of phrasings) on top of caution (ag) (a number nothing computed is a
    # comment, not data). One branch now, and it is the data's.
    lo_s, hi_s = f"{min(ratios):.2f}", f"{max(ratios):.2f}"
    assert f"between ${lo_s}$ and ${hi_s}$" in body, \
        (f"the prose must print the range the CSV holds: ${lo_s}$ and ${hi_s}$",
         min(ratios), max(ratios))


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
    # v10 (2026-09-24): appendix_robustness was retired into appendix_onset, which states the factor
    # beside the table ("the memorisers span a factor of $5.1$ in their own recall") and prints the
    # two endpoints in the table's mem. column rather than in the sentence
    ons = " ".join(open(TEX, encoding="utf-8").read().split())
    lim = " ".join(open(tex("sections/appendix_limitations.tex"), encoding="utf-8").read().split())
    assert f"span a factor of ${factor}$" in ons, \
        f"appendix_onset must say the memorisers span a factor of {factor}"
    cells = {c[4].strip("$ \\") for c in _rows()}          # the mem. column, one cell per pair
    assert f"{lo:.3f}" in cells and f"{hi:.3f}" in cells, \
        f"the table's mem. column must hold both ends of the range, {lo:.3f} and {hi:.3f}"
    assert f"a factor of ${factor}$ in sampled $k=-1$" in lim, \
        f"appendix_limitations must say a factor of {factor}"


def test_no_quoted_strength_comes_from_a_different_corpus():
    """The specific wrong number, pinned so it cannot come back. Every live section since v10
    (2026-09-24), which retired appendix_robustness into appendix_onset."""
    for name in (f"sections/{s}.tex" for s in LIVE):
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
    # case-free: v10 (2026-09-24) says it mid-sentence beside the table, "and five of the nine did
    # not converge"
    assert re.search(rf"\b{words[no]} of the nine did\b", body, flags=re.I), \
        f"{no} of nine did not converge; the text beside the table must say {words[no]}"


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


# ---------------------------------------------------------------------------
# The two sweeps that never ran their mandated baselines, and the runs that fixed it.
#
# Working Rules: every experiment reporting a copying metric at some k also reports k=-1 and k=0 on
# the same prompts and seeds. output/phase4/fine_tc and fine_comma report six budgets each and ran
# NEITHER. scripts/run_phase4_baselines.sh measured both on 2026-09-16 at the flags that reproduce
# each sweep's protocol line, and reproduced to three decimals the values the table had been reading
# off unrelated arms -- 0.492 and 0.719. These pin that, because the value is only usable while the
# protocol still matches: a re-run at a different --batch-size would move a SAMPLED k=-1 (caution
# (u)) and look identical at the CSV.
BASELINES = {"output/phase4/fine_tc": ("output/phase4/fine_tc_base", 0.492),
             "output/phase4/fine_comma": ("output/phase4/fine_comma_base", 0.719)}


def _single(d, k):
    with open(os.path.join(ROOT_DIR, d, "composition_summary.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["mode"] == "single" and float(r["k"]) == k:
                return float(r["nv_recall_mean"])
    return None


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("sweep,pair", list(BASELINES.items()))
def test_the_missing_baseline_was_measured_not_borrowed(sweep, pair):
    base, expected = pair
    assert os.path.isdir(os.path.join(ROOT_DIR, base)), f"{base} missing; run scripts/run_phase4_baselines.sh"
    k1, k0 = _single(base, -1.0), _single(base, 0.0)
    assert k1 is not None and k0 is not None, f"{base} lacks a k=-1 or k=0 single arm"
    assert abs(k1 - expected) < 5e-4, (
        f"{base} k=-1 is {k1}, not the {expected} it reproduced when measured; the protocol has "
        "moved (--batch-size is part of the seed for a sampled arm, caution (u))")
    assert k0 == 0.0, f"{base} k=0 is {k0}; the anchor alone must reproduce nothing"


@pytest.mark.parametrize("sweep,pair", list(BASELINES.items()))
def test_the_baseline_run_shares_its_sweep_s_protocol_line(sweep, pair):
    """The whole justification for a separate run is that it is the same experiment."""
    import analysis.onset_table as ot
    base = pair[0]
    want, got = ot._protocol(os.path.join(ROOT_DIR, sweep)), ot._protocol(os.path.join(ROOT_DIR, base))
    assert want and got, f"no protocol line for {sweep} or {base}"
    assert want == got, f"protocol drift:\n  sweep {want}\n  base  {got}"


@pytest.mark.parametrize("sweep,pair", list(BASELINES.items()))
def test_the_baseline_run_covers_the_modes_its_sweep_swept(sweep, pair):
    def modes(d):
        with open(os.path.join(ROOT_DIR, d, "composition_summary.csv")) as fh:
            return {r["mode"] for r in csv.DictReader(fh)}
    assert modes(sweep) <= modes(pair[0]), (
        f"{pair[0]} does not carry a baseline for every mode {sweep} swept")


def test_no_table_row_reads_its_baseline_from_an_unrelated_arm():
    """Before 2026-09-16 these two came from leakage_headtohead and comp_comma7b, which are
    different experiments that happened to share a protocol line. They now come from runs made
    for the purpose, and the CSV must say so."""
    with open(CSV) as fh:
        for r in csv.DictReader(fh):
            if r["pair"].startswith("ALL") or not r.get("strength_source"):
                continue
            src = r["strength_source"]
            assert src == "own sweep" or src.startswith("baseline run "), (
                r["pair"], f"strength read from {src!r}, which is not a dedicated baseline run")
