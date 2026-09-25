"""v14 (2026-09-25): the review items first skipped and then run. Every number the paper quotes from them is
read back out of the CSV that produced it, and every claim ABOUT those numbers is checked against their shape."""
import csv
import os
import re

from manuscript import ROOT, body

R = os.path.join(ROOT, "results")


def rows(name):
    return list(csv.DictReader(open(os.path.join(R, name))))


def f(x, nd):
    return f"{float(x):.{nd}f}"


# ---- review 2 Q3: the meter as speculative decoding ------------------------------------------------------------

def spec():
    a = {r["k"]: r for r in rows("speculative_acceptance_8b.csv") if r["quantity"] == "acceptance"}
    b = rows("speculative_acceptance_70b.csv")[0]
    return a, b


def test_the_speculative_paragraph_quotes_its_csv():
    a, b = spec()
    txt = body("appendix_selection.tex")
    i = txt.index("\\label{app:speculative}")
    par = txt[i:txt.index("\\paragraph", i + 10) if "\\paragraph" in txt[i + 10:i + 4000] else i + 4000]
    for want in (f(a["10"]["alpha_mean"], 3), f(a["3"]["alpha_mean"], 3), f(a["0.5"]["alpha_mean"], 3),
                 f(a["10"]["tokens_per_risky_forward"], 2), f(a["10"]["selection_ratio"], 3),
                 f(a["0.5"]["selection_ratio"], 3), f(a["10"]["alpha_star"], 3),
                 f(a["10"]["selection_ratio_committed"], 3), f(b["selection_ratio_at_alpha1"], 3),
                 f(a["10"]["pstar_median_abs_err"], 4)):
        assert f"${want}" in par, want
    assert a["10"]["best_g"] == "1" and "($g=1$)" in par


def test_the_main_text_bound_is_the_csvs_and_does_not_erase_the_advantage():
    a, b = spec()
    txt = body("experiments.tex")
    s70, s8 = float(b["selection_ratio_at_alpha1"]), float(a["10"]["selection_ratio"])
    assert f"at most ${s70:.2f}\\times$ and ${s8:.2f}\\times$" in txt
    # "lift these to at most" is a claim that selection still wins: both bounds must stay below parity,
    # and the 70B bound must hold at PERFECT acceptance, which is why that pair needs no measured alpha
    assert s70 < 1 and s8 < 1
    assert b["alpha_star"] == ""                                   # no acceptance reaches parity at g <= 8
    assert float(a["10"]["alpha_mean"]) < float(a["10"]["alpha_star"])   # measured below the 8B break-even


# ---- review 2 Q6: where the gain lies ---------------------------------------------------------------------------

def strata():
    return {(r["task"], r["stratum"]): r for r in rows("utility_surprisal.csv")}


def test_the_gain_location_paragraph_quotes_its_csv():
    s = strata()
    txt = body("appendix_selection.tex")
    i = txt.index("\\label{app:gainwhere}")
    par = txt[i:i + 2600]
    for task, n_hi, n_lo in (("gsm8k", 439, 61), ("triviaqa", 287, 213)):
        hi = s[(task, "pi_ho >= 1/32, all (S <= log n)")]
        lo = s[(task, "pi_ho = 0 (S > log n)")]
        assert (int(hi["questions"]), int(lo["questions"])) == (n_hi, n_lo)
        band = f"${float(hi['gain']):+.3f}$ $[{float(hi['gain_lo95']):+.3f}, {float(hi['gain_hi95']):+.3f}]$"
        assert band in par, band
        assert f"${n_hi}$" in par and f"${n_lo}$" in par
    j = s[("judged headline (judge B)", "served S > log 64")]
    band = f"${float(j['gain']):+.3f}$ $[{float(j['gain_lo95']):+.3f}, {float(j['gain_hi95']):+.3f}]$"
    assert band in par and f"${int(j['questions'])}$" in par
    assert f"${float(s[('judged headline (judge B)', 'served S, nats')]['s_median']):.1f}$" in par


def test_gains_only_where_log_n_reaches_S_is_true_of_the_numbers():
    """The main text says a vote gains only where log n >= S. That is a claim about a set of numbers
    (caution (ai)): it holds iff the held-out S > log n stratum gains exactly nothing on both tasks."""
    s = strata()
    for task in ("gsm8k", "triviaqa"):
        assert float(s[(task, "pi_ho = 0 (S > log n)")]["gain"]) == 0.0
        assert float(s[(task, "pi_ho >= 1/32, all (S <= log n)")]["share_of_total_gain"]) == 1.0
    assert re.search(r"a vote gains only where \$\\log n \\ge S\$", body("iclr_closing.tex"))
    # and the judged side says the opposite, which the paragraph must not blur: all of it where S > log 64
    assert float(s[("judged headline (judge B)", "served S <= log 64")]["gain"]) == 0.0
