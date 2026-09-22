"""Guards for the points of referee reports 2, 3 and 4 that an audit on 2026-09-23 found open.

Most of those reports had been acted on by 2026-09-20. Re-reading every point against the live
manuscript found the rest: a sentence stating the window event's vacuity in the wrong direction, an
introduction claim (the within-prompt AUC) with no appendix, no CSV and a pointer to an appendix that
does not hold it, two caveats two reports asked the abstract to carry, and several questions the
paper answered nowhere. Each guard reads the data the sentence is about, not the sentence's wording
(caution (an)), so a rewording cannot retire it.
"""
import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tests.manuscript import body  # noqa: E402


def _csv(name):
    return list(csv.DictReader(open(os.path.join(ROOT, "results", name), encoding="utf-8")))


def test_the_window_event_is_vacuous_ABOVE_its_threshold_not_below():
    """frontier.tex and Appendix A both said the 50-token window event is 'vacuous below k = 0.8'.
    The table beside the second sentence says the opposite: K = k * T_max reaches the window's
    surprisal at k = 0.799, so the certificate is empty at every k from there UP. Checked as a
    property: wherever a k is named beside that window's vacuity, it is the CSV's threshold and the
    inequality points up."""
    k = float(next(r for r in _csv("window_vacuity.csv") if r["event"] == "50-token window")
              ["k_at_vacuity"])
    txt = body("frontier.tex", "appendix_proofs.tex").replace("$", "")
    assert not re.search(r"vacuous below k\s*=", txt), "the inverted direction is back"
    hits = re.findall(r"vacuous at (?:every )?k \\ge ([\d.]+)", txt)
    assert len(hits) >= 2, hits
    assert all(abs(float(h) - k) < 0.05 for h in hits), (hits, k)


def _auc():
    return {r["column"]: r for r in _csv("currency_auc.csv")}


def test_every_auc_the_paper_prints_rounds_from_the_csv():
    """The introduction quoted AUC 0.526 and 0.537 from a scoring log with no CSV behind them."""
    a = _auc()
    txt = body("iclr_intro.tex", "appendix_selection.tex").replace("$", "")
    for col in ("logp_per_token", "n_tokens"):
        assert f"{float(a[col]['auc']):.3f}" in txt, col
    for col in ("logp_per_token", "logp_total", "n_tokens"):
        r = a[col]
        band = f"{float(r['auc']):.3f} [{float(r['lo95']):.3f}, {float(r['hi95']):.3f}]"
        assert band in txt, band
    d = a["difference"]
    assert f"{float(d['auc']):+.3f} [{float(d['lo95']):+.3f}, {float(d['hi95']):+.3f}]" in txt


def test_the_likelihood_is_called_worse_than_length_only_if_the_interval_says_so():
    """The paper said the likelihood predicts judged quality 'worse than the answer's length'. The
    paired difference is +0.011 [-0.023, +0.046]: it includes zero, so the honest word is 'no
    better'. Conditioned on the interval, so the word may change only if the data do."""
    d = _auc()["difference"]
    resolved = float(d["lo95"]) > 0
    for name in ("iclr_intro.tex", "orders.tex", "experiments.tex"):
        t = body(name)
        for m in re.finditer(r"(worse|no better) than (?:the answer's )?length", t):
            assert m.group(1) == ("worse" if resolved else "no better"), (name, m.group(0))
    assert re.search(r"no better than the answer's length", body("iclr_intro.tex"))


def test_the_failed_scorer_and_the_auc_point_to_a_paragraph_that_holds_them():
    """Caution (aj): both main-text claims pointed at appendices that did not contain them -- the
    AUC at Appendix A, the failed scorer's -0.006 and the oracle's +0.488 at Appendix H's top."""
    app = body("appendix_selection.tex")
    i = app.index(r"\label{app:currency}")
    para = app[i: app.index(r"\paragraph", i)]
    rows = {(r["rule"], r["n"]): float(r["u"]) for r in _csv("selection_decoding.csv")}
    base = rows[("per-token mean (primary)", "1")]
    fail = rows[("per-token mean (primary)", "8")] - base
    oracle = rows[("oracle: the judge itself", "8")] - base
    for v in (f"{fail:+.3f}".replace("+", "-" if fail < 0 else "+"), f"{oracle:+.3f}",
              f"{float(_auc()['logp_per_token']['auc']):.3f}", f"{float(_auc()['n_tokens']['auc']):.3f}"):
        assert v in para.replace("$", ""), v
    assert r"(Appendix~\ref{app:currency})" in body("orders.tex")
    exp = body("experiments.tex")
    j = exp.index("+0.488")
    assert r"\ref{app:currency}" in exp[j: j + 120], "the oracle's pointer no longer reaches it"


def test_the_gain_ratio_is_quoted_with_its_interval_and_only_while_it_excludes_one():
    """The introduction said selection 'wins by 2.6x' with no interval; its denominator's own
    interval comes within 0.014 of zero, so the ratio's is wide. Both places that print it must
    round from the CSV, and 'win by' may stand only while the interval excludes one."""
    r = _csv("h2h_ratio_interval.csv")[0]
    ratio, lo, hi = float(r["ratio"]), float(r["ratio_lo95"]), float(r["ratio_hi95"])
    intro = body("iclr_intro.tex")
    assert f"${ratio:.1f}\\times$ $[{lo:.1f}, {hi:.1f}]$" in intro, "intro ratio or interval stale"
    if r["excludes_one"] != "yes":
        assert "win by" not in intro, "the ratio's interval includes one; 'win by' overstates it"
    app = body("appendix_selection.tex")
    assert f"${ratio:.2f}\\times$ $[{lo:.2f}, {hi:.2f}]$" in app
    assert f"$[{float(r['fieller_lo95']):.2f}, {float(r['fieller_hi95']):.2f}]$" in app
    assert f"${100 * float(r['undefined_frac']):.2f}\\%$" in app


def test_the_cpfuse_audit_appendix_j_claims_is_printed_and_rounds_from_both_builds():
    """Appendix J said CP-Fuse 'is the second mechanism we audit' and printed no number from the
    audit. Worse, the 2026-09-20 rebuild was written over the phase-3 CSV under its own name while
    its scoring log said that file was untouched (caution (ax)). Both builds are now reported, each
    from its own file, and the two files must stay distinct."""
    def cells(name):
        return {(r["arm"], r["mode"], r["shard"]): float(r["nv_recall_mean"]) for r in _csv(name)}
    new, old = cells("cpfuse_audit_rebuild.csv"), cells("cpfuse_audit.csv")
    assert new != old, "the phase-3 CP-Fuse audit has been overwritten by the rebuild again"
    app = body("appendix_related.tex")
    i = app.index(r"\label{app:cpfuse}")
    para = app[i: app.index(r"\paragraph", i)].replace("$", "")
    want = [new[("a", "single", "0")], new[("b", "single", "1")], new[("a", "single", "1")],
            new[("cpfuse", "single", "0")], new[("cpfuse", "oracle", "0")],
            new[("cpfuse", "oracle", "1")], new[("mixture", "single", "0")],
            old[("a", "single", "0")], old[("b", "single", "1")], old[("cpfuse", "single", "0")]]
    for v in want:
        assert f"{v:.4f}" in para, v
    assert r"\ref{app:cpfuse}" in app[: i], "the one-line 'we audit' claim no longer points at it"


def test_the_prompt_screen_can_fire_and_the_appendix_reports_what_it_found():
    """A referee asked whether the 500 judged prompts were screened for overlap with the protected
    works. The screen is data-driven (8-grams of the passages, titles as recorded in the corpus);
    a screen that cannot fire is not a screen (caution (p)), so it is checked on a positive first."""
    import glob
    import json
    from analysis.blocklist import build_index, ngrams, toks
    files = sorted(glob.glob(os.path.join(ROOT, "data", "copybench_*.jsonl")))
    idx = build_index(files, 8)
    first = json.loads(open(files[0], encoding="utf-8").readline())
    assert ngrams(toks(first["raw_text"]), 8) & idx, "the 8-gram screen cannot fire on a passage"
    rows = _csv("prompt_set_profile.csv")
    hits = sum(int(r["shares_8gram_with_protected"]) + int(r["names_a_protected_title"]) for r in rows)
    app = body("appendix_selection.tex")
    claims_none = "none shares a word $8$-gram with any of its passages" in app
    assert claims_none == (hits == 0), (hits, claims_none)


def test_the_vetting_check_names_the_prefix_a_deployer_should_use():
    """Two reports asked what prefix length to vet at and why that is not an attack surface. The
    answer follows from Proposition 1 holding prompt by prompt: the user picks the prompt, so vet at
    the longest genuine prefix the deployment accepts. Both places a deployer reads must say it."""
    import tests.manuscript as ms
    ethics = " ".join(open(ms.tex("iclr_2027.tex"), encoding="utf-8").read().split())
    assert "longest genuine prefix the deployment accepts" in ethics
    vet = body("appendix_selection.tex")
    i = vet.index(r"\label{app:vetting}")
    assert "attack surface" in vet[i: vet.index(r"\paragraph", i)]


def test_the_draw_count_symbol_is_not_reused_for_a_prompt_count():
    """A report flagged 'n' as both the number of draws and, in Appendix A, a count of prompts
    ('on n = 73'). n is the draw count everywhere; a count of prompts is written in words."""
    assert not re.search(r"\$n\s*=\s*73\$", body("appendix_proofs.tex"))


def _abstract():
    import tests.manuscript as ms
    t = " ".join(open(ms.tex("iclr_2027.tex"), encoding="utf-8").read().split())
    return t.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0]


def test_the_abstract_carries_the_two_caveats_its_data_still_require():
    """Two reports asked the abstract to carry the regime selection cannot serve and the loss at
    matched compute; both were in the body and neither was in the abstract. Each is required
    exactly while its data say it (caution (aq): condition a withdrawal on data, not on a phrase)."""
    b3 = next(r for r in _csv("anchor_only_cost_bands.csv") if r["band"].startswith("B3"))
    lo, hi = map(float, re.search(r"\[([-+\d.]+), ([-+\d.]+)\]", b3["reading"]).groups())
    a = _abstract()
    if hi < 0:
        assert "loses at matched compute" in a, "the matched-compute loss left the abstract"
    rows = {(r["mechanism"], r["arm"]): r for r in _csv("verifiable_metered_tqa.csv")}
    best_sel = max((r for (m, _), r in rows.items() if m.startswith("selection")),
                   key=lambda r: float(r["acc"]))
    risky = rows[("metered decoder", "k=-1")]
    if float(best_sel["acc_hi95"]) < float(risky["acc_lo95"]):
        assert re.search(r"cannot do the task, no \$n\$ rescues it", a), \
            "the capability-gap concession left the abstract while TriviaQA still shows the gap"


def test_the_below_threshold_paragraph_rounds_from_the_window_csv():
    """Appendix A's new paragraph prices the KL order's linear looseness on the 50-token window at
    k = 0.5 (K = 100 at T_max = 200). Every number is recomputed here."""
    import math
    S = float(next(r for r in _csv("window_vacuity.csv") if r["event"] == "50-token window")
              ["S_median_nats"])
    K = 0.5 * 200
    lo, hi = K / S, (K + math.log(2)) / S
    assert f"{lo:.2f}" == f"{hi:.2f}", "the two ends of the KL bound no longer round alike"
    app = body("appendix_proofs.tex")
    i = app.index(r"\label{app:belowthreshold}")
    para = app[i: app.index(r"\paragraph", i)]
    assert f"about ${lo:.2f}$" in para
    assert f"$e^{{{round(K - S)}}}$" in para
    assert f"${0.01 * S:.1f}$ at" in para
    assert r"\ref{app:belowthreshold}" in body("selection.tex"), "section 3.2 lost its pointer"


def test_the_main_text_says_where_the_kl_bound_is_attained():
    """Two reports asked for the realised divergence of selection; Appendix H answers that the
    closed form is attained for a tie-free score and measures the tie rate. The main text now says
    so beside the head-to-head, and must keep pointing at the paragraph that shows it."""
    exp = body("experiments.tex")
    i = exp.index("attained for a tie-free score")
    assert r"\ref{app:realisedkl}" in exp[i: i + 80]
    app = body("appendix_selection.tex")
    j = app.index(r"\label{app:realisedkl}")
    assert r"\mathrm{Beta}(n,1)" in app[j: app.index(r"\paragraph", j)]


def test_the_open_markers_are_explained_by_the_gate_that_failed_them():
    """Two reports asked why the open markers of Figure 2 missed admission. The caption now says,
    and says it from the CSV: every anchor whose entry gate failed, at its own empty fraction."""
    fails = {r["anchor"]: float(r["empty_frac_n1"]) for r in _csv("selection_breadth.csv")
             if r["entry_gate"] == "FAIL"}
    assert len(fails) == 2, fails
    exp = body("experiments.tex")
    i = exp.index(r"\caption{\textbf{Every arm selection anchoring")
    cap = exp[i: exp.index(r"\label{fig:breadth}", i)]
    for frac in fails.values():
        assert f"${100 * frac:.1f}\\%$" in cap, frac
    assert r"$5\%$" in cap


def test_the_headline_pass_frontier_distances_round_from_their_csv():
    """A report asked whether log 64 is near Theorem 1's frontier; Appendix A answered only at n=8.
    Both headline-pass ratios are recomputed by analysis/frontier_distance_h2h.py."""
    r = {x["arm"]: float(x["over_frontier"]) for x in _csv("frontier_distance_h2h.csv")}
    sel, met = r["selection, n=64"], r["metered decoder, k=10"]
    app = body("appendix_proofs.tex")
    assert f"selection at $n=64$ is ${sel:.0f}\\times$ the frontier" in app
    assert f"${met:,.0f}\\times$".replace(",", "{,}") in app
    assert sel < met / 100, "selection is no longer two orders of magnitude nearer the frontier"


def test_the_related_work_table_claims_a_measurement_only_where_one_is_reported():
    """A report asked for the defences side by side. The table's last column says what this paper
    measured; a row that says 'measured' must point at a paragraph that prints that measurement."""
    app = body("appendix_related.tex")
    i = app.index(r"\label{tab:related}")
    tab = app[i: app.index(r"\end{tabular}", i)]
    for name in ("CP-Fuse", "MemFree", "TokenSwap", "TRBS", "BloomScrub", "SILO", "CPR"):
        assert name in tab, name
    # every 'measured' row says "(below)": the paragraphs it means follow the table in this appendix
    assert tab.count("(below)") == 3, "a row claims a measurement without saying where it is"
    for label, number in (("app:cpfuse", "0.6905"), ("app:blocklist", "0.4192")):
        j = app.index(rf"\label{{{label}}}")
        assert j > i and number in app[j: app.index(r"\paragraph", j)], (label, number)
    assert "$-0.1040$" in app[i:], "TokenSwap's row lost its numbers"
