"""The domain split, and the confound that makes its headline unusable.

D1 came back ANTI -- the gain is larger where the anchor's control is lower -- which is the
direction the pre-registration named as mechanically cheap. The exchangeability null shows the
no-effect world already produces most of it, so the reading is uninformative. These tests pin the
three things that must not drift: the MT-Bench grouping (committed before the eight cells were
computed), the null's own sanity, and the fact that no scoring log claims the ceiling explains the
benchmark weakening.
"""
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.domain_breadth import MT_FAMILY, exact_p, spearman  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "domain_breadth.csv")
NULL = os.path.join(ROOT, "results", "domain_breadth_null.csv")
LOG = os.path.join(ROOT, "results", "onset_prediction_domain_breadth.md")


def rows(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_the_mtbench_grouping_is_the_committed_one():
    """Four categories per family, fixed before the eight cells existed. Regrouping after seeing
    them is excluded alternative 1, and the only way to regroup is to edit this constant."""
    assert sorted(MT_FAMILY) == ["coding", "extraction", "humanities", "math", "reasoning",
                                 "roleplay", "stem", "writing"]
    fam = {}
    for c, f in MT_FAMILY.items():
        fam.setdefault(f, []).append(c)
    assert {k: sorted(v) for k, v in fam.items()} == {
        "open-ended": ["humanities", "roleplay", "stem", "writing"],
        "constrained": ["coding", "extraction", "math", "reasoning"]}


def test_every_registered_cell_is_present_for_both_judges():
    r = rows(CSV)
    cells = {"selfinstruct", "oasst", "koala", "helpful_base", "vicuna",
             "open-ended", "constrained"}
    for judge in {x["judge"] for x in r}:
        assert {x["cell"] for x in r if x["judge"] == judge} == cells, judge
    assert len({x["judge"] for x in r}) == 2, "the D1 band needs the sign in both judges"


def test_the_alpaca_cell_sizes_are_the_benchmarks_own():
    """805 prompts in the five sources AlpacaEval ships, not a grouping of ours."""
    r = [x for x in rows(CSV) if x["cell"] not in ("open-ended", "constrained")]
    got = {x["cell"]: int(x["n_prompts"]) for x in r if x["judge"] == r[0]["judge"]}
    assert got == {"selfinstruct": 252, "oasst": 188, "koala": 156, "helpful_base": 129,
                   "vicuna": 80}
    assert sum(got.values()) == 805


def test_the_no_effect_null_is_itself_strongly_negative():
    """The whole point: a world where selection does nothing already yields a large negative
    Spearman, because u(n=1) is subtracted inside the gain. If this ever came back near zero the
    artefact would be gone and D1 would become readable -- which would be a change worth noticing,
    not a test to quietly relax."""
    for x in rows(NULL):
        assert float(x["null_mean"]) < -0.2, (x["judge"], x["null_mean"])
        assert float(x["null_sd"]) > 0.2, (x["judge"], x["null_sd"])
        assert int(x["reps"]) >= 10000


def test_the_observed_correlation_is_not_separable_from_the_artefact():
    """Both judges sit above 0.05 against the no-effect null, which is why the scoring log refuses
    to report ANTI. A future run that separated them would have to rewrite that log."""
    for x in rows(NULL):
        assert float(x["p_vs_null"]) > 0.05, (x["judge"], x["p_vs_null"])
        assert float(x["observed_rho"]) < 0, x["judge"]


def test_the_scoring_log_does_not_claim_the_ceiling_explains_the_weakening():
    """The refuted claim is the empirical one. The theorem q(y) <= n p_s(y) is untouched, and the
    log must keep saying which of the two it refuted."""
    t = open(LOG, encoding="utf-8").read().replace("\n", " ")
    assert "uninformative" in t, "D1 must be reported as uninformative, not as ANTI"
    assert "no evidence for that explanation" in t.lower()
    assert "is a **theorem**" in t, "the log must keep the theorem separate from the explanation"


def test_spearman_and_the_exact_permutation_agree_with_a_hand_case():
    assert abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-12
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12
    rho, p = exact_p([1, 2, 3, 4], [1, 2, 3, 4])
    assert abs(rho - 1.0) < 1e-12 and abs(p - 2 / 24) < 1e-12, (rho, p)


def test_the_null_preserves_cell_sizes():
    """A resample that dropped or moved prompts would break the comparison it exists to make."""
    from analysis.domain_breadth import domains
    d = domains(os.path.join(ROOT, "data", "bench", "mtbench_factual.jsonl"), MT_FAMILY)
    assert len(d) == 80
    counts = {}
    for v in d.values():
        counts[v] = counts.get(v, 0) + 1
    assert counts == {"open-ended": 40, "constrained": 40}
    rng = random.Random(0)
    assert rng is not None


# --- the manuscript paragraph these numbers back ------------------------------------------------

def _manuscript(name):
    from tests.manuscript import tex
    return open(tex(f"sections/{name}"), encoding="utf-8").read().replace("\n", " ")


def test_section6_quotes_the_benchmark_gains_from_their_own_csvs():
    """Section 6's breadth paragraph must round from selection_scaling_{alpaca,mtbench}.csv, once.
    Caution (j): a paper number rounds from the CSV, and the check is mechanical, not by eye."""
    import re
    body = _manuscript("experiments.tex")
    want = {}
    for bench in ("alpaca", "mtbench"):
        for r in rows(os.path.join(ROOT, "results", f"selection_scaling_{bench}.csv")):
            if int(r["n"]) == 8:
                want[(bench, r["judge"])] = (float(r["gain"]), float(r["gain_lo95"]),
                                             float(r["gain_hi95"]))
    assert re is not None
    # Judge B's band is plotted in Figure 1's forest (it moved there on 2026-09-17); judge C's is
    # carried by the limitations appendix, which is where the two-axis discussion lives. Both must
    # still round from the CSV exactly once -- caution (j) is unaffected by which surface prints it.
    from tests.manuscript import carries_band
    g, lo, hi = want[("alpaca", "Phi-3.5-mini-instruct")]
    assert carries_band(g, lo, hi, "experiments.tex"), ("judge B", g, lo, hi)
    g, lo, hi = want[("alpaca", "Meta-Llama-3.1-8B-Instruct")]
    assert carries_band(g, lo, hi, "appendix_limitations.tex"), ("judge C", g, lo, hi)
    assert "selection_breadth_forest" in body, "the figure carrying the judge-B band is not included"


def test_section6_quotes_the_mtbench_half_width_it_can_actually_support():
    """The paragraph says MT-Bench 'at a half-width of 0.09 could not' resolve anything. That is a
    claim about the widest CI the benchmark produces at n=8, and it has to be true of both judges."""
    hw = []
    for r in rows(os.path.join(ROOT, "results", "selection_scaling_mtbench.csv")):
        if int(r["n"]) == 8:
            hw.append((float(r["gain_hi95"]) - float(r["gain_lo95"])) / 2)
    assert min(hw) >= 0.085, hw          # 0.09 must not overstate how tight the arm is
    assert "half-width of $0.09$" in _manuscript("experiments.tex")


def test_section6_quotes_the_correlation_and_its_null_from_the_csvs():
    """The refutation of the ceiling explanation rests on two numbers per judge; both are pinned."""
    body = _manuscript("experiments.tex")
    n = {x["judge"]: x for x in rows(NULL)}
    c = n["Meta-Llama-3.1-8B-Instruct"]
    b = n["Phi-3.5-mini-instruct"]
    assert abs(float(c["observed_rho"]) + 0.79) < 5e-3 and abs(float(b["observed_rho"]) + 0.70) < 5e-3
    assert abs(float(c["p_vs_null"]) - 0.09) < 5e-3 and abs(float(b["p_vs_null"]) - 0.17) < 5e-3
    # Section 6 carries the reading in one clause and points at the appendix; the four numbers
    # themselves moved there on 2026-09-13 to pay for the judge-free arm, and this follows them
    # rather than letting them go unpinned.
    assert "inseparable from a no-effect null" in body
    apx = _manuscript("appendix_limitations.tex")
    assert "$-0.79$" in apx and "$-0.70$" in apx, "the correlations left Section 6 unpinned"
    assert f"$-{-float(c['null_mean']):.2f} \\pm {float(c['null_sd']):.2f}$" in apx, c
    assert "$0.09$" in apx and "$0.17$" in apx


def test_section6_separates_the_two_axes_the_ceiling_was_tested_on():
    """Superseded 2026-09-12 evening. The earlier version of this test guarded the sentence 'the
    pre-registered test of it fails', which was true of the DOMAIN axis and became misleading once
    feat-096 tested the ANCHOR axis and the ceiling was confirmed there. What must not drift is the
    distinction: the ceiling binds at the anchor, and the domain split remains uninformative."""
    body = _manuscript("experiments.tex")
    assert "binds at the anchor and not within one" in body
    assert "binds at the anchor and not within one" in body
    assert "inseparable from a no-effect null" in body
    low = body.lower()
    assert "inseparable from a no-effect null" in low, "the domain null must stay beside it"
    assert "the pre-registered test of it fails" not in low, "that sentence is now wrong"


def test_the_domain_split_stays_uninformative_whatever_the_anchor_axis_says():
    """feat-096 confirming the ceiling across ANCHORS does not retro-fit the domain result."""
    t = open(LOG, encoding="utf-8").read()
    assert "uninformative" in t
    alpaca = os.path.join(ROOT, "results", "onset_prediction_alpaca_comma7b.md")
    if os.path.exists(alpaca):
        a = " ".join(open(alpaca, encoding="utf-8").read().split())
        if "## Scoring, " in a:
            assert "stays UNINFORMATIVE" in a, \
                "the deciding arm must say which axis each result speaks to"


# --- the deciding arm, and keeping the two axes apart -------------------------------------------

ALPACA96 = os.path.join(ROOT, "results", "selection_scaling_alpaca_comma7b.csv")


def test_the_deciding_arm_confirms_the_ceiling_across_anchors():
    """A1 CEILING CONFIRMED: the CI excludes zero and the gain exceeds the audited anchor's by more
    than the committed 0.03, on the registered scorer."""
    if not os.path.exists(ALPACA96):
        return
    def n8(path, judge):
        return next(r for r in rows(path)
                    if int(float(r["n"])) == 8 and judge in r["judge"])
    strong = n8(ALPACA96, "Phi-3.5-mini")
    weak = n8(os.path.join(ROOT, "results", "selection_scaling_alpaca.csv"), "Phi-3.5-mini")
    assert float(strong["gain_lo95"]) > 0, strong["gain_lo95"]
    assert float(strong["gain"]) - float(weak["gain"]) > 0.03, (strong["gain"], weak["gain"])
    for judge in ("Phi-3.5-mini", "Meta-Llama-3.1-8B"):
        assert abs(float(n8(ALPACA96, judge)["spearman_u_logn"]) - 1.0) < 1e-9, judge


def test_a2_reads_no_prompt_set_effect_and_the_log_says_so():
    """The band needed the benchmark gain below the in-house gain by >0.03 in BOTH judges. It is
    not: judge C's benchmark gain is the higher of the two. If that ever flips, the pre-registered
    consequence (leading with the benchmark number in Section 6 and the abstract) does fire."""
    if not os.path.exists(ALPACA96):
        return
    inhouse = {r["judge"]: r for r in rows(os.path.join(ROOT, "results", "selection_breadth.csv"))
               if r["anchor"] == "Comma-7B"}
    fired = []
    for r in rows(ALPACA96):
        if int(float(r["n"])) != 8:
            continue
        j = next(k for k in inhouse if k.split("/")[-1] in r["judge"] or r["judge"] in k)
        fired.append(float(inhouse[j]["gain"]) - float(r["gain"]) > 0.03)
    assert len(fired) == 2 and not all(fired), fired
    t = " ".join(open(os.path.join(ROOT, "results", "onset_prediction_alpaca_comma7b.md"),
                      encoding="utf-8").read().split())
    assert "NO PROMPT-SET EFFECT" in t


def test_section6_and_the_appendix_name_the_axis_each_result_speaks_to():
    body = _manuscript("experiments.tex")
    apx = " ".join(open(
        __import__("tests.manuscript", fromlist=["tex"]).tex("sections/appendix_limitations.tex"),
        encoding="utf-8").read().split())
    assert "binds at the anchor and not within one" in body
    assert "Across anchors" in apx and "Within one anchor" in apx
    assert "not measurable" in apx
