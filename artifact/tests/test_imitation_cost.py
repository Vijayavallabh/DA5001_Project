"""Proposition 3's empirical signature, checked against the CSVs it is quoted from.

The proposition says a bucket-metered decoder's realised spend is the cost of imitating p_r, not of
buying utility, so it must saturate once the cap stops binding and then stop tracking the cap. That
is a falsifiable prediction about data this repo already has, and it is the only evidence in the
paper that the proposition describes the deployed decoder rather than an idealisation.

results/utility_price.csv carries the arm means the body quotes; results/imitation_cost.csv carries
the per-step measurement of the proposition's two factors. The two are the same trajectories, which
is why the spend columns must agree -- they did not while one reported medians and the other means,
and a reader would have met 4.2% and 4.3% for one quantity."""
import csv
import re

from tests.manuscript import caption_of, tex

ROWS = {r["k"]: r for r in csv.DictReader(open("results/utility_price.csv"))}
IMIT = {(r["prompt_class"], r["k"]): r
        for r in csv.DictReader(open("results/imitation_cost.csv"))}
TEX = tex("sections/frontier.tex")
# v10 (2026-09-24): the measurements of Proposition~\ref{prop:imitation} moved from the proofs
# appendix to Appendix G (appendix_onset.tex); the proofs appendix keeps the restatements.
APX = tex("sections/appendix_onset.tex")
PRF = tex("sections/appendix_proofs.tex")


def _norm(path):
    return " ".join(open(path, encoding="utf-8").read().split())


def _imitation_rows():
    """Cells of Table~\\ref{tab:imitation}, one list per budget, $ stripped."""
    t = _norm(APX)
    i = t.index(r"\label{tab:imitation}")
    body = t[t.index(r"\midrule", i) + len(r"\midrule"): t.index(r"\bottomrule", i)]
    return [[c.strip().strip("$") for c in r.split("&")] for r in body.split("\\\\") if r.strip()]


def test_the_spend_saturates_while_the_cap_keeps_doubling():
    """k = 10 -> 20 doubles the cap and moves the realised spend by under a tenth of a nat."""
    lo, hi = ROWS["10.0"], ROWS["20.0"]
    assert float(hi["budget_K"]) == 2 * float(lo["budget_K"])
    assert abs(float(hi["mean_spend_nats"]) - float(lo["mean_spend_nats"])) < 0.1
    # and it is monotone up to there, i.e. the cap binds at small k and stops binding
    ks = sorted(ROWS, key=float)
    spend = [float(ROWS[k]["mean_spend_nats"]) for k in ks]
    assert spend[0] < spend[-1], spend
    assert max(spend) - spend[-1] < 0.1, "the largest budget should be at the saturated spend"


def test_the_manuscript_quotes_the_saturated_spend_and_the_unused_allowance():
    # v10: "... and spends $171.3$ nats, $4.3\%$ of the $4000$ it certifies at $k=20$" (v9: "stalls
    # at $171.30$ nats, $4.3\%$ of the allowance"); the cap is now printed too, so it is checked.
    body = _norm(TEX)
    sat = float(ROWS["20.0"]["mean_spend_nats"])
    frac = 100 * sat / float(ROWS["20.0"]["budget_K"])
    m = re.search(r"spends \$([\d.]+)\$ nats, \$([\d.]+)\\%\$ of the \$(\d+)\$ it certifies", body)
    assert m and float(m.group(1)) == round(sat, 2), (m.group(1) if m else None, sat)
    assert abs(float(m.group(2)) - frac) < 0.05, (m.group(2), frac)
    assert float(m.group(3)) == float(ROWS["20.0"]["budget_K"]), (m.group(3), ROWS["20.0"]["budget_K"])


def test_the_two_csvs_describe_the_same_arm():
    """The per-step scan must reproduce utility_price's spend, or the appendix table and the body
    sentence are summarising different trajectories at the same k."""
    for k in ("3", "5", "10", "20"):
        a = float(ROWS[f"{float(k)}"]["mean_spend_nats"])
        b = float(IMIT[("ordinary", k)]["spend_nats"])
        assert abs(a - b) < 0.05, (k, a, b)
        assert int(IMIT[("ordinary", k)]["n_trajectories"]) == int(ROWS[f"{float(k)}"]["n_trajectories"])


def test_the_binding_fraction_the_proposition_leans_on_is_measured():
    """beta is measured, not assumed, and the body quotes it at the two ends and at k=3."""
    # v10 names beta in words: "the share of steps at which it binds falls from ..."
    body = _norm(TEX)
    # v11: "the share of steps its bucket constrains falls from ...", since Table 3's "active" column
    # counts strict blends only and beta also counts the tokens the anchor writes outright.
    m = re.search(r"(?:\$\\beta\$|binds|constrains) falls from \$([\d.]+)\$ at \$k=([\d.]+)\$ to \$([\d.]+)\$ at\s*"
                  r"\$k=(\d+)\$ and \$([\d.]+)\$ at \$k=(\d+)\$", body)
    assert m, "the beta sentence has moved"
    lo_b, lo_k, mid_b, mid_k, hi_b, hi_k = m.groups()
    for b, k in ((lo_b, lo_k), (mid_b, mid_k), (hi_b, hi_k)):
        want = float(IMIT[("ordinary", k.rstrip("."))]["beta_binding_frac"])
        assert abs(float(b) - want) < 5e-4, (k, b, want)
    assert float(lo_b) > float(mid_b) > float(hi_b), m.groups()


def test_the_saturated_imitation_rate_in_the_body_comes_from_the_per_step_scan():
    # v10: "past the imitation rate of $0.857$ nats per token" (v9: "saturates at $0.857$ ...")
    body = _norm(TEX)
    m = re.search(r"(?:saturates at|imitation rate of) \$([\d.]+)\$ nats per token", body)
    rate = float(IMIT[("ordinary", "20")]["imitation_rate_nats_per_token"])
    assert m and abs(float(m.group(1)) - rate) < 5e-4, (m.group(1) if m else None, rate)
    # saturation means the last two budgets agree while the cap doubles
    prev = float(IMIT[("ordinary", "10")]["imitation_rate_nats_per_token"])
    assert abs(rate - prev) < 0.01, (prev, rate)


def test_the_appendix_table_rounds_from_the_csv_once():
    apx = open(APX, encoding="utf-8").read()
    for k in ("0.1", "0.5", "3", "20"):
        r = IMIT[("ordinary", k)]
        row = "${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ \\\\".format(
            k, int(float(r["budget_K"])),
            f'{float(r["beta_binding_frac"]):.4f}',
            f'{float(r["imitation_rate_nats_per_token"]):.3f}',
            f'{float(r["realised_rate_nats_per_token"]):.3f}',
            f'{float(r["spend_nats"]):.1f}',
            f'{100 * float(r["spend_over_cap"]):.1f}\\%',
            f'{float(r["median_cum_spend_vs_step_r2"]):.4f}')
        assert row in apx, row


def test_the_interior_blend_fraction_is_distinguished_from_beta():
    """beta counts the prefix-debt opening too. Conflating the two would put 0.4% where 3.4%
    belongs, which is the mistake the body sentence was rewritten to avoid."""
    r = IMIT[("ordinary", "3")]
    beta, active = float(r["beta_binding_frac"]), float(r["active_frac"])
    # the claim the body sentence rests on is that beta is mostly the prefix-debt opening, not
    # interior blending. Stripping post-EOS padding (dap/stats.py:strip_pad_steps) cut the
    # forced share more than the interior one, so the ratio is 2.7x rather than the 9.9x it
    # read when padding positions were counted as forced steps. The claim is unchanged.
    assert float(r["forced_safe_frac"]) > active, (r["forced_safe_frac"], active)
    assert abs(beta - (active + float(r["forced_safe_frac"]))) < 1e-6
    apx = open(APX, encoding="utf-8").read().replace("\n", " ")
    m = re.search(r"strictly interior blends alone\s+are \$([\d.]+)\\%\$ of steps at \$k=3\$", apx)
    assert m and abs(float(m.group(1)) - 100 * active) < 0.05, (m.group(1) if m else None, active)


def test_the_protected_passages_are_charged_a_higher_rate_than_ordinary_prompts():
    """The claim in the appendix: the meter charges most where the risky model is most distinctive
    from the anchor. If this ever inverted, the sentence would have to go."""
    o = float(IMIT[("ordinary", "20")]["imitation_rate_nats_per_token"])
    p = float(IMIT[("protected", "20")]["imitation_rate_nats_per_token"])
    assert p > o, (o, p)
    apx = _norm(APX)
    # v10: "On the $3{,}300$ protected-passage trajectories the rate is higher, $0.911$ ..."
    m = re.search(r"(?:same )?rate is higher, \$([\d.]+)\$ nats per token", apx)
    assert m and abs(float(m.group(1)) - p) < 5e-4, (m.group(1) if m else None, p)


def test_the_running_spend_is_linear_in_the_step_index_at_every_budget():
    """v10 moved the linearity claim out of Section 3's prose and into Table~\\ref{tab:imitation}'s
    last column (caution (al): a number that moves into a table takes its guard with it). So the
    table must carry EVERY ordinary budget, each with the median R^2 the CSV holds."""
    for (cls, k), r in IMIT.items():
        assert float(r["median_cum_spend_vs_step_r2"]) > 0.9, (cls, k, r)
    ordinary = {k: r for (cls, k), r in IMIT.items() if cls == "ordinary"}
    rows = _imitation_rows()
    assert sorted(float(r[0]) for r in rows) == sorted(float(k) for k in ordinary), \
        "the table no longer reports every budget the claim is about"
    for r in rows:
        want = float(next(v for k, v in ordinary.items() if float(k) == float(r[0]))
                     ["median_cum_spend_vs_step_r2"])
        assert r[-1] == f"{want:.4f}", (r, want)
    assert "step index" in caption_of("tab:imitation"), "the caption no longer says what R^2 regresses"


def test_the_sparsity_proposition_is_stated_and_its_one_number_is_measured():
    """Proposition 5 narrows what the paper used to call open: a causal policy on a budget that
    does not grow with the work MUST be the anchor almost everywhere. Its only empirical claim is
    the contrast -- the deployed rule serves p_r unchanged at 99.95% of steps at k=20."""
    apx = open(PRF, encoding="utf-8").read()
    # The label belongs to the main-text statement and MUST NOT also be set here: it was declared
    # in both places until 2026-09-14, so every \ref{prop:sparse} in the paper resolved to this
    # restatement's number rather than the proposition it names. The appendix restates and proves
    # it without a counter, so what is checked here is the restatement and the bound.
    assert r"\label{prop:sparse}" in open(tex("sections/frontier.tex"), encoding="utf-8").read(), \
        "the proposition is no longer stated in the body"
    for f in (PRF, APX):
        assert r"\label{prop:sparse}" not in open(f, encoding="utf-8").read(), "the duplicate label is back"
    assert r"Proposition~\ref{prop:sparse}, restated" in apx, "the appendix no longer restates it"
    assert r"\mathbb{E}_q[N_\varepsilon] \le K/\varepsilon" in apx, "the bound has changed"
    beta = float(IMIT[("ordinary", "20")]["beta_binding_frac"])
    # Once post-EOS padding stops being counted as steps forced to the anchor, beta at k=20 is
    # 0.0000 and the appendix says so as a beta rather than as a percentage. v10 moved the contrast
    # to Appendix G: the sentence, and beta in Table~\ref{tab:imitation}'s k=20 row.
    assert beta == 0.0, beta
    apx1 = _norm(APX)
    assert "At $k=20$ the decoder serves $p_{r,t}$ unchanged at every step" in apx1, \
        "the sparsity contrast left the appendix"
    r20 = next(r for r in _imitation_rows() if r[0] == "20")
    assert r20[2] == f"{beta:.4f}", ("the table no longer quotes beta at k=20", r20)


def test_the_limitations_no_longer_call_the_shape_question_open():
    """It was 'could in principle concentrate its spend'; Proposition 5 makes that a requirement,
    and only the quantitative question stays open. If the proposition were ever removed, the
    limitation would be an overclaim, so it must cite it and must still name what is open."""
    closing = open(tex("sections/iclr_closing.tex"), encoding="utf-8").read().replace("\n", " ")
    assert r"\ref{prop:sparse}" in closing, "the limitation no longer cites the proposition"
    assert "could in principle concentrate" not in closing, "the weaker claim is back"
    assert re.search(r"(quantitative half|how close a policy of (that|the) shape)", closing,
                     re.IGNORECASE), "the limitation must say what is still open"
    assert "is open" in closing, "the limitation must say that something is open"


def test_the_appendix_prose_carries_the_imitation_rate_and_both_shapes():
    """fig:imitation was cut on 2026-09-19 for the page budget: the table immediately above it
    carried the same rate-versus-cap series numerically, and the concentration shape its second
    panel drew is stated in the prose. Caution (f) -- a missing \\includegraphics halts tectonic and
    leaves the PREVIOUS pdf measurable -- still applies to the figures the document does place, and
    tests/test_figure_shrink.py checks those; what this guards is that neither shape left the paper
    with the picture.
    """
    apx = _norm(APX)
    sat = float(IMIT[("ordinary", "20")]["imitation_rate_nats_per_token"])
    m = re.search(r"(?:imitation rate|realised rate stops at) \$([\d.]+)\$ nats per token", apx)
    assert m and abs(float(m.group(1)) - sat) < 5e-4, (m.group(1) if m else None, sat)
    # the second shape: the spend is spread, not concentrated, which is the trivial horn's evidence.
    # v10 says "spread, not concentrated" where v9 said "nowhere near the left axis"; its two
    # numbers are now checked against the CSV's k=20 row as well as present.
    r20 = IMIT[("ordinary", "20")]
    top = 100 * float(r20["top1pct_of_steps_share_of_spend"])
    f90 = 100 * float(r20["frac_of_steps_for_90pct_of_spend"])
    assert f"busiest $1\\%$ of steps carry ${top:.1f}\\%$" in apx, \
        ("the concentration shape left the paper with the figure", top)
    assert f"covering $90\\%$ takes ${f90:.0f}\\%$ of the sequence" in apx, f90
    assert "spread, not concentrated" in apx or "nowhere near the left axis" in apx, \
        "the prose no longer says the deployed rule misses the shape Prop 3 requires"

def test_the_lorenz_curves_end_at_one_and_lie_above_the_diagonal():
    """The claim the figure makes with them: the spend is spread over the sequence, barely above
    uniform, which is the opposite of what Proposition 5 needs."""
    import csv as _csv
    from collections import defaultdict
    curves = defaultdict(list)
    for r in _csv.DictReader(open("results/imitation_lorenz.csv")):
        curves[(r["prompt_class"], r["k"])].append(
            (float(r["frac_of_steps"]), float(r["frac_of_spend"])))
    assert curves
    for key, v in curves.items():
        v.sort()
        assert abs(v[-1][1] - 1.0) < 1e-6, (key, v[-1])
        assert all(y >= x - 1e-9 for x, y in v), key          # above the uniform diagonal
        assert all(b >= a - 1e-9 for (_, a), (_, b) in zip(v, v[1:])), key
    half = dict(curves[("ordinary", "20")])[0.5]
    assert 0.5 < half < 0.9, half     # spread, not concentrated: half the steps carry most of it


def test_strip_pad_steps_trims_a_padded_tail_and_leaves_a_forced_step_alone():
    """The post-EOS tail is not decode steps. Two earlier predicates got this wrong in opposite
    directions and both would have put wrong numbers in the paper, so both mistakes are pinned:
    a real step forced to the anchor must survive, and a pad position whose risky probability has
    decayed to 0.01 must not."""
    from dap.stats import strip_pad_steps

    def pad(tid=128000, p_r=1.0):
        return dict(t=0, k_t=0.0, a_t=0.0, bd=0.0, p_star_prob=0.9999975,
                    p_s_prob=0.9999975, p_risky_prob=p_r, sampled_token_id=tid)

    def real(tok=42, p_s=0.28):
        return dict(t=0, k_t=0.0, a_t=0.0, bd=0.0, p_star_prob=p_s, p_s_prob=p_s,
                    p_risky_prob=0.0, sampled_token_id=tok)

    spend = dict(t=0, k_t=17.4, a_t=1.48, bd=1.0, p_star_prob=0.14, p_s_prob=0.02,
                 p_risky_prob=0.14, sampled_token_id=7)

    # an empty generation: one real EOS step then 199 pad positions
    log = [real(tok=128001)] + [pad() for _ in range(199)]
    assert len(strip_pad_steps(log)) == 1

    # the risky model's probability decays along a real tail; requiring it near 1 left 4,815
    # pad steps in the k=20 arm
    log = [spend] * 3 + [pad(p_r=1.0), pad(p_r=0.5), pad(p_r=0.0108)]
    assert len(strip_pad_steps(log)) == 3

    # a genuine run of steps forced to the anchor is NOT padding: different tokens
    log = [spend] + [real(tok=t) for t in (11, 12, 13)]
    assert len(strip_pad_steps(log)) == 4

    # nor is one where the anchor is merely confident but not a point mass, even repeating a token
    log = [spend] + [real(tok=11, p_s=0.9) for _ in range(3)]
    assert len(strip_pad_steps(log)) == 4

    # a single trailing pad-looking step is ambiguous and is kept
    log = [spend] * 3 + [pad()]
    assert len(strip_pad_steps(log)) == 4

    # nothing to trim, and degenerate inputs
    assert strip_pad_steps([]) == []
    assert len(strip_pad_steps([spend] * 5)) == 5


def test_no_imitation_csv_was_computed_over_padded_logs():
    """A regression guard on the arms themselves: if any arm's beta at its largest budget is
    inflated by padding again, the realised rate stops matching the imitation rate."""
    import glob as _glob
    for path in _glob.glob("results/imitation_cost*.csv"):
        if "lorenz" in path:
            continue
        rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
                if r["prompt_class"] == "ordinary" and r["k"] == "20"]
        for r in rows:
            i = float(r["imitation_rate_nats_per_token"])
            v = float(r["realised_rate_nats_per_token"])
            assert abs(v / i - 1) < 0.01, (path, i, v, "padding counted as decode steps?")


def test_the_linearity_claim_is_scoped_by_the_trajectory_count_it_quotes():
    """'linear in the step index at every budget (median R^2 >= 0.97)' is TRUE of one prompt class.

    Added 2026-09-18. results/imitation_cost.csv carries two classes: `ordinary` (16,500
    trajectories, min R^2 0.9794) and `protected` (3,300, min 0.9318). The sentence is correct
    because its own clause says "over 16,500 logged trajectories" -- but the scope sits three clauses
    from the adjective it licenses, which is the shape caution (ao) describes. If the arm is ever
    re-run over both classes, "at every budget" becomes false and nothing else would say so.

    So guard the pair: the count in the prose must select the class the claim holds of.
    """
    import csv as _csv
    import os as _os
    from collections import defaultdict
    from tests.manuscript import body
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    by = defaultdict(list)
    with open(_os.path.join(root, "results", "imitation_cost.csv"), encoding="utf-8") as fh:
        for r in _csv.DictReader(fh):
            if r["prompt_class"] == "prompt_class":
                continue
            by[r["prompt_class"]].append(
                (float(r["median_cum_spend_vs_step_r2"]), int(r["n_trajectories"])))

    # v10: the claim is Table~\ref{tab:imitation}'s R^2 column, scoped by its caption's count
    # ("$16{,}500$ logged trajectories"); the prose qualifier "median R^2 >= 0.97" became the
    # printed values, so the bar is checked on what the table prints.
    txt = caption_of("tab:imitation")
    printed = [float(r[-1]) for r in _imitation_rows()]
    assert printed and min(printed) >= 0.97, ("the linearity column no longer clears 0.97", printed)
    quoted = {c: sum(n for _r, n in v) for c, v in by.items()}
    assert f"${quoted['ordinary']:,}".replace(",", "{,}") + "$ logged trajectories" in txt, \
        ("the caption no longer quotes the ordinary class's trajectory count, which is what scopes "
         "'at every budget'", quoted)
    assert min(r for r, _n in by["ordinary"]) >= 0.97, by["ordinary"]
    # and the scope is load-bearing: the other class does NOT clear the bar
    assert min(r for r, _n in by["protected"]) < 0.97, \
        ("both classes now clear 0.97, so the claim could be stated unscoped -- revisit the wording "
         "deliberately rather than leaving it narrower than the evidence", by["protected"])

# RETIRED 2026-09-19, appendix reduction. The paragraph each of these read was removed
# when the appendix was cut from 52 pages, so the sentence they pinned no longer exists.
# A guard for a claim the paper does not make protects nothing; recorded here rather than
# silently deleted, so the removal is visible to the next reader:
#   test_the_spend_concentration_beside_the_sparsity_proposition_rounds_from_the_csv
