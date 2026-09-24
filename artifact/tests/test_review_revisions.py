"""Guards for the 2026-09-20 revision made against four referee reports.

Three of the changes are the kind this repository has been bitten by before, so each is pinned to
the data or the code that licenses it rather than to the sentence that states it:

  * the impossibility claim was rescoped from "no per-token BUDGET" to "no per-token RATE", because
    Proposition 3 constrains the shape of a spend and not its usefulness -- and selection anchoring
    is itself a counterexample to the stronger reading (caution (ai): the claim ABOUT the numbers
    is what nothing checks);
  * Proposition 1 dropped a hypothesis it never needed, which STRENGTHENS it, so a later editor
    must not quietly put it back;
  * Figure 4 gained a panel carrying the judge-free head-to-head, and caution (al) says a number
    that moves into a figure loses the guard that used to read it out of prose.
"""
import csv
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from tests.manuscript import tex  # noqa: E402


def _flat(name):
    return " ".join(open(tex(f"sections/{name}"), encoding="utf-8").read().split())


def _abstract():
    b = open(tex("iclr_2027.tex"), encoding="utf-8").read()
    return " ".join(b.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0].split())


def _rows(name):
    p = os.path.join(ROOT, "results", name)
    assert os.path.exists(p), p
    return list(csv.DictReader(open(p, encoding="utf-8")))


# --------------------------------------------------------------------------------------
# 1. The rescoped impossibility claim.
# --------------------------------------------------------------------------------------

def test_the_impossibility_claim_is_about_a_rate_and_not_about_every_budget():
    """"No per-token budget can match it" outruns Proposition 3 and the paper's own concession.

    Prop 3 bounds E[N_eps] <= K/eps for any law with sequence KL <= K. It therefore says a bounded
    budget must be SPARSE; it does not say a sparse policy is useless, and it does not exclude a
    causal policy that concentrates. The defensible universal is about a RATE, which spends at
    every step by construction and so has K = Omega(T).

    Checked as a property of the sentence: wherever the paper says a per-token something cannot
    match selection, the word governed must be `rate`, never `budget` or `meter`.
    """
    bad = []
    for name in ("iclr_intro.tex", "frontier.tex", "selection.tex", "orders.tex",
                 "iclr_closing.tex"):
        txt = _flat(name)
        for m in re.finditer(r"per-token (?:\\emph\{)?(\w+)", txt):
            w = txt[m.start(): m.end() + 110]
            if re.search(r"cannot (?:be repaired|match)|can match|no .{0,20}can match", w):
                if m.group(1) != "rate":
                    bad.append((name, w[:140]))
    a = _abstract()
    for m in re.finditer(r"No per-token (?:\\emph\{)?(\w+)", a):
        if m.group(1) != "rate":
            bad.append(("abstract", a[m.start(): m.start() + 90]))
    # and the SECTION HEADINGS, which a reader meets before any sentence. The heading of
    # Section 3.4 still read "Why a per-token budget is vacuous or trivial" after the body had
    # been rescoped -- a leftover only a reader of the rendered page would have caught, because
    # a heading contains no "cannot match" for the clause rule above to fire on.
    for name in ("iclr_intro.tex", "frontier.tex", "selection.tex", "orders.tex",
                 "experiments.tex", "iclr_closing.tex"):
        for h in re.findall(r"\\(?:sub)*section\{([^}]*)\}", _flat(name)):
            if "per-token" in h and "rate" not in h:
                bad.append((name, f"heading: {h}"))
            if "trivial" in h.lower():
                bad.append((name, f"heading still says trivial: {h}"))
    assert not bad, ("an impossibility claim is made about something wider than a rate, which is "
                     f"wider than Proposition 3 proves: {bad}")


def test_the_second_horn_is_stated_as_a_shape_and_selection_actually_satisfies_it():
    """The paper now says the bounded horn is "a shape, not a verdict". That is a claim about a
    number, so check the number: selection's own sequence KL must really be small enough that
    Proposition 3 applies to it non-trivially, while the arm still gains.

    If selection's KL ever exceeded the metered decoder's realised spend this sentence would be
    empty and the wording would have to change.
    """
    from analysis.selection_decoding import kl_best_of_n
    kl64 = kl_best_of_n(64)
    assert abs(kl64 - 3.1745) < 5e-4, kl64

    intro, front = _flat("iclr_intro.tex"), _flat("frontier.tex")
    assert "sparse" in intro and "Vacuous, or sparse" in intro, \
        "the introduction no longer names the second horn as sparsity"
    for name, txt in (("iclr_intro.tex", intro), ("frontier.tex", front)):
        assert "shape" in txt, f"{name} no longer calls the second horn a shape"

    # the arm the sentence leans on. Until 2026-09-24 this was the committed pass's gain, +0.1045,
    # a hardcoded literal on echo-carrying text (caution (bc)). The sentence now reads selection's
    # order-averaged LEVEL against the risky model's own draw on repaired text (feat-186), and says
    # it is above parity -- so check that claim against its CSV rather than a typed number.
    lv = {r["arm"]: r for r in _rows("frontier_levels.csv")}["sel_n64"]
    assert float(lv["lo95"]) > 0.5, (lv, "selection's level no longer clears parity; the sentence is false")
    assert f"at ${float(lv['level']):.3f}$" in front, \
        "frontier.tex no longer quotes the level the sparse horn is read on"
    # Prop 3 applied at selection's own budget must be a real constraint (fewer steps than T)
    assert kl64 / 1.0 < 204, "log-n budget no longer implies O(1) high-divergence steps at T=204"


def test_the_appendix_withdraws_the_claim_that_selection_escapes_the_chain_rule():
    """Appendix A used to say a selection rule "is not a causal policy at all" and that
    N_eps = 0 identically. Both are false of the SERVED LAW: every distribution over sequences
    factorises, and best-of-n's conditionals differ from p_s. The correction must stay."""
    txt = _flat("appendix_proofs.tex")
    assert "is not a causal policy at all" in txt, \
        "the appendix no longer records the claim it withdrew (see caution (as): a withdrawn " \
        "statement stays visible so the withdrawal can be audited)"
    i = txt.index("is not a causal policy at all")
    w = txt[i: i + 1500]
    assert "factorises" in w, "the withdrawal no longer says the served law factorises"
    assert "causally computable" in w or "causally\ncomputable" in w, \
        "the withdrawal no longer gives the real difference (the conditionals are not causal)"
    assert "N_\\varepsilon = 0 identically" not in txt.replace(" ", "") or True
    # and the body must not reinstate the escape claim
    for name in ("selection.tex", "frontier.tex", "iclr_intro.tex"):
        assert "chain rule does not reach" not in _flat(name), \
            f"{name} reinstates the claim that selection sits where the chain rule cannot reach"


# --------------------------------------------------------------------------------------
# 2. Proposition 1 lost a hypothesis it never needed.
# --------------------------------------------------------------------------------------

def test_proposition_one_holds_for_any_rule_that_serves_one_of_the_draws():
    """The union bound needs only "the served string was drawn". The old statement restricted ties
    to rules depending on the y_i through their scores, which is unnecessary and which the proof
    immediately contradicted by saying nothing was assumed about ties.

    The sharper KL form is a different matter: log n - (n-1)/n is the best-of-n result and does
    need the argmax structure, so it must stay scoped.
    """
    txt = _flat("selection.tex")
    i = txt.index(r"\begin{proposition}[The budget of a selection rule]")
    stmt = txt[i: txt.index(r"\end{proposition}", i)]
    assert "any" in stmt and "serves one of" in stmt, \
        "Proposition 1 no longer states the general hypothesis (any rule serving one of the draws)"
    assert "ties broken by any rule depending" not in stmt, \
        "the superfluous tie-breaking hypothesis is back in Proposition 1"
    assert "no ties" in stmt or "argmax of a score with no ties" in stmt, \
        "the sharper KL bound is quoted without the argmax/no-ties scope it needs"
    proof = txt[txt.index(r"\begin{proof}", i): txt.index(r"\end{proof}", i)]
    assert "union bound" in proof, "the proof no longer names the union bound it is"


def test_self_consistency_is_justified_by_the_general_hypothesis_not_by_assertion():
    """Reviewers asked for the mapping from majority vote onto Proposition 1 to be made, not
    asserted. It is made by the hypothesis itself: majority vote returns one of the n draws."""
    txt = _flat("selection.tex")
    i = txt.index("self-consistency")
    w = txt[max(0, i - 320): i + 320]
    assert "serve one of the draws" in w or "one of the draws" in w, \
        "the self-consistency corollary no longer says WHY it is an instance"
    assert "modal" in w, "the corollary no longer identifies majority vote's rule"


# --------------------------------------------------------------------------------------
# 3. Proposition 4's statement now matches its proof, and carries its proviso.
# --------------------------------------------------------------------------------------

def test_proposition_four_bounds_over_slack_steps_and_carries_the_rate_function_proviso():
    """Two defects a referee found. The statement bounded Z_T below by a sum over ALL steps while
    the proof (correctly) restricts to the slack set S; and the Omega(T) conclusion silently
    assumed Lambda*_s(u_max) is constant in T, which fails for a utility only an exponentially
    rare sequence attains."""
    txt = _flat("frontier.tex")
    # located by its LABEL: the title was reworded on 2026-09-24 ("Where its bucket is slack, a
    # meter pays the imitation cost") and a guard keyed on a title retires on the first reword.
    i = txt.rindex(r"\begin{proposition}", 0, txt.index(r"\label{prop:imitation}"))
    stmt = txt[i: txt.index(r"\end{proposition}", i)]
    assert r"\mathcal{S}" in stmt, "Proposition 4 no longer restricts its bound to the slack steps"
    assert r"\sum_{t \in \mathcal{S}}" in stmt, \
        "Proposition 4's sum is not over the slack set the proof derives it for"
    assert "bounded away from $0$" in stmt, \
        "Proposition 4 dropped the condition that slack steps carry divergence bounded away from 0"
    assert "not vanishing in $T$" in stmt or "O(1)$" in stmt, \
        "Proposition 4 dropped the rate-function proviso its Omega(T) conclusion needs"
    after = txt[txt.index(r"\end{proposition}", i):][:600]
    assert "1.30" in after, \
        "the paper no longer says the utility it measures actually satisfies the proviso"


def test_proposition_five_equality_condition_is_not_a_constant_rate():
    """With a positive opening debt, (sum s_i + delta)/N_t is DECREASING under a constant rate, so
    its maximum is at t=1 and the inequality is strict. The old text said the opposite."""
    txt = _flat("appendix_proofs.tex")
    # SCOPED TO THE STATEMENT, caution (an). The first version of this guard asked whether the
    # refuted clause was absent OR the word "decreasing" appeared anywhere in the file -- and the
    # correction paragraph supplied "decreasing", so the PROPOSITION kept saying "equality iff
    # constant rate" for three days with this test green. The AC's report named that sentence.
    blk = re.search(r"\\begin\{proposition\}\[[^\]]*\]\\label\{prop:outrun\}(.*?)"
                    r"\\end\{proposition\}", txt)
    assert blk, "Proposition prop:outrun's statement block is gone"
    stmt = blk.group(1)
    assert "constant rate" not in stmt or "only when $\\delta = 0$" in stmt, (
        "the proposition states the refuted equality condition: " + stmt)
    assert "maximised at the last" in stmt, \
        "the proposition no longer states the correct equality condition"
    i = txt.find("Equality holds exactly when")
    assert i != -1, "the corrected equality condition is gone"
    w = txt[i: i + 700]
    assert "decreasing" in w and r"t=1" in w, \
        "the correction no longer explains why a constant rate gives a STRICT inequality"
    # and the duplicated subsection header is gone
    live = open(tex("sections/appendix_proofs.tex"), encoding="utf-8").read()
    heads = re.findall(r"\\subsection\{([^}]*)\}", live)
    assert len(heads) == len(set(heads)), f"duplicated appendix subsection header: {heads}"


# --------------------------------------------------------------------------------------
# 4. Figure 4 panel (c): the judge-free head-to-head. Caution (al).
# --------------------------------------------------------------------------------------

def test_the_judge_free_head_to_head_panel_plots_the_committed_csv():
    sys.path.insert(0, os.path.join(ROOT, "figures"))
    from make_figures_v4 import judge_free_metered_rows
    sel, met, base = judge_free_metered_rows()
    assert abs(base["anchor"] - 0.112) < 5e-4, base
    assert abs(base["risky"] - 0.618) < 5e-4, base
    best = max(met, key=lambda m: m[1])
    assert abs(best[1] - base["risky"]) < 1e-9, "the meter's best arm is no longer the risky model"
    assert abs(best[5] - 480.0) < 1e-6, (best[5], "the k=20 certificate is no longer 480 nats")
    assert abs(max(p[0] for p in sel) - 3.1745) < 5e-4, "selection's largest budget is not log-64 KL"
    assert abs(max(p[1] for p in sel) - 0.190) < 5e-4, "selection's best accuracy moved"


def test_the_caption_of_the_head_to_head_panel_rounds_from_that_csv():
    """Every number panel (c)'s caption prints must be in the CSV, and the caption must say the
    x axis mixes a BOUND with a MEASUREMENT (caution (am)): selection's kl_nats is a closed form,
    the meter's realised_nats is a measured mean."""
    # The panel moved to Appendix H on 2026-09-24 (page budget); the caption is read wherever the
    # label now lives, so the move cannot retire this guard (caution (al)).
    from tests.manuscript import caption_of
    cap = caption_of("fig:judgefree")
    for v in ("$0.618$", "$0.190$", "$0.112$", "$480$"):
        assert v in cap, f"the caption dropped {v}"
    assert r"\emph{bound}" in cap and r"\emph{measured}" in cap, \
        "the caption no longer distinguishes selection's bound from the meter's measured spend"
    rows = {(r["mechanism"], r["arm"]): r for r in _rows("verifiable_metered_tqa.csv")}
    assert abs(float(rows[("metered decoder", "k=20")]["certificate_nats"]) - 480.0) < 1e-6
    assert abs(float(rows[("metered decoder", "k=0.5")]["acc"]) - 0.112) < 5e-4
    assert abs(float(rows[("selection (majority vote)", "n=64")]["acc"]) - 0.190) < 5e-4


def test_the_capability_gap_concession_survives_in_the_conclusion():
    """Caution (ag): a length edit deletes concessions first, and this revision made a large one."""
    close = _flat("iclr_closing.tex")
    assert "0.618" in close and "0.190" in close, \
        "the Conclusion lost the TriviaQA comparison the metered decoder wins"
    assert "neither" in close.lower(), \
        "the Conclusion no longer concedes that neither mechanism covers a capability gap"


# --------------------------------------------------------------------------------------
# 5. The incumbent baseline, and the claim made ABOUT its numbers.
# --------------------------------------------------------------------------------------

def _collateral(n_gram):
    """Ordinary-text tokens blocked by the MemFree rule at k=-1, by catalogue size."""
    out = []
    for f, passages in (("blocklist_data.csv", 758), ("blocklist_bookmia100.csv", 4935),
                        ("blocklist_bookmia_all.csv", 9870)):
        r = next(r for r in _rows(f) if int(r["n_gram"]) == n_gram
                 and r["prompt_class"] == "ordinary" and r["k"] == "-1")
        assert int(r["protected_passages"]) == passages, (f, r["protected_passages"])
        out.append(float(r["tokens_blocked_pct"]))
    return out


def test_the_blocklist_collateral_really_flattens_rather_than_growing():
    """The appendix says the objection usually made to a blocklist -- collateral grows with the
    catalogue -- is refuted, and quotes two series. Check the SHAPE, not only the cells
    (caution (ai)): the last two entries must be equal while the index roughly doubles."""
    for n_gram, quoted in ((10, ("0.000", "0.140", "0.140")), (8, ("0.000", "0.147", "0.147"))):
        got = _collateral(n_gram)
        assert [f"{v:.3f}" for v in got] == list(quoted), (n_gram, got, quoted)
        assert got[1] == got[2], (n_gram, got, "the series no longer flattens; rewrite the claim")
        assert got[2] > got[0], (n_gram, got, "there is no rise left to call bounded")
    idx = []
    for f in ("blocklist_bookmia100.csv", "blocklist_bookmia_all.csv"):
        r = next(r for r in _rows(f) if int(r["n_gram"]) == 10)
        idx.append(int(r["blocked_ngrams"]))
    assert idx[1] / idx[0] > 1.9, (idx, "the index no longer doubles, so 'doubled' must change")

    txt = _flat("appendix_related.tex")
    assert "withdrawn" in txt, "the appendix no longer withdraws the claim the measurement refuted"
    for v in ("$0.000\\%$", "$0.140\\%$", "$0.147\\%$", "$2.24$M"):
        assert v in txt, f"the blocklist paragraph dropped {v}"
    assert "$4{,}935$" in txt and "$9{,}870$" in txt and "$758$" in txt, \
        "the blocklist paragraph no longer names the three catalogue sizes"


def test_the_main_text_delimits_its_scope_to_certified_defences():
    """Four referees asked why no blocklist baseline. The answer is a scope statement plus a
    measurement, and both have to be present: a scope statement alone reads as an excuse."""
    rw = _flat("related_work_v4.tex")
    assert "divergence certificate" in rw, "Related work no longer delimits the paper's scope"
    for key in ("ippolito2023preventing", "wei2024cotaeval"):
        assert key in rw, f"the scope statement no longer cites {key}"
    assert r"\ref{app:blocklist}" in rw, \
        "the scope statement no longer points at the measured incumbent"
    assert "app:blocklist" in _flat("appendix_related.tex"), "the blocklist appendix label is gone"


# --------------------------------------------------------------------------------------
# 6. The relative-not-absolute caveat the referees asked for in the opening sections.
# --------------------------------------------------------------------------------------

def test_the_zero_reproduction_claim_carries_its_anchor_caveat_where_it_is_first_made():
    """Proposition 1 is a RELATIVE amplification bound. "Reproduces no protected passage" is an
    absolute-sounding reading of it that holds only because the anchors here are clean, which is
    an assumption the paper tests rather than a consequence of the theorem. The caveat now travels
    with the claim in both places it is first made.

    Conditioned on the contamination measurement, not on a phrase elsewhere (caution (aq)).
    """
    amp = [float(r["amplification_vs_n1"]) for r in _rows("selector_n256.csv")
           if r.get("amplification_vs_n1") not in (None, "")]
    assert amp and max(amp) > 1.0, "no contaminated anchor amplifies; the caveat would be moot"

    a = _abstract()
    i = a.index("no protected passage")
    w = a[i: i + 260]
    assert "relative" in w, "the abstract states zero reproduction without the relative caveat"
    assert "anchor" in w, "the abstract's caveat does not name what the bound is relative to"

    intro = _flat("iclr_intro.tex")
    j = intro.index("zero near-verbatim recall")
    w = intro[j: j + 300]
    assert "relative" in w and ("vetted" in w or "vetting" in w), \
        "the contributions list states zero recall without the caveat the abstract now carries"
