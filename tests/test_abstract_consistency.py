"""Every number in the abstract has to appear in the body that justifies it.

The abstract is edited under a page budget and the body is edited for content, so they drift in
opposite directions; audit_numbers.py cannot catch it because it only asks whether a literal appears
in SOME CSV, and an abstract number that has gone stale still does."""
import re

from tests.manuscript import tex

# appendix_seed was retired on 2026-09-19 (page budget): the seed and anchor-warp interventions
# left the manuscript, so the file is no longer part of the body this guard scans.
# v10 (2026-09-24): onset, orders, appendix_opening, appendix_robustness and appendix_second_anchor
# were retired too (kept as *_v9_2026-09-24.tex, never compiled); their content now lives in
# frontier, appendix_onset and appendix_proofs. These are the section files the manuscript \inputs.
# fig_overview is left out on purpose: it is TikZ, and a coordinate such as 0.22 is a number that
# would satisfy an abstract literal by coincidence.
SECTIONS = ("iclr_intro", "selection", "frontier", "experiments", "related_work_v4",
            "iclr_closing", "appendix_proofs", "appendix_selection", "appendix_onset",
            "appendix_limitations", "appendix_related")
# Numbers that are structural rather than measured: page/section counts, an exponent, a budget the
# body writes as k = 10 rather than $10$.
ALLOWED = {"1", "2", "10"}


def _abstract():
    body = open(tex("iclr_2027.tex"), encoding="utf-8").read()
    return body.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0]


def _literals(text):
    return set(re.findall(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])",
                          " ".join(re.findall(r"\$([^$]*)\$", text))))


def test_every_number_in_the_abstract_appears_in_the_body():
    body = "".join(open(tex(f"sections/{s}.tex"), encoding="utf-8").read() for s in SECTIONS)
    body_nums = _literals(body) | set(re.findall(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])", body))
    missing = sorted(n for n in _literals(_abstract()) - body_nums if n not in ALLOWED)
    assert not missing, f"in the abstract but nowhere in the body: {missing}"


def test_the_abstract_the_intro_and_the_onset_section_agree_on_the_pair_count():
    """The count is spelled in words, so audit_numbers.py cannot see it at all, and it has to be
    changed in three places every time a pair is added. The phrasing is free; the number is not."""
    import csv
    n = len([r for r in csv.DictReader(open("results/onset_table.csv"))
             if not r["pair"].startswith("ALL")])
    words = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
             "eleven", "twelve"]
    word = words[n]
    # v10 (2026-09-24): onset.tex was retired and the onset prose moved into frontier.tex (Section 3
    # and Figure horns); the abstract and the introduction now give the pair count alone ("over
    # nine pairs", "across nine pairs"), and the anchor count beside it survives in Figure horns'
    # caption. So every spelled pair count in the three places must be the CSV's, and the one
    # sentence that pairs it with the anchor count must agree with it too.
    places = {"abstract": _abstract(),
              "intro": open(tex("sections/iclr_intro.tex"), encoding="utf-8").read(),
              "onset (frontier.tex)": open(tex("sections/frontier.tex"), encoding="utf-8").read()}
    counted = rf"\b({'|'.join(words[1:])}) (?:model |\(anchor, memoriser\) )?pairs\b"
    for where, body in places.items():
        found = re.findall(counted, " ".join(body.split()), flags=re.I)
        # v11 (2026-09-24): the abstract no longer quotes the measured onset band or its pair count
        # (it states the vacuity theorem and the window exposure instead); a count it DOES state must
        # still be the CSV's. The introduction (Figure 1's caption) and Section 3 must state it.
        if where != "abstract":
            assert found, f"{where} does not state the pair count"
        assert all(f.lower() == word for f in found), (where, found, word)
    fr = " ".join(places["onset (frontier.tex)"].split())
    found = re.findall(r"(\w+) (?:model |\(anchor, memoriser\) )?pairs with (\w+) distinct anchors",
                       fr)
    assert found, "frontier.tex does not state the pair count beside the anchor count"
    assert all(p.lower() == a.lower() == word for p, a in found), (found, word)


def test_every_results_file_the_paper_names_exists():
    """A citation to a file that is not in the artifact is a reviewer's first click and a desk
    rejection risk on the artifact track. Names are typeset with escaped underscores."""
    import glob
    import os
    body = "".join(open(f, encoding="utf-8").read()
                   for f in [tex("iclr_2027.tex")] + sorted(glob.glob(tex("sections/*.tex")))
                   if os.path.basename(f).replace(".tex", "") in SECTIONS or f.endswith("iclr_2027.tex"))
    # Long paths carry \allowbreak breakpoints after / and \_ so a 50-character \texttt token
    # cannot strand the line before it (2026-09-17; 158 underfull hboxes -> 13). They are
    # zero-width and print nothing, but they are inside the path as far as a regex is concerned.
    body = body.replace("\\allowbreak ", "").replace("\\allowbreak", "")
    missing = []
    # a trailing `*` names a family (v10: `results/onset_prediction_*.md`, every registration), and
    # a family is only honest if at least one file on disk belongs to it
    for n, star, ext in set(re.findall(r"results/([A-Za-z0-9_\\]+)(\*?)((?:\.[a-z]+)?)", body)):
        n = n.replace("\\_", "_")
        if star:
            if not glob.glob(f"results/{n}*{ext}"):
                missing.append(f"{n}*{ext}")
        elif n and not (os.path.exists(f"results/{n}.csv") or os.path.exists(f"results/{n}.md")):
            missing.append(n)
    missing = sorted(set(missing))
    assert not missing, f"named in the paper, absent from results/: {missing}"


# ---------------------------------------------------------------------------
# Added 2026-09-19, after cutting the abstract to 200 words. Mutation-testing the CUT (rather
# than trusting 680 green tests) found three claims the abstract could lose, or silently invert,
# with nothing failing. All three are caution (ai) one level up: the literals were guarded, and
# the sentences around them were not.


def test_the_abstract_price_is_the_measured_wall_clock_and_not_the_flop_proxy():
    """Caution (ae): $61.3\\times$ is a parameter-count FLOP proxy, $35.4\\times$ is measured.

    Swapping one for the other in the abstract fired nothing, because
    test_every_number_in_the_abstract_appears_in_the_body is satisfied by either -- the body
    legitimately quotes the proxy for *forward passes* in Section 3. The two are not
    interchangeable and the error is directional: the proxy implies a smaller scorer cuts the
    cost, and the measurement says the draws are 90.7% of the clock and the scorer 9.3%, so the
    lever is n. A deployer reads the abstract's number as a price, which only one of them is.
    """
    import csv as _csv
    lat = {r[0]: r for r in _csv.reader(open("results/serving_latency.csv")) if r}
    measured = float(lat["ratio, measured"][2])
    proxy = float(lat["ratio, analytical (serving_cost.py)"][2])
    assert abs(measured - proxy) > 1.0, "the two ratios have converged; this guard is moot"
    a = " ".join(_abstract().split())
    # Either measured ratio: the harness's end-to-end 35.4x, or the deployable 64 * C/D (21.8x,
    # results/anchor_only_cost.csv) that caution (ay) showed is the per-request price. Never the
    # FLOP proxy, which is the directional error this guard exists for.
    ao = {int(r["width"]): r for r in _csv.DictReader(open("results/anchor_only_cost.csv"))}
    deploy = 64 * float(ao[200]["C_over_D"])
    assert f"${measured:.1f}\\times$" in a or f"${deploy:.1f}\\times$" in a, \
        f"the abstract must quote a MEASURED wall-clock ratio (${measured:.1f}x or ${deploy:.1f}x)"
    # the proxy may appear only if it is labelled as forward passes, never as a price
    tok = f"${proxy:.1f}\\times$"
    if tok in a:
        i = a.index(tok)
        w = a[max(0, i - 140): i + 140]
        assert "forward pass" in w or "FLOP" in w, \
            f"the abstract prints the FLOP proxy {tok} without saying it is forward passes"


def test_the_abstract_states_the_certificate_at_the_order_and_exactness_it_proves():
    """Proposition 1 gives D_inf(q||p_s) <= log n: the PATHWISE order, and an exact constant.

    "a bound of about $\\log n$ nats" passed every test in the suite. It understates the order
    (the contribution is that the strongest Renyi order is free, not that some order is bounded)
    and overstates the slack (log n is exact; it is the sharper KL form that carries -(n-1)/n).
    Guard the two adjectives against the proposition, not the spelling of the sentence.
    """
    from tests.manuscript import body as _body
    a = " ".join(_abstract().split())
    assert a.count(r"\log n") == 1, "the abstract states the certificate more than once"
    i = a.index(r"\log n")
    w = a[max(0, i - 200): i + 200]
    assert "pathwise" in w, "the abstract no longer calls the certificate pathwise"
    assert "exactly" in w, "the abstract no longer says log n is exact"
    for hedge in (" about ", " roughly ", " approximately ", " around "):
        assert hedge not in w, f"log n is exact; the abstract hedges it with '{hedge.strip()}'"
    assert r"D_\infty" in _body("selection.tex"), \
        "Section 3 no longer proves the infinity-order bound the abstract advertises"


def test_the_judge_free_lift_rounds_from_the_csv_and_the_section_still_makes_the_claim():
    """The judge-free claim could be dropped from the abstract with nothing failing.

    test_selection_verifiable.py::test_the_abstract_claims_the_judge_free_axis_only_because_it
    _was_measured is conditional on "no judge" appearing, and its withdrawal branch asserted the
    phrase "Off the judge" was absent from experiments.tex. The v8 restructure reworded that
    heading, so the branch became vacuously true: the guard returned a pass on an abstract with
    the claim deleted. That is the THIRD instance of caution (aj) -- and it happened inside the
    guard whose own docstring warns about it, because the repair was itself a phrase trigger.

    Condition on the measured arm instead. While the CSV holds the lift and Section 4 reports it,
    the abstract carries it, and both numbers round from the CSV once (caution (j)).
    """
    import csv as _csv
    from tests.manuscript import body as _body
    rows = [r for r in _csv.DictReader(open("results/selection_verifiable_comma7b.csv"))
            if r["arm"].startswith("majority")]
    mv = {int(r["n"]): r for r in rows}
    best = max(mv.values(), key=lambda r: float(r["acc"]))
    import math as _math
    a = " ".join(_abstract().split())
    assert "no judge" in a, "the abstract dropped the judge-free claim while the arm still stands"
    assert f"${float(mv[1]['acc']):.3f}" in a and f"{float(best['acc']):.3f}$" in a, \
        (mv[1]["acc"], best["acc"], "the GSM8K lift does not round from the CSV")
    exp = _body("experiments.tex")
    assert "no judge" in exp, "Section 4 dropped the judge-free claim the abstract makes"
    # v10 (2026-09-24): the abstract dropped the draw count ("$32$ draws"); Section 4 states the lift
    # with its n and its certificate, so the best-arm check is made there, every number from the CSV
    n = int(best["n"])
    assert (f"from ${float(mv[1]['acc']):.3f}$ to ${float(best['acc']):.3f}$ at $n={n}$ "
            f"(${_math.log(n):.2f}$ nats)") in exp, \
        (n, "Section 4's judge-free lift is not quoted at the best arm's n and log n")


from tests.manuscript import ROOT  # noqa: E402


def _abstract_text():
    from tests.manuscript import tex as _tex
    b = open(_tex("iclr_2027.tex"), encoding="utf-8").read()
    return " ".join(b.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0].split())


def test_the_abstract_says_self_consistency_is_an_instance_because_the_csv_shows_it_is():
    """Found by mutation-testing the abstract (caution (aq)): `Self-consistency is an instance.`
    could be DELETED with all 492 manuscript guards passing. tests/test_selfconsistency_positioning
    pins Related Work and the appendix -- the places a reviewer checks a positioning claim -- and
    nothing pinned the place the claim is MADE.

    Conditioned on measured data and not on a phrase elsewhere: the judge-free CSV carries an arm
    literally named `majority vote (self-consistency)` whose budget column is exactly log n, the
    same certificate the pointwise-reward arm gets. So it IS an instance, by construction, and the
    abstract has to say so while that row exists."""
    import csv as _csv
    import math as _math
    import os as _os
    rows = list(_csv.DictReader(open(_os.path.join(
        ROOT, "results", "selection_verifiable_comma7b.csv"), encoding="utf-8")))
    sc = [r for r in rows if "self-consistency" in r["arm"]]
    assert sc, "the self-consistency arm is gone; this guard and the abstract need revisiting"
    for r in sc:
        n, k = int(r["n"]), float(r["budget_nats"])
        assert abs(k - _math.log(n)) < 5e-4, \
            f"self-consistency at n={n} no longer carries exactly log n ({k}); it is not an instance"
    import re as _re
    txt = _abstract_text()
    # v9 "Self-consistency is an instance when it samples a safe model"; v10 "self-consistency over
    # a safe model is an instance" -- the claim with its qualifier moved in front of the verb.
    # v11 (2026-09-24): the abstract names majority vote as one of the rules the certificate covers
    # ("whether a reward model, a vote or an adversary picks it") and reports its lift; the
    # attribution to self-consistency, which needs a citation, is made in the introduction, which is
    # now the place the claim is MADE. Both halves are pinned.
    assert "vote" in txt and "majority vote" in txt, "the abstract dropped majority vote as a rule"
    from tests.manuscript import body as _b
    m = _re.search(r"[Ss]elf-consistency(?: [^.;$]{0,60})? is an instance", _b("iclr_intro.tex"))
    assert m and not _re.search(r"\b(?:not|never|no longer)\b", m.group(0)), \
        "the introduction dropped (or negated) the positioning claim its own judge-free arm measures"


def test_the_abstract_keeps_the_vetting_requirement_because_contamination_amplifies():
    """The second mutation nothing caught. `and which must be vetted` could be deleted with the
    suite green -- a CONCESSION, which caution (ag) says a length edit reaches for first, and a
    safety-relevant one: it is the clause that stops a deployer reading log n as an absolute bound
    rather than one relative to the anchor.

    Conditioned on the arm that forces it: over the contaminated anchors, selecting by the
    memorising model's own likelihood raises the rate at which a passage is reproduced by up to 4x.
    While any anchor amplifies, the abstract must say the anchor has to be vetted."""
    import csv as _csv
    import os as _os
    rows = list(_csv.DictReader(open(_os.path.join(
        ROOT, "results", "selector_n256.csv"), encoding="utf-8")))
    amp = [float(r["amplification_vs_n1"]) for r in rows if r["amplification_vs_n1"] not in ("", None)]
    assert amp, "the contaminated-anchor arm has no amplification column any more"
    assert max(amp) > 1.0, \
        "no contaminated anchor amplifies; the vetting requirement may be revisited"
    txt = _abstract_text()
    assert "must be vetted" in txt, (
        f"the abstract dropped the vetting requirement while contamination still amplifies by up "
        f"to {max(amp):.3f}x -- the clause that keeps the certificate RELATIVE to the anchor")
    assert "relative" in txt.lower(), \
        "the abstract no longer says the guarantee is relative to the anchor"
