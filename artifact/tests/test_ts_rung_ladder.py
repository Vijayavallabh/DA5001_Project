"""feat-167: the auxiliary ladder's two findings, each derived from the data that found it.

Both are concessions -- B2 concedes that TokenSwap's suppression can fail, and the headline
correction concedes that feat-165's INCUMBENT WINS holds only at an auxiliary 22x larger than the
method's own. Caution (ag) says a length edit deletes concessions first and caution (aq) found six
of eight surviving deletion, so these are pinned to their CSVs rather than to their phrasing.
"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "results", "onset_prediction_tokenswap_aux.md")


def _norm(p):
    return " ".join(open(p, encoding="utf-8").read().split())


def _gates():
    path = os.path.join(ROOT, "results", "tokenswap_gates.csv")
    return {(r["arm_dir"], r["arm"]): r for r in csv.DictReader(open(path, encoding="utf-8"))}


def _leak(tag):
    path = os.path.join(ROOT, "results", f"blocklist_decode__tsleak_{tag}.csv")
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["arm"] == "tokenswap"]
    assert len(rows) == 1, path
    return rows[0]


def test_exactly_one_rung_fails_to_suppress_and_it_is_named():
    """B2's band is 0.05 recall. If a second rung ever crosses it, or KL3M stops crossing it, the
    scoring log's whole account of the mechanism is wrong and this must fail rather than let the
    prose stand."""
    recalls = {t: float(_leak(t)["nv_recall_mean"]) for t in
               ("distilgpt2", "kl3m170m", "pleias350m")}
    over = sorted(t for t, v in recalls.items() if v > 0.05)
    assert over == ["kl3m170m"], f"B2's failing rung is no longer KL3M alone: {recalls}"
    assert recalls["kl3m170m"] > 0.05 < 0.3761, "KL3M must sit between suppression and no rule"
    txt = _norm(LOG)
    assert "`0.2113`" in txt and "`23`/`100`" in txt, "the failing rung's numbers left the log"


def test_the_failing_rung_is_the_one_whose_rule_barely_binds():
    """The claim is causal -- suppression fails because G does not survive the tokenizer, not
    because the auxiliary is small. Guard the ORDERING that makes that claim legible: the rung
    that leaks is the rung with the fewest G ids and the lowest bind rate, and it is NOT the
    smallest auxiliary. Caution (ai): the claim about the numbers is what nothing checks."""
    g = _gates()
    binds = {t: float(g[(f"leak_{t}", "tokenswap")]["bind_rate"]) for t in
             ("distilgpt2", "kl3m170m", "pleias350m")}
    ids = {t: int(g[(f"leak_{t}", "tokenswap")]["g_token_ids"]) for t in binds}
    leaker = max(binds, key=lambda t: float(_leak(t)["nv_recall_mean"]))
    assert leaker == min(binds, key=binds.get), "the leaking rung is not the least-binding one"
    assert leaker == min(ids, key=ids.get), "the leaking rung is not the one with the fewest G ids"
    assert binds[leaker] * 10 < min(v for t, v in binds.items() if t != leaker), \
        "the leaking rung's bind rate is no longer an order of magnitude below the others"
    # Size does not order it: the leaker sits between two clean rungs.
    sizes = {"distilgpt2": 82, "kl3m170m": 170, "pleias350m": 350}
    assert min(sizes.values()) < sizes[leaker] < max(sizes.values()), \
        "the leaking rung is now an extreme of the size range, so the log's 'not size' claim fails"


def test_our_own_activity_gate_certified_the_leaking_rung():
    """G1 required the rule to bind on more than 1% of steps and the leaking rung binds on 1.79%.
    That is the defect the scoring log records; if the rung ever falls below 1% the gate would have
    caught it and the log's account of G1 has to change."""
    b = float(_gates()[("leak_kl3m170m", "tokenswap")]["bind_rate"])
    assert b > 0.01, f"G1 would now have failed this rung ({b:.4f}); the recorded defect is stale"
    txt = _norm(LOG)
    assert "certified" in txt, "the log no longer states that G1 certified the rung"


def test_the_headline_correction_survives_a_length_edit():
    """feat-165's INCUMBENT WINS is only true at TinyComma. The three paired readings and the
    word TIE must stay in the log, and the sign must agree with the CSV."""
    txt = _norm(LOG)
    for tag, val in (("ts_distilgpt2", -0.0325), ("ts_pleias350m", -0.0035)):
        path = os.path.join(ROOT, "results", f"order_averaged_h2h__{tag}.csv")
        rows = [r for r in csv.reader(open(path, encoding="utf-8"))]
        d5 = [r for r in rows if r and r[0].startswith("D5")]
        assert len(d5) == 1, path
        got, lo, hi = float(d5[0][2]), float(d5[0][3]), float(d5[0][4])
        assert abs(got - val) < 5e-5, f"{tag}: D5 moved to {got}, the log says {val}"
        assert lo < 0 < hi or hi <= 0, f"{tag}: D5 no longer straddles or sits below zero"
        assert d5[0][7].strip() == "TIE", f"{tag}: the verdict is now {d5[0][7].strip()!r}, not TIE"
        assert f"`{val:+.4f}`" in txt or f"{val:+.4f}" in txt, f"{tag}'s reading left the log"
    assert "`+0.0615 [+0.0285, +0.0950]`" in txt, "feat-165's own reading left the log"
    assert "22x" in txt.replace("$", "").replace("\\", ""), \
        "the log no longer says how much larger feat-165's auxiliary is"
