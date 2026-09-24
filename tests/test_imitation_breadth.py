"""Proposition 3's measured shape, at every pair it has been measured at.

The proposition is proved and no measurement can refute it. What these pin is the SHAPE the paper
reads off `imitation_cost.csv` -- an imitation rate that saturates in k, a realised spend that
converges on it, a cumulative spend linear in the step index, and a decoder that ends up serving
p_r almost everywhere -- and that the shape is not one pair's. Bands are
`results/onset_prediction_imitation_breadth.md`, committed before either Llama-3.2 pair was
generated, and `results/onset_prediction_imitation_70b.md` for the 70B pair.
"""
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def arms():
    """tag -> {k: row} for the ordinary-prompt rows of every imitation_cost CSV on disk."""
    out = {}
    for path in glob.glob(os.path.join(ROOT, "results", "imitation_cost*.csv")):
        base = os.path.basename(path)
        if "lorenz" in base:
            continue
        tag = base[len("imitation_cost"):-len(".csv")] or "_audited"
        with open(path, encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r["prompt_class"] == "ordinary"]
        if rows:
            out[tag] = {r["k"]: r for r in rows}
    return out


def test_more_than_one_pair_is_measured():
    """The whole arm. If this ever drops back to one, the paper's Proposition 3 paragraph is a
    single setup again and has to say so."""
    a = arms()
    assert "_audited" in a, sorted(a)
    assert len(a) >= 2, sorted(a)


def test_the_rate_saturates_at_every_pair():
    """I1: r_imit(20)/r_imit(3) within 1.10."""
    for tag, ks in arms().items():
        if "3" not in ks or "20" not in ks:
            continue
        r3 = float(ks["3"]["imitation_rate_nats_per_token"])
        r20 = float(ks["20"]["imitation_rate_nats_per_token"])
        assert 0 < r3 <= r20 * 1.001, (tag, r3, r20)   # monotone up to float noise
        assert r20 / r3 <= 1.10, (tag, r20 / r3)


def test_the_spend_is_the_imitation_cost_at_every_pair():
    """I2: |r_real/r_imit - 1| < 0.05 at k=3 and k=20."""
    for tag, ks in arms().items():
        for k in ("3", "20"):
            if k not in ks:
                continue
            imit = float(ks[k]["imitation_rate_nats_per_token"])
            real = float(ks[k]["realised_rate_nats_per_token"])
            assert abs(real / imit - 1) < 0.05, (tag, k, imit, real)


def test_the_spend_is_linear_in_the_step_index_at_every_pair():
    """I3: median within-trajectory R^2 >= 0.95 at every budget."""
    for tag, ks in arms().items():
        for k, r in ks.items():
            assert float(r["median_cum_spend_vs_step_r2"]) >= 0.95, (tag, k, r)


def test_the_decoder_ends_as_the_risky_model_at_every_pair():
    """I4: p_r served unchanged at >= 99% of steps at the largest budget on the grid."""
    for tag, ks in arms().items():
        k = max(ks, key=float)
        r = ks[k]
        unchanged = 1 - float(r["forced_safe_frac"]) - float(r["active_frac"])
        assert unchanged >= 0.99, (tag, k, unchanged)


def test_the_rate_differs_between_pairs_because_the_pairs_differ():
    """I5's point: the rate measures the distance between the two models, so two pairs must not
    give the same number. If they ever do, the reading of imitation_cost.csv as a distance is what
    has to change, not this test."""
    a = arms()
    r3 = {t: float(k["3"]["imitation_rate_nats_per_token"]) for t, k in a.items() if "3" in k}
    assert len(r3) >= 2, r3
    assert max(r3.values()) - min(r3.values()) > 0.05, r3


# The row of tab:imitation-pairs each CSV feeds (v10 moved the table to Appendix G, appendix_onset).
ROW = {"_audited": "TinyComma-1.8B, 8B-Inst.", "_llama70b": "TinyComma-1.8B, 70B base",
       "_llama321b": "Llama-3.2-1B, 8B-Inst.", "_llama323bi": "Llama-3.2-3B-Inst., 8B-Inst."}


def _pairs_table():
    """label -> cells of tab:imitation-pairs, whitespace-normalised."""
    from tests.manuscript import body
    apx = body("appendix_onset.tex")
    i = apx.index("\\label{tab:imitation-pairs}")
    tab = apx[apx.index("\\midrule", i) + len("\\midrule"):apx.index("\\bottomrule", i)]
    rows = [[c.strip() for c in r.split("&")] for r in tab.split("\\\\") if r.strip()]
    return {r[0]: r[1:] for r in rows}


def test_the_appendix_table_rounds_from_the_csvs():
    """Caution (j): every cell of the four-pair table comes from its own CSV, once -- and, since
    v10 moved it from appendix_proofs to appendix_onset, in the row and column it belongs to."""
    a = arms()
    assert len(a) == 4, sorted(a)
    assert set(a) == set(ROW), sorted(a)
    tab = _pairs_table()
    for tag, ks in a.items():
        cells = tab.get(ROW[tag])
        assert cells, (tag, "its row is gone from tab:imitation-pairs")
        r3, r20 = ks["3"], ks["20"]
        i3 = float(r3["imitation_rate_nats_per_token"])
        i20 = float(r20["imitation_rate_nats_per_token"])
        assert cells[0] == f"${i3:.4f}$", (tag, i3, cells)
        assert cells[1] == f"${i20:.4f}$", (tag, i20, cells)
        assert cells[2] == f"${i20 / i3:.3f}$", (tag, i20 / i3, cells)
        dev = abs(float(r3["realised_rate_nats_per_token"]) / i3 - 1)
        assert cells[3] == f"${dev:.4f}$", (tag, dev, cells)
        assert cells[4].replace("\\ ", "") == f"${float(r20['spend_nats']):.1f}$", (tag, r20["spend_nats"], cells)


def test_section2_quotes_the_further_saturated_rates():
    """v10 (2026-09-24): Section 3 quotes only the audited pair's saturated rate and the further
    pairs moved to Appendix G, whose prose states the pair count and whose table carries each
    further pair's saturated rate r_imit(20) in its own row."""
    from tests.manuscript import body
    main = body("frontier.tex")
    # the quoted rate is the SATURATED one, r_imit(20), which is what "the imitation rate of 0.857"
    # means for the audited pair
    aud = float(arms()["_audited"]["20"]["imitation_rate_nats_per_token"])
    assert f"${aud:.3f}$" in main, aud
    a = arms()
    words = {3: "three", 4: "four", 5: "five"}
    apx = body("appendix_onset.tex")
    assert f"repeats the measurement at {words[len(a)]} pairs" in apx, \
        "Appendix G no longer says how many pairs the shape was measured at"
    tab = _pairs_table()
    for t, ks in a.items():
        if t == "_audited":
            continue
        v = float(ks["20"]["imitation_rate_nats_per_token"])
        assert tab[ROW[t]][1] == f"${v:.4f}$", (t, v)


def test_the_narrow_i5_pass_is_reported_as_narrow():
    """0.3010 against a threshold of 0.30785 is a pass by 2.2% of the threshold. A scoring log that
    presented that as comfortable would be overclaiming."""
    a = arms()
    r = {t: float(ks["3"]["imitation_rate_nats_per_token"]) for t, ks in a.items()}
    # the I5 band was Llama-3.2-3B-Instruct against half of Llama-3.2-1B, named rather than
    # picked by sorting: a third non-audited pair (the 70B) now exists and would change the pick
    lo, mid = r["_llama323bi"], r["_llama321b"]
    assert lo < mid / 2, (lo, mid / 2)
    assert (mid / 2 - lo) / (mid / 2) < 0.05, "no longer narrow; the scoring log must be updated"
    log = open(os.path.join(ROOT, "results",
                            "onset_prediction_imitation_breadth.md"), encoding="utf-8").read()
    assert "narrow" in log.lower()
    # v10 moved the note from appendix_proofs to appendix_onset; it must also carry the margin
    from tests.manuscript import body
    apx = body("appendix_onset.tex")
    i = apx.find("passed narrowly")
    assert i != -1, "the narrow I5 pass is no longer reported as narrow"
    margin = (mid / 2 - lo) / (mid / 2) * 100
    assert f"by ${margin:.1f}\\%$ of its threshold" in apx[i:i + 80], (margin, apx[i:i + 80])
