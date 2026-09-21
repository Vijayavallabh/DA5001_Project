"""feat-169: |G| does not predict TokenSwap's suppression; the rate the rule fires at does.

The load-bearing claim is NEGATIVE -- the count axis mispredicts a held-out rung by 0.2057 -- and
negative claims are the ones a length edit deletes first (caution (ag)). Every number is rebuilt
from the CSVs, and the shape claims are checked against the data, not the prose (caution (ai)).
"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "results", "onset_prediction_tokenswap_gsize.md")


def _ladder():
    path = os.path.join(ROOT, "results", "tokenswap_gsize.csv")
    return list(csv.DictReader(open(path, encoding="utf-8")))


def _txt():
    t = open(LOG, encoding="utf-8").read()
    return " ".join(t.partition("\n## Scoring, ")[2].split())


def test_the_44_word_rung_suppresses_at_kl3m_s_own_G_size():
    """B2. Both seeds must be inside the committed 0.01 band, and their |G| must bracket KL3M's
    171 -- if a re-run moved either, the factor-of-38 comparison is not the comparison."""
    rows = [r for r in _ladder() if int(r["words"]) == 44]
    assert len(rows) == 2, "the 44-word rung no longer has two seeds"
    for r in rows:
        assert float(r["nv_recall"]) <= 0.01, f"{r['tag']} no longer suppresses: {r['nv_recall']}"
        assert abs(int(r["g_token_ids"]) - 171) <= 5, \
            f"{r['tag']}'s |G| is {r['g_token_ids']}, no longer comparable to KL3M's 171"
    assert "factor" in _txt() and "`38`" in _txt(), "the size of the discrepancy left the log"


def test_the_count_axis_mispredicts_the_held_out_rung_by_a_wide_margin():
    """The strong result. KL3M's |G| = 171 sits INSIDE the ladder, so this is an interpolation and
    not an edge effect -- assert that too, since it is what makes the refutation clean."""
    rows = _ladder()
    ids = [int(r["g_token_ids"]) for r in rows]
    assert min(ids) < 171 < max(ids), \
        "KL3M's |G| is no longer inside the ladder's range; the refutation becomes an extrapolation"
    # Interpolate recall at |G| = 171 the way the scorer does.
    pts = sorted((int(r["g_token_ids"]), float(r["nv_recall"])) for r in rows)
    pred = next(y0 + (171 - x0) / (x1 - x0) * (y1 - y0)
                for (x0, y0), (x1, y1) in zip(pts, pts[1:]) if x0 <= 171 <= x1)
    actual = float([r for r in csv.DictReader(
        open(os.path.join(ROOT, "results", "blocklist_decode__tsleak_kl3m170m.csv"),
             encoding="utf-8")) if r["arm"] == "tokenswap"][0]["nv_recall_mean"])
    assert actual - pred > 0.15, \
        f"the count axis now predicts KL3M within {actual - pred:.4f}; the log's refutation is stale"
    assert "`0.2057`" in _txt(), "the mispredicition size left the log"


def test_the_ladder_is_monotone_and_both_seeds_agree_at_every_rung():
    """B1/B3. Recall must fall as |G| rises, and the two seeds must give the same READING at every
    rung -- that agreement is why two seeds were run and is what rules out a lucky subset."""
    rows = _ladder()
    # Monotonicity is a claim about the RUNGS, not about individual draws: the two seeds at one
    # word count differ only in which words were dropped, and at the 10-word rung -- where the
    # rule is weakest -- they straddle by 0.0732, the widest spread on the ladder. Checking all
    # eleven points pointwise fails on that pair and would be checking the wrong thing.
    means = {}
    for r in rows:
        means.setdefault(int(r["words"]), []).append(float(r["nv_recall"]))
    series = [sum(v) / len(v) for _w, v in sorted(means.items())]
    assert all(b <= a + 1e-9 for a, b in zip(series, series[1:])), \
        f"rung-mean recall no longer falls as G grows: {sorted(means.items())}"
    assert len(series) == 6, f"the ladder no longer has six rungs: {sorted(means)}"
    seeds = {}
    for r in rows:
        seeds.setdefault(int(r["words"]), []).append(r["reading"])
    for w, rs in seeds.items():
        assert len(set(rs)) == 1, f"the two seeds at {w} words now disagree: {rs}"


def test_the_log_concedes_what_it_could_not_test_and_that_the_check_was_post_hoc():
    """Two concessions, both of which a tidy-up would delete: the held-out check was not
    registered, and the ladder does not reach KL3M's mass so the mass axis is untested there."""
    t = _txt()
    assert "not registered" in t, "the log no longer states the held-out check was post hoc"
    assert "cannot test the mass axis" in t, \
        "the log no longer concedes that the ladder does not reach KL3M's mass"
    rows = _ladder()
    lowest = min(float(r["mass_on_g"]) for r in rows)
    kl = [r for r in csv.DictReader(
        open(os.path.join(ROOT, "results", "tokenswap_gates.csv"), encoding="utf-8"))
        if r["arm_dir"] == "leak_kl3m170m" and r["arm"] == "tokenswap"][0]
    assert float(kl["mass_on_g"]) < lowest, \
        "the ladder now reaches KL3M's mass, so that concession is stale and must be revisited"
    assert "half right" in t, "the log no longer records that our own prediction named mass"
