"""feat-131: KL3M-1.7B's CLIMBS reading did NOT survive a disjoint draw.

Paired g(64)-g(8) reads +0.0040 [-0.0330, +0.0400] at seeds 52 53 54 against +0.0650
[+0.0270, +0.1030] at 42 43 44 -- a move of 0.061 where the audited anchor's own seed replication
moved 0.0000. The committed consequence applies: the climb to n=64 is established at TinyComma and
Comma-7B and at no other anchor reproducibly.

The instrument lesson is the transferable part, and it is a CAVEAT ON THE PRECEDENT the
pre-registration leaned on: a paired difference is stable where the effect is large relative to its
own interval (TinyComma, ~2.1 half-widths, reproduced exactly) and not where it is marginal
(KL3M-1.7B, ~1.7, did not). Order averaging and within-pass pairing remove the position lottery;
they do not make a marginal reading safe.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import body  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JUDGE_B = "Phi-3.5-mini-instruct"


def _csv(name):
    return list(csv.DictReader(open(os.path.join(ROOT, "results", name), encoding="utf-8")))


def test_the_band_rounds_from_its_csv_and_the_verdict_follows_from_it():
    r = _csv("kl3m_seed_scoring.csv")[0]
    assert r["seeds"] == "52 53 54", r["seeds"]
    assert abs(float(r["gain_diff"]) - 0.004) < 5e-4, r
    assert abs(float(r["lo95"]) - (-0.033)) < 5e-4, r
    assert abs(float(r["hi95"]) - 0.040) < 5e-4, r
    assert float(r["lo95"]) < 0 < float(r["hi95"]), "the interval must contain zero"
    assert r["verdict"] == "DOES NOT REPLICATE", r["verdict"]
    assert r["integrity"] == "PASS" and int(r["n_prompts"]) == 500
    assert abs(float(r["distance"]) - 0.061) < 1e-3, r["distance"]


def test_the_two_seed_sets_really_are_disjoint():
    """The one thing that was supposed to change. If they ever overlap this is not a replication."""
    sys.path.insert(0, ROOT)
    from dap.stats import build_trajectory_seeds
    a = set(build_trajectory_seeds("x", (42, 43, 44), 64))
    b = set(build_trajectory_seeds("x", (52, 53, 54), 64))
    assert len(a) == len(b) == 64
    assert not (a & b), f"{len(a & b)} of 64 seeds collide"


def test_the_precedent_still_holds_at_the_anchor_it_was_taken_from():
    """The caveat is about marginal effects, not about the construction. If TinyComma's paired
    difference ever stops reproducing, the caveat as written is wrong and must be revisited."""
    import random
    from analysis.selection_decoding import boot_mean

    def paired(tag):
        rows = [r for r in _csv(f"selection_scaling_per_prompt{tag}.csv") if JUDGE_B in r["judge"]]
        d = [float(r["u_n64"]) - float(r["u_n8"]) for r in rows]
        g = sum(d) / len(d)
        lo, hi = boot_mean(d, random.Random(20260918))
        return g, (hi - lo) / 2
    g0, h0 = paired("")
    g1, _ = paired("_seed52")
    assert abs(g0 - g1) < 5e-4, (g0, g1, "TinyComma's paired difference no longer reproduces")
    # and it really is the larger effect, which is the whole point of the caveat
    assert g0 / h0 > 2.0, (g0, h0)
    rep = _csv("kl3m_seed_scoring.csv")[0]
    kl_half = (float(rep["hi95"]) - float(rep["lo95"])) / 2
    assert float(rep["orig_gain_diff"]) / kl_half < 2.0, "KL3M's original reading was not marginal"


def test_the_appendix_reports_the_failed_replication_and_the_caveat():
    txt = body("appendix_selection.tex")
    assert "did not survive a fresh draw" in txt, "the failed replication was trimmed"
    assert "$+0.0040$ $[-0.0330, +0.0400]$" in txt, "the replication band was trimmed"
    assert "stable where the effect is large relative to its own interval and not\nwhere it is marginal" in txt \
        or "stable where the effect is large relative to its own interval and not where it is marginal" in txt, \
        "the caveat on the precedent was trimmed"
    assert "We do not pool the two draws" in txt, \
        "the refusal to pool -- a post-hoc rescue the pre-registration excluded -- was trimmed"
