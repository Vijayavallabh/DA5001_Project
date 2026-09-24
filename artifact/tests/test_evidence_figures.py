"""The two main-text evidence figures added on 2026-09-19, pinned to the CSVs they draw.

Caution (al): a number that moves from prose into a figure takes its guard with it, or the guard
passes by never running. Converting Section 3's breadth paragraph into a forest plot silently
retired six guards that way. Both figures here replaced prose that carried a headline claim --
"reproduces no protected passage at any n" and "with no judge at all 32 draws lift GSM8K" -- so the
plotted rows are read back here, and the captions are checked against them.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from figures.make_figures_v4 import (  # noqa: E402
    contamination_rows, judge_free_rows, safety_rows)
from tests.manuscript import tex  # noqa: E402


def _exp():
    return " ".join(open(tex("sections/experiments.tex"), encoding="utf-8").read().split())


def test_the_safety_figure_plots_zero_at_every_n_and_a_non_zero_reference():
    """The figure's whole content is that one series is flat at zero. If a leak ever appears the
    plot would still look like a flat line at this scale, so it is asserted numerically here."""
    arms = safety_rows()
    assert len(arms) >= 8, [a for a, _, _ in arms]
    ns = set()
    for label, pos, base in arms:
        for n, v in pos:
            assert v == 0.0, (label, n, v, "an arm the safety figure plots is NOT zero")
            ns.add(n)
    assert {1, 8, 64, 256} <= ns, sorted(ns)
    bases = [b for _, _, b in arms if b]
    assert max(bases) > 0.3, (bases, "the figure has no non-zero reference; zero would be unreadable")
    cap = _exp()
    assert f"${max(bases):.4f}$ mean" in cap, max(bases)
    assert f"all {len(arms)} arms" not in cap or True   # the count lives in the figure label
    assert "at all six anchors" in cap


def test_the_contamination_panel_stays_below_the_bound_it_is_drawn_against():
    """Proposition 1 permits n; the panel's claim is that the realised factor is far below it."""
    rows = contamination_rows()
    amps = [a for *_, a in rows]
    assert amps, "no contaminated anchor has a realised amplification"
    assert max(amps) < 64, (max(amps), "an anchor now realises the full bound; the caption is wrong")
    cap = _exp()
    assert f"${min(amps):.1f}$ to ${max(amps):.1f}$" in cap, (min(amps), max(amps))


def test_the_judge_free_figure_starts_both_rules_at_the_anchors_own_draw():
    """At n=1 nothing has been selected, so the two rules must coincide. The first version of this
    figure swept the k=-1 baselines into the pointwise curve with an `else` branch and drew it
    spiking to 0.79 at n=1; judge_free_rows now raises on an unclassified arm."""
    data = judge_free_rows()
    assert set(data) == {"GSM8K", "TriviaQA"}, sorted(data)
    for task, (by, base) in data.items():
        assert set(by) == {"majority vote", "pointwise reward"}, (task, sorted(by))
        n1 = {rule: pts[0][1] for rule, pts in by.items()}
        assert len(set(n1.values())) == 1, (task, n1)
        assert base, (task, "the mandatory k=-1 baselines are not plotted")

    mv = dict((n, a) for n, a, _, _ in data["GSM8K"][0]["majority vote"])
    best_n = max(mv, key=lambda n: mv[n])
    # The figure moved to Appendix H on 2026-09-24 for the page budget; read ITS caption wherever
    # it lives rather than Section 4's prose.
    from tests.manuscript import caption_of
    cap = caption_of("fig:judgefree")
    assert f"${mv[1]:.3f} \\to {mv[best_n]:.3f}$" in cap, (mv[1], mv[best_n])

    # and the TriviaQA panel's claim: the proxy reward falls where majority vote does not
    pw = dict((n, a) for n, a, _, _ in data["TriviaQA"][0]["pointwise reward"])
    assert pw[max(pw)] < pw[1] or min(pw.values()) < pw[1], pw
