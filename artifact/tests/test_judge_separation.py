"""Plan v5 item D: the separation the manuscript's headline quotes must come from a script."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.judge_separation import crossing, moments, z


def test_moments_of_the_three_point_score():
    u, var_u, l, var_l = moments(win=40.0, tie=20.0, loss=40.0, n=100)
    assert u == 0.5                                   # 0.4 + 0.5*0.2
    assert math.isclose(var_u, (0.4 + 0.25 * 0.2 - 0.25) / 100)
    assert l == 0.4 and math.isclose(var_l, 0.4 * 0.6 / 100)


def test_z_reproduces_the_published_k1_separation():
    """utility_v4_summary: KL k=1 loses 59.4% of 180, the anchor 61.7% of 180 -> -0.45 sigma."""
    _, _, l_arm, v_arm = moments(28.9, 100 - 28.9 - 59.4, 59.4, 180)
    _, _, l_ref, v_ref = moments(30.5, 100 - 30.5 - 61.7, 61.7, 180)
    assert round(z(l_arm, v_arm, l_ref, v_ref), 2) == -0.45


def test_z_is_none_without_judged_pairs():
    assert z(0.5, 0.0, 0.5, 0.0) is None


def test_crossing_interpolates_inside_its_bracket():
    assert math.isclose(crossing([(1.0, -0.5), (3.0, -2.5)], -2.0), 2.5)


def test_crossing_refuses_to_extrapolate_backwards():
    """Already past the threshold at the smallest budget probed: there is no bracket, so the
    crossover is unknown, not zero. Reporting a number here would invent one."""
    assert crossing([(1.0, -2.5), (3.0, -3.0)], -2.0) is None


def test_crossing_returns_none_when_no_budget_reaches_it():
    assert crossing([(1.0, -0.5), (3.0, -1.2)], -2.0) is None


def test_vacuous_pct_counts_passages_whose_budget_covers_them(tmp_path):
    """Prop. 1: the certificate is vacuous once K = k*T_max reaches S(x)."""
    from analysis.judge_separation import vacuous_pct
    p = tmp_path / "caps.csv"
    p.write_text("S_safe\n100\n200\n300\n400\n")
    f = vacuous_pct(str(p), t_max=200)
    assert f(0.5) == 25.0            # K=100 covers only the first
    assert f(1.0) == 50.0
    assert f(2.0) == 100.0


def test_vacuous_pct_is_none_without_a_caps_file(tmp_path):
    from analysis.judge_separation import vacuous_pct
    assert vacuous_pct(str(tmp_path / "missing.csv"), 200) is None
    assert vacuous_pct("", 200) is None
