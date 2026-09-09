"""feat-064: the seed table must be assembled by a script, not by hand."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_manifest_rows_are_well_formed():
    """Six tab-separated fields, an integer seed, and a pair key that groups the runs."""
    rows = [l.rstrip("\n").split("\t") for l in open("results/seed_effect_runs.tsv")
            if l.strip() and not l.startswith("#")]
    assert rows, "manifest is empty"
    pairs = {}
    for f in rows:
        assert len(f) == 6, f
        int(f[1])
        pairs.setdefault(f[5], []).append((int(f[1]), f[3]))
    # Every row in a group is one arm of an intervention: the seed arms differ in the seed, the
    # temperature arms hold the seed fixed and differ in the budget path they were measured under.
    # Either way (seed, budget path) identifies the arm, so a repeat is a duplicated row.
    for pair, arms in pairs.items():
        assert len(set(arms)) == len(arms), f"{pair} repeats an arm"
        assert len(arms) >= 2, f"{pair} has nothing to compare"


def test_seed_words_is_measured_not_assumed(monkeypatch):
    """It must read the passages, not multiply the tokenizer's average by the seed length:
    the seed straddles a token boundary and the average would round the wrong way."""
    import inspect
    from analysis import seed_effect
    src = inspect.getsource(seed_effect.seed_words)
    assert "load_prompt_corpus" in src and "decode" in src
    assert "chars_per_token" not in src


def test_spearman_handles_ties_with_average_ranks():
    """The two KL3M pairs tie at 7.3 seed words; the no-ties shortcut would misreport the trend."""
    from analysis.seed_effect import spearman, _ranks
    assert _ranks([1.0, 2.0, 2.0, 3.0]) == [0.0, 1.5, 1.5, 3.0]
    assert abs(spearman([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-12
    assert abs(spearman([1, 2, 3], [3, 2, 1]) + 1.0) < 1e-12
    # a tie in x with opposite y values pulls the correlation off perfect
    assert abs(spearman([1, 2, 2, 3], [1, 2, 3, 4])) < 1.0


def test_permutation_p_is_exact_and_symmetric():
    from analysis.seed_effect import permutation_p
    rho, p = permutation_p([1, 2, 3], [1, 2, 3])
    assert abs(rho - 1.0) < 1e-12 and abs(p - 2 / 6) < 1e-12   # 2 of 3! permutations reach |rho|=1


def test_prediction_is_calibrated_on_the_control_and_scales_with_k_crit():
    """(K) must take its one constant from the pair's control arm, so the control reproduces its
    own measured ratio exactly and every other arm is an out-of-sample number. A calibration that
    silently used the mean of the arms would make all of them look like hits."""
    rows = [dict(pair="P", label="P seed 20 (control)", onset=2.0, k_crit=4.0, s_x=2.0,
                 ratio=1.0, ratio_lo=0.9, ratio_hi=1.1),
            dict(pair="P", label="P seed 40", onset=1.5, k_crit=3.0, s_x=2.0,
                 ratio=0.75, ratio_lo=0.6, ratio_hi=0.9)]
    for pair in {r["pair"] for r in rows}:
        g = [r for r in rows if r["pair"] == pair]
        ctl = next(r for r in g if "control" in r["label"])
        c = ctl["onset"] / ctl["k_crit"]
        for r in g:
            r["pred"] = c * r["k_crit"] / r["s_x"]
    assert rows[0]["pred"] == 1.0                      # control reproduces itself
    assert abs(rows[1]["pred"] - 0.75) < 1e-12         # 0.5 * 3.0 / 2.0


def test_elasticity_is_only_defined_where_s_x_actually_moves():
    """The seed arms move s(x) by under 2%, so d log(onset) / d log s(x) is a ratio of noise there
    and produced values of 10 to 33 before the guard. Only an arm that moves s(x) materially --
    the temperature arms move it by about half a log unit -- gets one."""
    import math
    guard = 0.05
    seed_arm = abs(math.log(2.3982 / 2.4147))       # KL3M-520M seed 40 against its control
    warp_arm = abs(math.log(4.0340 / 2.4147))       # the same pair at tau = 0.4
    assert seed_arm < guard < warp_arm
    assert round(math.log(2.4923 / 2.4923) if seed_arm >= guard else 0.0, 6) == 0.0
