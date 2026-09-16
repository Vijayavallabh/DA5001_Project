"""The corpus-vs-memoriser arm, and the three things that would make it answer the wrong question.

(1) Strength must be MEASURED, not read off the epoch count -- the knob is not assumed monotone.
(2) The ladder must share one anchor, one corpus and one grid, or the ratios are not comparable.
(3) The bands in the scorer must be the bands in the pre-registration, to the digit (caution (j)).
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREREG = os.path.join(ROOT, "results/onset_prediction_strength.md")


def _head():
    return open(PREREG, encoding="utf-8").read().partition("\n## Scoring log")[0]


def test_the_scorer_correlates_against_measured_strength_not_the_epoch_count():
    src = open(os.path.join(ROOT, "analysis/strength_ladder.py"), encoding="utf-8").read()
    body = src.partition("def main(")[2]
    assert "strengths = [r[\"k_minus1_recall\"] for r in ok]" in body, \
        "the ladder must correlate against the measured k=-1 arm"
    # the epoch count appears only as a LABEL; it must never be an input to rho
    assert "epochs" not in body.partition("rho, p = ")[0].rpartition("strengths =")[2]


def test_the_committed_bands_are_the_scorer_bands():
    from analysis.strength_ladder import (ENTRY_GATE, RHO_EXPLAINS, SPAN_EXPLAINS,
                                          SPAN_REFUTES, STRENGTH_SPAN_MIN)
    head = _head()
    # the prereg writes these to two decimals; a bare repr of 0.10 is "0.1" and would never match,
    # which is the same trap as quoting a CSV's float straight into prose (caution (j))
    assert f"`rho <= {RHO_EXPLAINS:g}`" in head, RHO_EXPLAINS
    assert f"span `>= {SPAN_EXPLAINS:.2f}`" in head, SPAN_EXPLAINS
    assert f"span `< {SPAN_REFUTES:.2f}`" in head, SPAN_REFUTES
    assert f"less than `{STRENGTH_SPAN_MIN:g}x`" in head, STRENGTH_SPAN_MIN
    assert f"at least `{ENTRY_GATE:.2f}`" in head, ENTRY_GATE


def test_the_ladder_shares_one_anchor_one_corpus_and_one_grid():
    sh = open(os.path.join(ROOT, "scripts/run_strength_ladder.sh"), encoding="utf-8").read()
    # every flag except --epochs is fixed in the script, so the ladder cannot vary two things
    code = "\n".join(l for l in sh.splitlines() if not l.lstrip().startswith("#"))
    assert code.count("--epochs") == 1 and '--epochs "$ep"' in code
    for fixed in ("--rank 128", "--lr 3e-4", "--stop-loss 0.02", "--target-modules all-linear"):
        assert fixed in sh, fixed
    # Every pair's grid must carry both mandatory baselines, and must be the grid ITS OWN corner
    # run was measured on: Pleias needed feat-120's licensed extension (43.1% no-crossing), KL3M
    # did not (0.0%) and must not silently acquire it, or its ladder and its corner disagree.
    grids = {g[0]: g[1].split() for g in re.findall(r'(\w+)\)\s+BASE=\S+;\s+PTAG=\w+\s+'
                                                    r'GRID="([^"]+)"', sh)}
    if not grids:      # tolerate reformatting: fall back to every GRID= line, keyed by order
        grids = {str(i): g.split() for i, g in enumerate(re.findall(r'GRID="([^"]+)"', sh))}
    assert len(grids) == 2, sorted(grids)
    for k, g in grids.items():
        assert g[:2] == ["-1", "0"], (k, "both mandatory baselines")
    ext = {"4.6", "5.3", "6.6"}
    has = [set(g) >= ext for g in grids.values()]
    assert sorted(has) == [False, True], \
        "exactly one pair (Pleias) should carry the licensed extension points"
    # one card by default; the GPU override exists only for a window the user opened explicitly
    assert 'CUDA_VISIBLE_DEVICES="${GPU:-2}"' in sh and "CUDA_DEVICE_ORDER=PCI_BUS_ID" in sh
    assert "the standing rule in AGENTS.md is one card" in sh


def test_the_forty_epoch_point_is_feat_120s_own_merged_run():
    from analysis.strength_ladder import POINTS, S_X
    labels = [l for l, _ in POINTS]
    assert len(POINTS) == 4 and any("feat-120" in l for l in labels)
    d = dict((l, r) for l, r in POINTS)
    assert d["epochs=40 (feat-120)"] == "output/phase5/fineb_pleias12b_full", \
        "the 40-epoch point must use the MERGED grid, or it sits at a ceiling the others do not"
    # the anchor never changes, so s(x) is one number and the ratios share a denominator
    import csv
    for r in csv.DictReader(open(os.path.join(ROOT, "results/onset_theory_bookmia.csv"))):
        if r["pair"].startswith("Pleias-1.2B"):
            assert abs(float(r["s_safe_median"]) - S_X) < 1e-9, (r["s_safe_median"], S_X)
            return
    pytest.fail("Pleias row missing from results/onset_theory_bookmia.csv")


def test_no_outcome_is_allowed_to_restore_the_retracted_sentence():
    """The one thing a 'strength explains it' result would be spun into. Committed against."""
    head = _head()
    assert "No outcome of this arm restores the retracted sentence" in head
    src = open(os.path.join(ROOT, "analysis/strength_ladder.py"), encoding="utf-8").read()
    assert "does NOT restore" in src and "inherits the confound" in src


def test_the_two_ladders_are_orthogonal_and_share_one_corner():
    """The seed arm only means something if it differs from the epoch arm in exactly one flag and
    lands on the same grid, same corpus, same passages. Both also share feat-120's run as their
    corner (epochs=40, seed=0), which is what makes their two spans comparable."""
    sh = open(os.path.join(ROOT, "scripts/run_strength_ladder.sh"), encoding="utf-8").read()
    code = "\n".join(l for l in sh.splitlines() if not l.lstrip().startswith("#"))
    assert 'AXIS="${1:?' in code and "epochs|seeds" in code
    # the epochs axis pins the seed, the seeds axis pins the epochs -- neither varies two things
    assert 'ep="$v"; sd=0' in code and 'ep=40; sd="$v"' in code
    # one fine-tune call and one sweep call, shared by both axes
    assert code.count("finetune_memorizing.py") == 1 and code.count("composition_attack.py") == 1
    assert code.count("--seed") == 1 and code.count("--epochs") == 1
    # the corner: the seed axis's fixed epoch count is the one feat-120 and every published
    # memoriser in the paper used, not one chosen after seeing feat-121's points
    import json
    r = json.load(open(os.path.join(ROOT, "output/phase5/memb_Pleias-1_2b/recipe.json")))
    assert r["epochs"] == 40 and r.get("seed", 0) == 0


def test_the_seed_arm_bands_are_committed_and_say_what_they_license():
    p = os.path.join(ROOT, "results/onset_prediction_seedspread.md")
    head = open(p, encoding="utf-8").read().partition("\n## Scoring log")[0]
    assert "seed-only span **`>= 0.20`**" in head and "seed-only span **`< 0.10`**" in head
    assert "0.4721" in head, "the band must be read against feat-121's measured epoch-only span"
    # the secondary is committed in advance, not discovered afterwards
    assert "Pooling the two ladders" in head and "1/2520" in head
    assert "No outcome of this arm restores the retracted sentence" in head


def test_both_axes_end_on_the_same_corner_run():
    """The two spans are only comparable because both ladders contain feat-120's run. If the axes
    ever pointed at different corner directories the numbers would be two unrelated spreads."""
    from analysis.strength_ladder import AXES, CORNER
    assert set(AXES) == {"epochs", "seeds"}
    for ax, pts in AXES.items():
        assert len(pts) == 4, ax
        assert CORNER[0] in [r for _, r in pts], f"{ax} does not contain the corner"
    # and the corner is the merged grid, so it is not sitting at a ceiling the others are not
    assert CORNER[0].endswith("_full")
    # the three non-corner runs of each axis are disjoint between axes
    e = {r for _, r in AXES["epochs"] if r != CORNER[0]}
    s = {r for _, r in AXES["seeds"] if r != CORNER[0]}
    assert e and s and not (e & s)


def test_the_seed_bands_in_the_scorer_match_the_committed_ones():
    from analysis.strength_ladder import (SEED_SPAN_NOISE, SEED_SPAN_STRENGTH,
                                          EPOCH_SPAN_MEASURED)
    head = open(os.path.join(ROOT, "results/onset_prediction_seedspread.md"),
                encoding="utf-8").read().partition("\n## Scoring log")[0]
    assert f"`>= {SEED_SPAN_NOISE:.2f}`" in head, SEED_SPAN_NOISE
    assert f"`< {SEED_SPAN_STRENGTH:.2f}`" in head, SEED_SPAN_STRENGTH
    assert f"{EPOCH_SPAN_MEASURED}" in head, "the band must quote feat-121's measured span"
    # Caution (j): one source, checked mechanically. The paper quotes each point's ratio WITH its
    # bootstrap interval, and only onset_ci.csv carries both, so the span must come from there too.
    # strength_ladder.csv interpolates the summary curve instead and reads 0.4714; the two differ by
    # 0.0007 (0.15%), which is documented in the scorer's docstring and bounded by the second
    # assertion below. Mixing them is how a number ends up one off in its last digit.
    import csv
    ci = {r["pair"]: r for r in csv.DictReader(open(os.path.join(ROOT, "results/onset_ci.csv")))
          if r["mode"] == "single" and "(BookMIA, epochs=" in r["pair"]}
    ci["corner"] = next(r for r in csv.DictReader(open(os.path.join(ROOT, "results/onset_ci.csv")))
                        if r["pair"] == "Pleias-1.2B (BookMIA, extended grid)")
    v = [float(r["ratio_point"]) for r in ci.values()]
    assert len(v) == 4, sorted(ci)
    assert abs((max(v) - min(v)) - EPOCH_SPAN_MEASURED) < 5e-5, (max(v) - min(v), EPOCH_SPAN_MEASURED)

    sl = [float(r["ratio"]) for r in
          csv.DictReader(open(os.path.join(ROOT, "results/strength_ladder.csv"))) if r["ratio"]]
    assert abs((max(sl) - min(sl)) - EPOCH_SPAN_MEASURED) < 1e-3, \
        "the two crossing conventions have drifted apart by more than 0.001"


def test_the_second_pair_uses_its_own_corner_and_its_own_s_x():
    """KL3M's ladder must anchor on KL3M's own BookMIA run and divide by KL3M's own s(x). Using
    Pleias' s_x would rescale every ratio and the two pairs would not be comparable at all."""
    import csv
    from analysis.strength_ladder import PAIRS, KL3M_CORNER, KL3M_S_X, S_X, CORNER
    assert set(PAIRS) == {"pleias", "kl3m", "kl3m_cb", "pleias_cb"}
    assert PAIRS["kl3m"]["axes"].keys() == {"seeds"}, "kl3m has no epoch ladder and must not claim one"
    assert KL3M_CORNER in [r for _, r in PAIRS["kl3m"]["axes"]["seeds"]]
    assert KL3M_CORNER != CORNER[0] and abs(KL3M_S_X - S_X) > 0.5
    # both s(x) values round from the theory CSV, once (caution (j))
    want = {"KL3M-520M": KL3M_S_X, "Pleias-1.2B": S_X}
    seen = 0
    for r in csv.DictReader(open(os.path.join(ROOT, "results/onset_theory_bookmia.csv"))):
        for k, v in want.items():
            if r["pair"].startswith(k):
                assert abs(float(r["s_safe_median"]) - v) < 1e-9, (k, r["s_safe_median"], v)
                seen += 1
    assert seen == 2
    # KL3M's corner grid carries NO licensed extension: its no-crossing there was 0.0%
    ks = [r["k"] for r in csv.DictReader(open(os.path.join(ROOT, KL3M_CORNER,
                                                           "composition_summary.csv")))
          if r["mode"] == "single" and r["L"] == "0"]
    assert "4.6" not in ks and "-1.0" in ks and "0.0" in ks, ks


def test_the_extra_seeds_cannot_enter_the_committed_primary():
    """Seeds 3 and 4 were added mid-arm under a recorded amendment. A span grows with the number of
    draws, so letting them into the cross-pair comparison would favour the very conclusion the arm
    tests. The primary is pinned to the three seeds BOTH pairs have."""
    from analysis.strength_ladder import PAIRS, PRIMARY_SEEDS, seed_of
    kl3m = [lab for lab, _ in PAIRS["kl3m"]["axes"]["seeds"]]
    pleias = [lab for lab, _ in PAIRS["pleias"]["axes"]["seeds"]]
    cb = [lab for lab, _ in PAIRS["kl3m_cb"]["axes"]["seeds"]]
    assert len(kl3m) == 5 and len(pleias) == 4 and len(cb) == 3, (kl3m, pleias, cb)
    # the primary set exists in full on EVERY pair -- that is what makes it like-for-like, and it
    # is keyed on the seed NUMBER so a pair's label wording cannot quietly exclude it
    assert PRIMARY_SEEDS == (0, 1, 2)
    for labs in (kl3m, pleias, cb):
        assert set(PRIMARY_SEEDS) <= {seed_of(l) for l in labs}, (PRIMARY_SEEDS, labs)
    src = open(os.path.join(ROOT, "analysis/strength_ladder.py"), encoding="utf-8").read()
    assert "never in place of it" in src, "the secondary must be labelled as a secondary"
    # and the amendment is on record, below the scoring log where amendments belong
    txt = open(os.path.join(ROOT, "results/onset_prediction_seedspread2.md"), encoding="utf-8").read()
    head, _, tail = txt.partition("\n## Scoring log")
    assert "seed" in tail and "Amendment" in tail, "the amendment must be BELOW the scoring log"
    assert "Amendment" not in head, "nothing above the scoring log may be edited"


def test_the_copybench_pair_reproduces_the_nine_pair_table_row():
    """This arm's whole point is that it is IN Section 4's table. If its corner did not reproduce
    that table's own ratio, the ladder would be measuring something else."""
    import csv
    from analysis.strength_ladder import PAIRS, CB_CORNER, CB_S_X
    assert CB_CORNER in [r for _, r in PAIRS["kl3m_cb"]["axes"]["seeds"]]
    row = next(r for r in csv.DictReader(open(os.path.join(ROOT, "results/onset_ci.csv")))
               if r["pair"] == "KL3M-520M + mem. KL3M-520M" and r["mode"] == "single")
    assert abs(float(row["s_x"]) - CB_S_X) < 1e-9, (row["s_x"], CB_S_X)
    # the grid in the runner is the table's own, not one we picked
    sh = open(os.path.join(ROOT, "scripts/run_copybench_seeds.sh"), encoding="utf-8").read()
    # the grid lives inside the kl3m520m branch of the case and is indented, so anchor on the
    # branch rather than the line start -- and take THAT branch's grid, not whichever comes first
    branch = sh.partition("kl3m520m)")[2].partition(";;")[0]
    grid = re.search(r'GRID="([^"]+)"', branch).group(1).split()
    assert grid[:2] == ["-1", "0"], "both mandatory baselines"
    # compare as numbers: the CSV stores budgets through %g, so it writes "2" where the CLI
    # string says "2.0". composition_attack.py parses --k-values to float and never puts the raw
    # string in a path (unlike h1.py -- caution (o)), so the two spellings are the same grid.
    assert [float(g) for g in grid[2:]] == [float(g) for g in row["k_grid"].split()], \
        (grid[2:], row["k_grid"])
    # and the fine-tune flags match the table memoriser's recipe.json, field by field
    import json
    r = json.load(open(os.path.join(ROOT, "output/phase5/mem_kl3m-002-520m/recipe.json")))
    assert r["splits"] == ["attack_train", "val"] and "--splits attack_train val" in sh
    # Compare the VALUES, not their spelling: the script writes "3e-4" where recipe.json records
    # 0.0003. Both are the same number, and pinning the spelling would fail on a cosmetic edit
    # while passing on a real change of value -- the wrong way round.
    # batch/accum/stop-loss are per-pair shell variables now, so read them from the branch; the
    # flags that are still literal in the command are checked against the command.
    for name, val in (("BATCH", r["batch"]), ("ACCUM", r["accum"]), ("STOP", r["stop_loss"])):
        m = re.search(rf"{name}=(\S+?)[;\s]", branch)
        assert m, (name, branch[:200])
        assert float(m.group(1)) == float(val), (name, m.group(1), val)
    for flag, val in (("--rank", r["rank"]), ("--lr", r["lr"]), ("--epochs", r["epochs"])):
        m = re.search(rf"{re.escape(flag)}\s+(\S+)", sh)
        assert m, flag
        assert float(m.group(1)) == float(val), (flag, m.group(1), val)
    # and the sweep's safe model is the MATERIALISED anchor, not the fine-tune's base: that
    # confusion cost two aborted sweeps (kl3m-002-520m ships only pytorch_model.bin)
    assert "SAFE=output/phase5/anchor_kl3m-002-520m" in branch
    assert '--safe-model "$SAFE"' in sh and '--base "$BASE"' in sh
    assert r["base"] in sh and r["target_modules"] in sh and r["no_chat"] is True and "--no-chat" in sh
