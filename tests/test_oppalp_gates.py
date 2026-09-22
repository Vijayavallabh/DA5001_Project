"""feat-178 L: the gates, mutation-tested BEFORE the data they gate exist.

Caution (ap): repair -- and validate -- a gate before you look at the result it is gating, so the
fix cannot be tuned to the answer. The ladder was still generating when this was written; nothing
under results/oppalp_* existed. Five deliberate defects, each of which must be caught by name, and
one clean run which must not be.

The helpers are monkeypatched rather than faked on disk: the gates are four boolean expressions
over what those helpers return, and building five run directories to exercise them would test the
filesystem instead.
"""
import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis import score_oppalp as S  # noqa: E402

TAGS = [t for _, t in S.ARMS]
# A clean ladder: strengths spanning 0.24, D3 falling monotonically, nothing crossing zero.
CLEAN = {t: dict(strength=s, d3=d, lo=d - 0.02, hi=d + 0.02, empty=0.0, words=120.0,
                 gen="blocklist_decode", pids={f"p{i}" for i in range(S.N_PROMPTS)})
         for t, s, d in zip(TAGS, [0.52, 0.58, 0.64, 0.70, 0.76],
                            [0.02, 0.005, -0.01, -0.03, -0.05])}


def run(spec, capsys):
    def per_prompt(sfx):
        a = spec[sfx.split("__oppalp_")[1]]
        return [dict(prompt_id=p, u_anchor_k0=str(1 - a["strength"])) for p in sorted(a["pids"])]

    def d3_row(sfx):
        a = spec[sfx.split("__oppalp_")[1]]
        return dict(value=str(a["d3"]), lo95=str(a["lo"]), hi95=str(a["hi"]),
                    reading="REVERSAL UNRESOLVED")

    def shape(run_dir, pids=None):
        a = spec[run_dir.split("output/oppalp_")[1]]
        return len(a["pids"]), a["empty"], a["words"]

    monkey = dict(per_prompt=per_prompt, d3_row=d3_row, shape=shape,
                  pipeline=lambda d: spec[d.split("output/oppalp_")[1]]["gen"])
    old = {k: getattr(S, k) for k in monkey}
    for k, v in monkey.items():
        setattr(S, k, v)
    argv = sys.argv
    # mkdtemp rather than a fixed path: this writes a CSV on the clean runs, and a hardcoded
    # /tmp name is both a collision between concurrent runs and litter on a filesystem this
    # project has already had fill to 100%.
    sys.argv = ["score_oppalp.py", "--out", tempfile.mkdtemp(prefix="oppalp_")]
    try:
        S.main()
    finally:
        sys.argv = argv
        for k, v in old.items():
            setattr(S, k, v)
    return capsys.readouterr().out


def mutate(**kw):
    """One rung changed; everything else the clean ladder. Returns the spec."""
    spec = {t: dict(v, pids=set(v["pids"])) for t, v in CLEAN.items()}
    tag = kw.pop("tag")
    spec[tag].update(kw)
    return spec


def test_the_clean_ladder_passes_every_gate_and_reads_both_bands(capsys):
    out = run({t: dict(v, pids=set(v["pids"])) for t, v in CLEAN.items()}, capsys)
    assert "FAIL" not in out, out
    assert "NOT SCORED" not in out and "NOT TESTED" not in out
    assert "**CONSISTENT**" in out, out          # strength up, D3 down, monotone
    assert "**TASK TYPE SURVIVES**" in out, out  # nothing crosses zero from below


@pytest.mark.parametrize("name,kw,gate", [
    ("a rung from the other generator", dict(tag="qwen3b", gen="h1.py"), "G0"),
    ("a rung on a different prompt set", dict(tag="qwen15b", pids={"x"}), "G1"),
    ("an opponent that emits empties", dict(tag="qwen05b", empty=0.15), "G2"),
    ("an opponent three times as long", dict(tag="qwen14b", words=400.0), "G2"),
])
def test_each_defect_is_caught_by_its_own_gate(name, kw, gate, capsys):
    out = run(mutate(**kw), capsys)
    assert f"{gate} " in out and "FAIL" in out, f"{name}: no gate fired\n{out}"
    assert gate in out.split("GATE(S) FAILED:")[1], f"{name}: {gate} is not the gate that fired"
    assert "NOT SCORED" in out, f"{name}: a band was computed past a failed gate"
    assert "**CONSISTENT**" not in out and "**TASK TYPE" not in out


def test_a_ladder_with_no_range_is_not_tested_rather_than_not_scored(capsys):
    """G3's failure has its own registered reading: an instrument with no range did not test the
    question. Reporting it as NOT SCORED would say the arm was invalid, which it is not."""
    spec = {t: dict(v, pids=set(v["pids"]), strength=0.60 + 0.01 * i)
            for i, (t, v) in enumerate(CLEAN.items())}
    out = run(spec, capsys)
    assert "G3" in out and "FAIL" in out
    assert "**NOT TESTED**" in out, out
    assert "NOT SCORED" not in out, "a rangeless ladder was reported as an invalid arm"


def test_a_crossing_rung_reads_opponent_explains_the_split(capsys):
    """The outcome that would falsify the appendix's task-type sentence must be reachable; a band
    nothing can trigger is caution (p)'s gate that passes everything, in the other direction."""
    spec = mutate(tag="llama8b", d3=0.08, lo=0.04, hi=0.12)
    out = run(spec, capsys)
    assert "**OPPONENT EXPLAINS THE SPLIT**" in out, out
    assert "llama8b" in out.split("cross zero from below")[1]
