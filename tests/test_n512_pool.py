"""feat-181's pool gates on synthetic trajectory files: G0 must refuse one changed character and one
missing prompt; the merge's G2 must refuse a duplicated and a missing trajectory id."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import n512_pool as P  # noqa: E402


def _file(d, cls, recs):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"trajectories_k0_{cls}.jsonl"), "w", encoding="utf-8") as fh:
        for p, t, g in recs:
            fh.write(json.dumps({"metadata": {"prompt_id": p, "seed": (7 << 16) | t, "trajectory_id": t},
                                 "aggregate": {"generation": g, "per_step": "x" * 50}}) + "\n")


def test_g0_passes_identical_regeneration_and_refuses_one_character_or_one_prompt(tmp_path):
    com, reg = str(tmp_path / "c"), str(tmp_path / "r")
    _file(com, "factual", [(p, t, f"{p}-{t}") for p in ("a", "b") for t in (254, 255)])
    _file(reg, "factual", [(p, 255, f"{p}-255") for p in ("a", "b")])
    assert P.g0(com, reg, ["factual"]) == {"factual": (2, 0)}
    _file(reg, "factual", [("a", 255, "a-255"), ("b", 255, "b-255 ")])
    assert P.g0(com, reg, ["factual"]) == {"factual": (2, 1)}
    _file(reg, "factual", [("a", 255, "a-255")])
    assert P.g0(com, reg, ["factual"]) == {"factual": (2, 1)}


def test_merge_concatenates_in_order_and_g2_refuses_a_duplicate_or_a_gap(tmp_path):
    old, new = str(tmp_path / "old"), str(tmp_path / "new")
    _file(old, "neutral", [(p, t, "g") for p in ("a", "b") for t in (0, 1)])
    _file(new, "neutral", [(p, t, "g") for p in ("a", "b") for t in (2, 3)])
    res = P.merge(str(tmp_path / "m"), ["neutral"], [old, new], n=4)
    assert res == {"neutral": (2, [])}
    lines = open(tmp_path / "m" / "trajectories_k0_neutral.jsonl").read().splitlines()
    assert [json.loads(x)["metadata"]["trajectory_id"] for x in lines] == [0, 1, 0, 1, 2, 3, 2, 3]
    _file(new, "neutral", [("a", 2, "g"), ("a", 3, "g"), ("b", 2, "g"), ("b", 2, "g")])
    assert P.merge(str(tmp_path / "m2"), ["neutral"], [old, new], n=4)["neutral"][1] == ["b"]
    _file(new, "neutral", [("a", 2, "g"), ("a", 3, "g"), ("b", 3, "g")])
    assert P.merge(str(tmp_path / "m3"), ["neutral"], [old, new], n=4)["neutral"][1] == ["b"]
