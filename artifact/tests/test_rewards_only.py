"""--rewards-only must stop before ANY judged number exists.

An arm whose registration says "no n>64 number is read until the gate clears" can only keep that
promise by not looking at numbers already on disk -- and caution (ap) records exactly that order
being broken, the cross-host FAIL read before its threshold was questioned. The flag makes the
order enforceable instead of promised, so what is guarded here is that it really does stop: an
early `return` that someone later moves below the judging loop would restore the old hazard while
leaving the flag, the help text and the comment all in place.
"""
import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "analysis", "selection_scaling.py")


def _main_fn():
    tree = ast.parse(open(SRC, encoding="utf-8").read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    return fn


def test_the_flag_exists_and_is_a_store_true():
    src = open(SRC, encoding="utf-8").read()
    assert '"--rewards-only", action="store_true"' in src


def test_it_returns_before_anything_judged_is_computed():
    fn = _main_fn()
    # the line of the `return` inside `if a.rewards_only:`
    guard = [n for n in ast.walk(fn)
             if isinstance(n, ast.If) and "rewards_only" in ast.dump(n.test)]
    assert len(guard) == 1, "expected exactly one `if a.rewards_only:` in main()"
    assert any(isinstance(b, ast.Return) for b in guard[0].body), \
        "the rewards-only branch no longer returns"
    stop = guard[0].lineno

    # every reference to the judging machinery INSIDE main() must come after it. The scan starts
    # at main()'s own first line: a module-level `from analysis.utility import judge_batch` is a
    # name being bound, not a number being computed, and flagging it would make this guard cry
    # wolf on a correct file.
    src_lines = open(SRC, encoding="utf-8").read().splitlines()
    for i, line in enumerate(src_lines, 1):
        if i < fn.lineno or i >= stop or line.lstrip().startswith("#"):
            continue
        for tok in ("judge_batch", "a.judges", "picks", "per_judge"):
            assert tok not in line, \
                f"{tok!r} is computed at line {i}, BEFORE the rewards-only return at {stop}"


def test_the_reward_cache_is_written_before_the_return():
    src_lines = open(SRC, encoding="utf-8").read().splitlines()
    write = next(i for i, l in enumerate(src_lines, 1) if "wrote {a.reward_cache}" in l)
    fn = _main_fn()
    guard = [n for n in ast.walk(fn)
             if isinstance(n, ast.If) and "rewards_only" in ast.dump(n.test)][0]
    assert write < guard.lineno, \
        "--rewards-only would return before the cache it exists to produce is written"
