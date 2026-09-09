"""feat-060: target length in tokens is a consequence of the tokenizer for a fixed corpus, so
separating the two needs an explicit truncation. It has to bite in both the attack and the budget
path, or the ratio compares an onset on one work to an s(x) on another."""
import os, re, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(name):
    return open(os.path.join(REPO, "analysis", name)).read()


def test_both_scripts_take_the_flag():
    for f in ("composition_attack.py", "budget_path.py"):
        assert "--max-target-tokens" in _src(f), f


def test_truncation_is_applied_after_tokenising_and_before_the_seed_split():
    """ids[:seed + N] keeps the seed intact and shortens only what must be generated."""
    for f in ("composition_attack.py", "budget_path.py"):
        s = _src(f)
        assert re.search(r"ids = ids\[:\s*args\.seed_tokens \+ args\.max_target_tokens\s*\]", s), f


def test_zero_means_no_truncation():
    for f in ("composition_attack.py", "budget_path.py"):
        s = _src(f)
        assert re.search(r"if args\.max_target_tokens > 0:", s), f


def test_slicing_semantics():
    """The behaviour the two scripts rely on, pinned so a refactor cannot silently change it."""
    ids, seed, n = list(range(100)), 20, 30
    assert ids[:seed + n][:seed] == ids[:seed]          # seed unchanged
    assert len(ids[:seed + n]) - seed == n              # exactly n target tokens
    assert ids[:seed + 0 + 1000] == ids                 # a cap beyond the passage is a no-op


def test_reference_coverage_is_reported_not_enforced():
    """feat-063, corrected: recall is scored against `target`, and prompt_text is 930 characters of
    the same novel, so a target that stops before the CopyBench `reference` field still measures
    reproduction of protected text. Coverage is worth printing; aborting on it would have blocked
    feat-060, whose truncated run reaches an unconstrained recall of 0.696."""
    import inspect
    from analysis import composition_attack
    src = inspect.getsource(composition_attack.main)
    assert "covered = sum(" in src, "coverage should still be counted and printed"
    assert "raise SystemExit(msg)" not in src, "coverage must not abort a valid run"
