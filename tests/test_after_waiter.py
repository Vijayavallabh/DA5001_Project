"""scripts/after.sh's wait condition, checked against mock directories BEFORE it is ever launched.

Caution (ay), the ninth incident: a waiter obeyed the rule to poll the filesystem and still could
not exit, because `ls A B` is an AND over two paths that are mutually exclusive by design and `ls`
exits 2 when either is missing. The arm wrote .done at 21:11 and the loop opened an ssh every 45
seconds for 19 hours. "Check its exit status against a mock directory holding only one of them
before launching it --- four seconds, and it prints EXIT 2."

This is those four seconds, as a test, so the next waiter cannot be written the other way.
"""
import os
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COND = 'test -e "$1" -o -e "$2"'          # the condition after.sh actually uses
WRONG = 'ls "$1" "$2" >/dev/null 2>&1'    # the one that cost 19 hours


def _status(expr, have):
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "x.done"), os.path.join(d, "x.fail")
        for p in have:
            open({"done": a, "fail": b}[p], "w").close()
        return subprocess.run(["bash", "-c", f'{expr}', "_", a, b]).returncode


def test_the_or_condition_is_true_when_either_marker_exists():
    assert _status(COND, ["done"]) == 0, "a finished arm must satisfy the wait"
    assert _status(COND, ["fail"]) == 0, "a failed arm must satisfy the wait"
    assert _status(COND, ["done", "fail"]) == 0
    assert _status(COND, []) != 0, "an unfinished arm must NOT satisfy it"


def test_the_and_condition_that_cost_nineteen_hours_still_fails_that_way():
    """Kept as the counter-example. If this ever starts passing, the lesson has been lost."""
    assert _status(WRONG, ["done"]) != 0, \
        "`ls A B` now succeeds with one path missing; the caution's premise has changed"
    assert _status(WRONG, ["done", "fail"]) == 0


def test_after_sh_uses_the_or_form_and_carries_a_deadline():
    src = open(os.path.join(ROOT, "scripts", "after.sh"), encoding="utf-8").read()
    assert 'test -e "$D" -o -e "$F"' in src, "after.sh no longer uses the OR form"
    assert "DEADLINE" in src and "exit 3" in src, "after.sh lost its deadline"
    # CODE ONLY. The file's own comments name `pgrep` to explain why it is not used, and a check
    # over the whole text fails on the explanation -- which is the same class of mistake as a
    # guard satisfied by a different occurrence of its phrase, pointing the other way.
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "pgrep" not in code, "after.sh must never wait on the absence of a process (caution (c))"


def test_after_sh_refuses_a_malformed_invocation():
    r = subprocess.run(["bash", os.path.join(ROOT, "scripts", "after.sh"), "x", "1", "echo", "hi"],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "usage" in r.stderr


def test_after_sh_gives_up_rather_than_polling_forever():
    """A deadline of 0 minutes must end immediately with exit 3, not run the command."""
    env = dict(os.environ, HOME=tempfile.mkdtemp())
    os.makedirs(os.path.join(env["HOME"], "v", "logs"))
    r = subprocess.run(["bash", os.path.join(ROOT, "scripts", "after.sh"),
                        "nothing", "0", "--", "echo", "SHOULD NOT RUN"],
                       capture_output=True, text=True, env=env, timeout=30)
    assert r.returncode == 3, r
    assert "SHOULD NOT RUN" not in r.stdout
