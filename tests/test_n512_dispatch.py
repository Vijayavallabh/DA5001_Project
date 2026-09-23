"""feat-181's placement rule: most free memory among cards with room and fewer than MAXPER of ours."""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "n512_dispatch", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "scripts", "n512_dispatch.py"))
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)


def test_choose_takes_the_most_free_card_that_qualifies_and_nothing_when_none_does():
    rows = [("0", "81559", "75000"), ("1", "81559", "20000"), ("2", "81559", "30000"), ("3", "81559", "0")]
    assert D.choose(rows, {}) == "3"
    assert D.choose(rows, {"3": 2}) == "1"                      # full of ours: skipped
    assert D.choose(rows, {"3": 2, "1": 2, "2": 2}) is None      # card 0 has only 6.5 GB free
    assert D.choose([("0", "81559", "45600")], {}) is None      # 35.9 GB free < 36 GB


def test_the_jobs_cover_indices_256_to_511_once_per_process():
    for arm in ("a", "small", "factual"):
        assert sorted(i for a, s, c in D.JOBS if a == arm for i in range(s, s + c)) == list(range(256, 512))
