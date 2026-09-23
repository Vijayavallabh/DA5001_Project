"""The evening's local dispatcher may move a job's CARD and nothing else: every command must carry the
registered flags of the launcher it replaces, and the card rule must refuse a card without room."""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "local_dispatch", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "scripts", "local_dispatch.py"))
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)
JOBS = dict(D.JOBS + D.VET_OWED)


def _flags(args):
    return " ".join(args) + " "


def test_the_vetting_rungs_are_the_registered_screen_at_their_prefix():
    for name, anchor in (("vetladder_L150_tinycomma", D.TINY), ("vetladder_L150_kl3m17b", D.K17)):
        a = _flags(JOBS[name])
        for f in (f"--safe-model {anchor} ", "--seed-tokens 150 ", "--raw-prompt ", "--split test ",
                  "--novel harry_potter ", "--limit 50 ", "--max-new-tokens 200 ", "--n-values 1 8 64 ",
                  "--batch-size 8 ", f"--risky-model {D.MEM} "):
            assert f in a, (name, f)
        assert "--seed " not in a, "the screen on record ran at the default seed"


def test_grid64_and_the_redraw_match_their_launchers():
    g = _flags(JOBS["selfix_clean_grid64"])
    assert "--n-values 1 2 4 8 16 32 64 " in g and "--limit 100 " in g and "--safe-model" not in g
    assert "--batch-size" not in g and "--seed " not in g           # defaults, as on record
    reds = {p: _flags(a) for p, a in JOBS.items() if p.startswith("selfixR_")}
    assert len(reds) == 12
    for p, a in reds.items():
        assert "--seed 5678 " in a and "--n-values 1 8 64 256 " in a and "--batch-size 32 " in a, p
        assert ("--experts-impl eager " in a) == (p in ("selfixR_kl3m37b", "selfixR_kl3m520m")), p


def test_choose_refuses_cards_without_room():
    assert D.choose({"0": 5000, "1": 30000, "2": 29000, "4": 0}) == "1"
    assert D.choose({"0": 5000, "1": 27999}) is None


def test_the_two_rungs_moved_to_host_b_are_not_also_placed_here():
    names = [p for p, _ in D.JOBS]
    assert "vetladder_L150_tinycomma" not in names and "vetladder_L150_kl3m17b" not in names
