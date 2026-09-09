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
        pairs.setdefault(f[5], []).append(int(f[1]))
    # every pair varies only the seed, so each must have at least two distinct seeds
    for pair, seeds in pairs.items():
        assert len(set(seeds)) == len(seeds), f"{pair} repeats a seed"
        assert len(seeds) >= 2, f"{pair} has nothing to compare"


def test_seed_words_is_measured_not_assumed(monkeypatch):
    """It must read the passages, not multiply the tokenizer's average by the seed length:
    the seed straddles a token boundary and the average would round the wrong way."""
    import inspect
    from analysis import seed_effect
    src = inspect.getsource(seed_effect.seed_words)
    assert "load_prompt_corpus" in src and "decode" in src
    assert "chars_per_token" not in src
