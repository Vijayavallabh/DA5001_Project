"""feat-056: scoring a rule on the pairs it was fitted to is not a test, and the two manifests
spell pairs differently, so the join and the held-out split are what this guards."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.score_predictions import canonical, load_measurements


def test_the_two_manifest_spellings_join():
    assert canonical("TinyComma-1.8B + memorised Llama-3.1-8B") == \
           canonical("TinyComma-1.8B + mem. Llama-3.1-8B")


def test_a_re_measurement_joins_to_its_own_pair():
    assert canonical("Pleias-350M + mem. Pleias-350M (n=458)") == \
           canonical("Pleias-350M + mem. Pleias-350M")


def test_distinct_pairs_stay_distinct():
    assert canonical("Comma-7B + mem. Comma-7B") != canonical("Pleias-1.2B + mem. Pleias-1.2B")


def test_the_larger_measurement_wins(tmp_path):
    p = tmp_path / "onset_ci.csv"
    p.write_text(
        "pair,mode,n_passages,onset_point,onset_lo95,onset_hi95\n"
        "Pleias-350M + mem. Pleias-350M,single,100,3.1796,3.12,3.90\n"
        "Pleias-350M + mem. Pleias-350M (n=458),single,458,3.2667,3.21,3.41\n"
        "Comma-7B + mem. Comma-7B,oracle,100,1.57,1.4,1.8\n")
    got = load_measurements(str(p))
    assert set(got) == {canonical("Pleias-350M + mem. Pleias-350M")}   # oracle rows excluded
    assert got[canonical("Pleias-350M + mem. Pleias-350M")]["n_passages"] == "458"


def test_tightest_subset_check_is_an_exact_enumeration():
    """The matched-context claim is a 5-of-7 subgroup, which is exactly the shape of a finding
    that appears by chance, so the multiplicity check must enumerate every subset of that size
    rather than approximate. Seven values whose tightest five are obvious: the check must find
    exactly one and return 1/21."""
    import itertools, statistics as st
    r = [0.90, 0.91, 0.92, 0.89, 0.90, 1.30, 1.60]      # last two are the loose pair
    ours = (0, 1, 2, 3, 4)
    cv = lambda idx: st.stdev([r[i] for i in idx]) / st.mean([r[i] for i in idx])
    subs = list(itertools.combinations(range(7), 5))
    assert len(subs) == 21
    assert sum(1 for c in subs if cv(c) <= cv(ours) + 1e-12) == 1
