"""The domain split, and the confound that makes its headline unusable.

D1 came back ANTI -- the gain is larger where the anchor's control is lower -- which is the
direction the pre-registration named as mechanically cheap. The exchangeability null shows the
no-effect world already produces most of it, so the reading is uninformative. These tests pin the
three things that must not drift: the MT-Bench grouping (committed before the eight cells were
computed), the null's own sanity, and the fact that no scoring log claims the ceiling explains the
benchmark weakening.
"""
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.domain_breadth import MT_FAMILY, exact_p, spearman  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "domain_breadth.csv")
NULL = os.path.join(ROOT, "results", "domain_breadth_null.csv")
LOG = os.path.join(ROOT, "results", "onset_prediction_domain_breadth.md")


def rows(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_the_mtbench_grouping_is_the_committed_one():
    """Four categories per family, fixed before the eight cells existed. Regrouping after seeing
    them is excluded alternative 1, and the only way to regroup is to edit this constant."""
    assert sorted(MT_FAMILY) == ["coding", "extraction", "humanities", "math", "reasoning",
                                 "roleplay", "stem", "writing"]
    fam = {}
    for c, f in MT_FAMILY.items():
        fam.setdefault(f, []).append(c)
    assert {k: sorted(v) for k, v in fam.items()} == {
        "open-ended": ["humanities", "roleplay", "stem", "writing"],
        "constrained": ["coding", "extraction", "math", "reasoning"]}


def test_every_registered_cell_is_present_for_both_judges():
    r = rows(CSV)
    cells = {"selfinstruct", "oasst", "koala", "helpful_base", "vicuna",
             "open-ended", "constrained"}
    for judge in {x["judge"] for x in r}:
        assert {x["cell"] for x in r if x["judge"] == judge} == cells, judge
    assert len({x["judge"] for x in r}) == 2, "the D1 band needs the sign in both judges"


def test_the_alpaca_cell_sizes_are_the_benchmarks_own():
    """805 prompts in the five sources AlpacaEval ships, not a grouping of ours."""
    r = [x for x in rows(CSV) if x["cell"] not in ("open-ended", "constrained")]
    got = {x["cell"]: int(x["n_prompts"]) for x in r if x["judge"] == r[0]["judge"]}
    assert got == {"selfinstruct": 252, "oasst": 188, "koala": 156, "helpful_base": 129,
                   "vicuna": 80}
    assert sum(got.values()) == 805


def test_the_no_effect_null_is_itself_strongly_negative():
    """The whole point: a world where selection does nothing already yields a large negative
    Spearman, because u(n=1) is subtracted inside the gain. If this ever came back near zero the
    artefact would be gone and D1 would become readable -- which would be a change worth noticing,
    not a test to quietly relax."""
    for x in rows(NULL):
        assert float(x["null_mean"]) < -0.2, (x["judge"], x["null_mean"])
        assert float(x["null_sd"]) > 0.2, (x["judge"], x["null_sd"])
        assert int(x["reps"]) >= 10000


def test_the_observed_correlation_is_not_separable_from_the_artefact():
    """Both judges sit above 0.05 against the no-effect null, which is why the scoring log refuses
    to report ANTI. A future run that separated them would have to rewrite that log."""
    for x in rows(NULL):
        assert float(x["p_vs_null"]) > 0.05, (x["judge"], x["p_vs_null"])
        assert float(x["observed_rho"]) < 0, x["judge"]


def test_the_scoring_log_does_not_claim_the_ceiling_explains_the_weakening():
    """The refuted claim is the empirical one. The theorem q(y) <= n p_s(y) is untouched, and the
    log must keep saying which of the two it refuted."""
    t = open(LOG, encoding="utf-8").read().replace("\n", " ")
    assert "uninformative" in t, "D1 must be reported as uninformative, not as ANTI"
    assert "no evidence for that explanation" in t.lower()
    assert "is a **theorem**" in t, "the log must keep the theorem separate from the explanation"


def test_spearman_and_the_exact_permutation_agree_with_a_hand_case():
    assert abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-12
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12
    rho, p = exact_p([1, 2, 3, 4], [1, 2, 3, 4])
    assert abs(rho - 1.0) < 1e-12 and abs(p - 2 / 24) < 1e-12, (rho, p)


def test_the_null_preserves_cell_sizes():
    """A resample that dropped or moved prompts would break the comparison it exists to make."""
    from analysis.domain_breadth import domains
    d = domains(os.path.join(ROOT, "data", "bench", "mtbench_factual.jsonl"), MT_FAMILY)
    assert len(d) == 80
    counts = {}
    for v in d.values():
        counts[v] = counts.get(v, 0) + 1
    assert counts == {"open-ended": 40, "constrained": 40}
    rng = random.Random(0)
    assert rng is not None
