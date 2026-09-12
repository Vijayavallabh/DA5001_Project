"""The vacuity statement at 13x the corpus, and the control that caught a wrong criterion.

The first version of analysis/bookmia_regimes.py scored vacuity as `k >= s(x)` per token, which is
Proposition 1's asymptotic criterion and what Figure 1(a) draws, not what the paper's 100% means.
A deployment publishes K = k*T_max and analysis/certificate_cap.py scores S(x) <= K at T_max = 200.
The CopyBench arm is a positive control for exactly that, and these tests keep it one.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.bookmia_regimes import K_PUBLISHED, T_MAX  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "bookmia_regimes.csv")
CAPS = os.path.join(ROOT, "results", "certificate_cap_summary.csv")
LOG = os.path.join(ROOT, "results", "onset_prediction_bookmia_regimes.md")


def rows(path=CSV):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def arm(name):
    return next(r for r in rows() if r["arm"].startswith(name))


def test_the_positive_control_reproduces_the_papers_own_number():
    """The whole point of running the 758 again. If this ever drifts, the criterion has drifted,
    and the BookMIA numbers beside it mean something other than what the paper's 100% means."""
    cb = arm("CopyBench")
    assert int(cb["n_passages"]) == 758 and int(cb["n_books"]) == 16
    assert float(cb["frac_vacuous"]) == 1.0, cb["frac_vacuous"]
    published = [r for r in rows(CAPS)
                 if r["split"] == "all" and abs(float(r["k"]) - K_PUBLISHED) < 1e-9]
    assert published, "certificate_cap_summary.csv has no k=3 row for all splits"
    assert abs(float(published[0]["vacuous_pct"]) - 100.0) < 1e-9, published[0]


def test_the_budget_is_the_deployed_one_and_not_the_rate():
    """K = k * T_max, the criterion certificate_cap.py uses, and the rate criterion is carried
    beside it as a labelled secondary rather than substituted for it."""
    assert T_MAX == 200 and K_PUBLISHED == 3.0
    for r in rows():
        assert abs(float(r["K"]) - float(r["k"]) * int(r["t_max"])) < 1e-9, r["arm"]
        assert "frac_rate_below_k" in r
        # the two criteria genuinely differ here, which is why one cannot stand in for the other
        assert float(r["frac_rate_below_k"]) < float(r["frac_vacuous"]), r["arm"]
    src = open(os.path.join(ROOT, "analysis", "bookmia_regimes.py"), encoding="utf-8").read()
    assert "NOT V1" in src, "the secondary must stay labelled as not the band"


def test_the_corpus_is_the_size_the_scale_up_claims():
    seen, unseen = arm("BookMIA seen"), arm("BookMIA unseen")
    assert int(seen["n_passages"]) == 4935 and int(seen["n_books"]) == 50
    assert int(unseen["n_passages"]) == 4935 and int(unseen["n_books"]) == 50
    total = sum(int(r["n_passages"]) for r in rows() if "BookMIA" in r["arm"])
    assert total == 9870, total


def test_v1_is_reported_as_weaker_because_it_is():
    """0.9868 is below the committed 0.99, so the reading is WEAKER and the log must say WEAKER,
    not round it up to the replication it nearly is."""
    f = float(arm("BookMIA seen")["frac_vacuous"])
    assert 0.90 <= f < 0.99, f
    t = open(LOG, encoding="utf-8").read()
    assert "**WEAKER**" in t
    assert "REPLICATES" not in t.split("## Scoring, 2026-09-12")[1] or "called" in t


def test_the_seen_unseen_label_is_recorded_as_confounded():
    """Under an anchor that saw neither half their s(x) should match, and it does not. Any later
    arm that treats the halves as exchangeable is making an error this test names."""
    seen, unseen = arm("BookMIA seen"), arm("BookMIA unseen")
    d = abs(float(seen["s_tok_median"]) / float(unseen["s_tok_median"]) - 1)
    assert d > 0.05, d
    t = open(LOG, encoding="utf-8").read()
    assert "CONFOUNDED" in t
    assert "may treat the halves as exchangeable" in t


def test_the_correction_is_disclosed_rather_than_folded_in():
    t = open(LOG, encoding="utf-8").read()
    assert "A correction, made before any band was read" in t
    assert "0.326" in t, "the wrong criterion's control reading must stay on the record"
