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


def _bookmia_paragraph():
    """The app:bookmia paragraph and its table, whitespace-normalised. It lived in
    appendix_robustness.tex through v9; the v10 restructure (2026-09-24) folded that file into
    appendix_onset.tex (Appendix G). Scoped to the paragraph so another occurrence of a number
    elsewhere in the appendix cannot satisfy the guard (caution (an))."""
    from tests.manuscript import body
    txt = body("appendix_onset.tex")
    assert txt.count(r"\label{app:bookmia}") == 1, "the BookMIA vacuity paragraph is gone"
    i = txt.index(r"\label{app:bookmia}")
    return txt[txt.rindex(r"\paragraph{", 0, i): txt.index(r"\paragraph{", i)]  # heading included


def test_the_manuscript_quotes_the_bookmia_table_from_the_csv():
    """Caution (j): a paper number rounds from the CSV, once, and the check is mechanical."""
    apx = _bookmia_paragraph()
    for r in rows():
        n = f"{int(r['n_passages']):,}".replace(",", "{,}")
        assert f"${n}$" in apx, (r["arm"], n)
        assert f"${float(r['frac_vacuous']):.4f}$" in apx, (r["arm"], r["frac_vacuous"])
        assert f"${float(r['S_median']):.1f}$" in apx, (r["arm"], r["S_median"])
        assert f"${float(r['s_tok_median']):.3f}$" in apx, (r["arm"], r["s_tok_median"])


def test_section4_quotes_the_scaled_vacuity_fraction():
    """The prose carries the seen half's vacuity reading, rounded once from the CSV, beside the corpus
    size it was measured on.

    Through v9 this was the clause "vacuous for ... $98.7\\%$ of $9{,}870$ across a hundred books" in
    the repairs summary of appendix_proofs.tex; v10 (2026-09-24) cut that summary and states the same
    reading in Appendix G's BookMIA paragraph as a shortfall count, "$65$ of $4{,}935$ seen passages".
    The count is derived from the CSV, so it moves if `frac_vacuous` moves by a single passage, and it
    pairs the seen half's fraction with the seen half's own denominator (4,935) -- v9's clause set it
    beside the whole corpus (9,870), one fraction with two denominators (caution (ai))."""
    para = _bookmia_paragraph()
    seen = arm("BookMIA seen")
    n = int(seen["n_passages"])
    short = round(n * (1 - float(seen["frac_vacuous"])))
    cnt = f"${n:,}$".replace(",", "{,}")
    assert f"${short}$ of {cnt} seen passages" in para, (short, n, "the shortfall is not quoted")
    assert "hundred books" in para
    total = sum(int(r["n_books"]) for r in rows() if "BookMIA" in r["arm"])
    assert total == 100, total


def test_the_shortfall_is_the_count_the_log_states():
    """65 of 4,935. If the CSV moves, the sentence in the scoring log has to move with it."""
    seen = arm("BookMIA seen")
    n = int(seen["n_passages"])
    short = round(n * (1 - float(seen["frac_vacuous"])))
    assert short == 65, short
    t = open(LOG, encoding="utf-8").read()
    assert f"`{short}`" in t, f"the scoring log must state the shortfall count {short}"
    assert f"`{n:,}`".replace(",", "{,}") in t, n
