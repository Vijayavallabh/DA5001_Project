"""Appendix E's subgroup reading, checked against the CSV rather than by eye.

Caution (j): a paper number must round from the CSV once. This file pins the dispersion figures
Appendix~\\ref{app:seed} now quotes, and one structural fact that matters more than any of them --
that the re-seed ladders re-trained rows OF the nine-pair table, so their dispersion is a noise
floor for that table and not a number from somewhere adjacent to it.

The claim being guarded is a retraction, not a result: the >10-word family's tightness is SMALLER
than what re-training one of its own members produces, so it is not resolved. A future edit that
quietly drops the comparison, or that lets the two sides drift apart, should fail here.
"""
import csv
import os

import pytest

from manuscript import ROOT, tex

CSV = os.path.join(ROOT, "results/onset_group_dispersion.csv")
TEX = tex("sections/appendix_seed.tex")


@pytest.fixture(scope="module")
def rows():
    with open(CSV) as fh:
        return {r["subset"]: r for r in csv.DictReader(fh)}


@pytest.fixture(scope="module")
def body():
    """Whitespace-collapsed, so a grep for a phrase is not defeated by where the line wrapped.

    LaTeX prose is hard-wrapped at ~100 columns and any sentence long enough to be worth pinning
    will contain a newline somewhere. A test that greps the raw file passes or fails on the wrap
    point, which is exactly the kind of check that silently stops checking after a reflow.
    """
    assert os.path.exists(TEX), f"manuscript not found at {TEX}; set $SATML_DIR"
    return " ".join(open(TEX).read().split())


def test_the_csv_carries_every_subset_the_appendix_reads(rows):
    for key in ("all_nine", "adversary_holds_gt10_words", "adversary_holds_le10_words",
                "family_mean_gap", "reseed_pleias12b_copybench", "reseed_kl3m520m_copybench"):
        assert key in rows, f"{key} missing from {CSV}"


@pytest.mark.parametrize("subset,field,fmt", [
    ("adversary_holds_gt10_words", "sd", "{:.4f}"),      # 0.0212
    ("adversary_holds_le10_words", "sd", "{:.4f}"),      # 0.0748
    ("reseed_pleias12b_copybench", "sd", "{:.4f}"),      # 0.0450
    ("reseed_kl3m520m_copybench", "sd", "{:.4f}"),       # 0.0180
    # the two re-seed SPAN figures were cut with their paragraph on 2026-09-19; the sd and the
    # nine-pair span below are still quoted and still checked.
    ("all_nine", "span", "{:.4f}"),                      # 0.2874
    ("family_mean_gap", "mean", "{:.3f}"),               # 0.159
])
def test_each_quoted_figure_rounds_from_the_csv(rows, body, subset, field, fmt):
    want = fmt.format(float(rows[subset][field]))
    assert f"${want}$" in body, f"{subset}.{field} rounds to {want}, not found in appendix_seed.tex"


@pytest.mark.parametrize("subset,want", [
    ("adversary_holds_gt10_words", "2.4"),
    ("adversary_holds_le10_words", "7.1"),
    ("reseed_pleias12b_copybench", "5.3"),
    ("reseed_kl3m520m_copybench", "1.7"),
])
def test_each_quoted_cv_rounds_from_the_csv(rows, body, subset, want):
    assert f"{float(rows[subset]['cv_pct']):.1f}" == want, "CSV moved; update the appendix too"
    assert f"${want}\\%$" in body, f"{subset} cv {want}% not found in appendix_seed.tex"


def test_the_family_the_appendix_calls_unresolved_really_is(rows):
    """The whole retraction rests on one inequality. If it ever flips, the text is wrong."""
    family = float(rows["adversary_holds_gt10_words"]["sd"])
    member = float(rows["reseed_pleias12b_copybench"]["sd"])
    assert family < member, (
        f"the >10-word family now disperses MORE ({family}) than re-seeding its member Pleias-1.2B "
        f"({member}); the appendix says the opposite and must be rewritten")


def test_the_split_the_appendix_keeps_really_is_resolved(rows):
    """The units claim rests on the separation, which must stay above both measured floors."""
    gap = float(rows["family_mean_gap"]["mean"])
    floors = [float(rows[k]["sd"]) for k in
              ("reseed_pleias12b_copybench", "reseed_kl3m520m_copybench")]
    assert gap > 2 * max(floors), f"family gap {gap} no longer clears the noise floor {max(floors)}"
    assert float(rows["all_nine"]["span"]) > max(
        float(rows[k]["span"]) for k in ("reseed_pleias12b_copybench", "reseed_kl3m520m_copybench"))


def test_the_appendix_does_not_still_call_the_subgroup_a_measurement(body):
    """The pre-correction text read the subgroup tightness as a finding. Guard the words too."""
    assert "not resolved against fine-tune noise" in body
    assert "treat" in body and "point estimates" in body, \
        "the appendix must still disclose that the exact p treats the ratios as noiseless"


def test_the_strength_ranking_the_appendix_quotes_rounds_from_the_csv(rows):
    """Added with the strength column: a reader can now compute this, so we must state it right."""
    r = rows["strength_vs_ratio_spearman"]
    rho, pval, factor = float(r["mean"]), float(r["cv_pct"]), float(r["span"])
    onset = " ".join(open(tex("sections/appendix_onset.tex"), encoding="utf-8").read().split())
    assert f"Spearman ${rho:.3f}$, exact $p = {pval:.2f}$" in onset, \
        f"appendix_onset must say Spearman {rho:.3f}, exact p = {pval:.2f}"
    assert int(r["n"]) == 9
    # The claim the sentence makes is that strength does NOT rank the pairs. Guard it.
    assert pval > 0.05, (
        f"strength now ranks the nine at p={pval}; the appendix says it fails to explain the "
        "ordering and must be rewritten")
    assert f"{factor:.1f}" == "5.1", "the strength factor moved; the two appendices quote it"
