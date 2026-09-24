"""Appendix E's subgroup reading, checked against the CSV.

The manuscript side of this file was retired on 2026-09-19: appendix_seed left the paper for the
page budget (kept verbatim as sections/appendix_seed_v8_2026-09-19.tex), so there is no longer a
paragraph quoting these dispersions. The arm and results/onset_group_dispersion.csv are unchanged.

What is kept is the structural fact that mattered more than any quoted figure, because it is a
RETRACTION rather than a result: the re-seed ladders re-trained rows OF the nine-pair table, so
their dispersion is a noise floor for that table, and the >10-word family's tightness is SMALLER
than what re-training one of its own members produces -- which is why that family was never claimed
to be measurably tight. If the subgroup reading is ever promoted back into the paper, restore the
per-figure manuscript checks from git history with it.
"""
import csv
import os

import pytest

from manuscript import ROOT

CSV = os.path.join(ROOT, "results/onset_group_dispersion.csv")


@pytest.fixture(scope="module")
def rows():
    with open(CSV) as fh:
        return {r["subset"]: r for r in csv.DictReader(fh)}


def test_the_csv_still_carries_every_subset_the_reading_needed(rows):
    for k in ("all_nine", "adversary_holds_gt10_words", "adversary_holds_le10_words",
              "reseed_pleias12b_copybench", "reseed_kl3m520m_copybench"):
        assert k in rows, (k, sorted(rows))


def test_the_tight_family_is_not_tighter_than_retraining_its_own_member(rows):
    """The retraction, as an inequality over the data. Pleias-1.2B is a member of the >10-word
    family, and re-seeding it alone disperses the ratio MORE than the five pairs disperse."""
    fam = float(rows["adversary_holds_gt10_words"]["sd"])
    member = float(rows["reseed_pleias12b_copybench"]["sd"])
    assert member > fam, (member, fam,
                          "re-seeding no longer dominates the family spread; the retraction "
                          "would have to be revisited before the tightness is claimed again")


def test_the_loose_family_is_looser_than_retraining_its_own_member(rows):
    """The other half, which is what survives: the split between the families is larger than the
    noise even though the tight side's internal tightness is not."""
    fam = float(rows["adversary_holds_le10_words"]["sd"])
    member = float(rows["reseed_kl3m520m_copybench"]["sd"])
    assert fam > member, (fam, member)
