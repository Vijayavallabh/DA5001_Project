"""Appendix E's nine-row seed table left the manuscript on 2026-09-19 (page budget: the appendix
went 30 -> 25 pages and appendix_seed was retired whole, kept verbatim as
sections/appendix_seed_v8_2026-09-19.tex). The 45-cell check against the manuscript is therefore
retired -- deliberately, and recorded here rather than deleted.

What is NOT retired is the defect that check existed for. Two columns of that table had been
maintained by hand and reproduced from no definition: `steps to it`, and open-calm's characters per
token, which disagreed with the seed length in its own row (53.1/20 = 2.66, printed as 2.74). Every
column is written by analysis/seed_effect.py into results/onset_seed_words.csv, that file still
ships in the artifact, and the internal consistency a hand-maintained column broke is asserted
below. If the seed arm is ever promoted back into the paper, restore the cell-by-cell check from
git history alongside it.
"""
import csv

CSV = "results/onset_seed_words.csv"


def rows():
    with open(CSV, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_the_seed_word_csv_still_ships_all_nine_pairs():
    rs = rows()
    assert len(rs) == 9, [r["pair"] for r in rs]
    assert len({r["pair"] for r in rs}) == 9


def test_characters_per_token_is_the_seed_length_over_twenty_tokens():
    """The hand-maintained column disagreed with its own row; a derived column cannot."""
    for r in rows():
        cpt, chars = float(r["chars_per_token"]), float(r["seed_chars"])
        assert abs(cpt - chars / 20) < 5e-3, (r["pair"], cpt, chars / 20)


def test_a_finer_tokenizer_buys_the_adversary_fewer_words():
    """The relationship the retired appendix was about, kept as a property of the data: seed words
    rise with characters per token over the nine pairs."""
    rs = sorted(rows(), key=lambda r: float(r["chars_per_token"]))
    words = [float(r["seed_words"]) for r in rs]
    assert words == sorted(words), words
