"""feat-024: the Gutenberg body-start picker must skip front matter (title, credits, contents) and start at the first prose paragraph."""
from analysis.latent_leakage import pick_prose_paragraphs


def test_picker_skips_short_front_matter_and_contents():
    paras = ["THE NOVEL", "by Someone", "CONTENTS", "CHAPTER I. The beginning", "CHAPTER II. The middle", "CHAPTER I",
             "It was a long paragraph of actual prose " * 12, "Another prose paragraph that is also long enough " * 10]
    body = pick_prose_paragraphs(paras, min_chars=300)
    assert body[0].startswith("It was a long paragraph")
    assert len(body) == 2


def test_picker_rejects_paragraphs_that_look_like_contents():
    paras = ["CHAPTER I. The Boy Who Lived. CHAPTER II. The Vanishing Glass. " * 10, "Real prose sentence here, long enough to count. " * 10]
    body = pick_prose_paragraphs(paras, min_chars=300)
    assert body[0].startswith("Real prose")
