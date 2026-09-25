"""feat-213: the fit rule for uncut judging of 1,000-token outputs (analysis/matched_h2h.py fit_pairs)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.matched_h2h import fit_pairs  # noqa: E402


class CharTok:
    """One token per character, so a limit is a character count and the arithmetic is checkable."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "<u>" + msgs[0]["content"] + "</u><a>"

    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": list(s)}


def n_tok(prompt, x, y):
    from analysis.utility import JUDGE_TMPL
    return len(CharTok().apply_chat_template([{"content": JUDGE_TMPL.format(prompt=prompt, a=x, b=y)}]))


def test_a_pair_that_fits_is_untouched_and_one_that_does_not_is_cut_evenly():
    prompts = {"p": "Say something.", "q": "Say something."}
    base = n_tok("Say something.", "", "")
    limit = base + 3000
    pairs = {"p": ("a" * 1000, "b" * 1000), "q": ("a" * 4000, "b" * 2600)}
    out, n_cut = fit_pairs(CharTok(), pairs, prompts, limit)
    assert n_cut == 1
    assert out["p"] == pairs["p"]                                    # fits whole: never cut
    x, y = out["q"]
    assert len(x) == len(y) == 1500                                  # both texts at one 500-char multiple
    assert n_tok("Say something.", x, y) <= limit                    # and it now fits
    assert n_tok("Say something.", "a" * 2000, "b" * 2000) > limit   # the next multiple up would not


def test_the_cut_is_the_same_in_both_orders():
    prompts = {"q": "P"}
    limit = n_tok("P", "", "") + 1200
    out, _ = fit_pairs(CharTok(), {"q": ("x" * 900, "y" * 900)}, prompts, limit)
    x, y = out["q"]
    assert n_tok("P", x, y) == n_tok("P", y, x) <= limit
