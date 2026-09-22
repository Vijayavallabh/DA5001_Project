"""The adversarial selector must score each candidate's OWN tokens, whatever the padding side.

Found 2026-09-23: load_tok() pads on the left, as generation needs, and score() read positions
[plen, mask.sum()) as if the batch were right-padded. For every candidate shorter than the longest
in its batch that window lands on the prompt's tail and on the padding: a one-token continuation
was scored as '<|end_of_text|>'. Every selection_extraction arm since 2026-09-11 ranked its
candidates partly on padding. Clean-anchor zeros are untouched (no draw in any of their pools
reproduces anything, so no selector could serve a leak); the contaminated-anchor amplification and
the ROUGE-L counts at n > 1 are not.

The check needs no model download: a fake model whose next-token logits depend only on the current
token makes a candidate's score a function of its own tokens, so scoring it alone and scoring it
inside a padded batch must agree exactly.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_extraction import score  # noqa: E402


class _Enc(dict):
    def to(self, device):
        return self


class _Tok:
    """Character-level, BOS = 1, pad = 0."""

    def __init__(self, side):
        self.padding_side = side

    @staticmethod
    def _ids(s):
        return [1] + [2 + ord(c) % 60 for c in s]

    def __call__(self, x, return_tensors=None, padding=False, truncation=False, max_length=None):
        if isinstance(x, str):
            return {"input_ids": self._ids(x)}
        rows = [self._ids(s) for s in x]
        width = max(map(len, rows))
        ids, mask = [], []
        for r in rows:
            pad = width - len(r)
            left = self.padding_side == "left"
            ids.append([0] * pad + r if left else r + [0] * pad)
            mask.append([0] * pad + [1] * len(r) if left else [1] * len(r) + [0] * pad)
        return _Enc(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask))


class _Model:
    device = "cpu"

    def __init__(self):
        self.w = torch.randn(64, 64, generator=torch.Generator().manual_seed(0))

    def __call__(self, input_ids, attention_mask=None):
        return type("Out", (), {"logits": self.w[input_ids]})()


PAIRS = [("It was a bright cold day in April", " and the clocks were striking thirteen."),
         ("It was a bright cold day in April", " W"),
         ("It was a bright cold day in April", " Winston Smith, his chin nuzzled")]


def test_a_padded_batch_scores_every_candidate_as_it_scores_alone():
    for side in ("left", "right"):
        tok, model = _Tok(side), _Model()
        alone = [score(model, tok, [p], 1)[0] for p in PAIRS]
        batched = score(model, tok, PAIRS, len(PAIRS))
        for a, b in zip(alone, batched):
            assert abs(a - b) < 1e-6, (side, alone, batched)


def test_the_extraction_tokenizer_really_pads_left():
    """The case this guards: if load_tok() ever stops padding left, the test above still holds
    but loses its point, so pin the side it was written for."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "analysis", "selection_extraction.py"), encoding="utf-8").read()
    assert 'padding_side="left"' in src[src.index("def load_tok("):]
