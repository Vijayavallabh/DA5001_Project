"""feat-202..206 code paths that run before any data exists (no GPU, no download)."""
import torch

from analysis import utility
from analysis.factscore_oracle import FACT_TMPL, pick
from analysis.selection_scaling import REWARD_TMPL


class _Enc(dict):
    def to(self, device):
        return self


class _Tok:
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self):
        self.seen = []

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return msgs[0]["content"]

    def __call__(self, texts, return_tensors=None, padding=False, truncation=False, max_length=None,
                 add_special_tokens=True):
        self.seen.append((list(texts), max_length))
        ids = [[1] * min(len(t), 50) for t in texts]
        if return_tensors is None:
            return {"input_ids": ids}
        return _Enc(input_ids=torch.ones(len(texts), 3, dtype=torch.long))

    def decode(self, ids, skip_special_tokens=True):
        return "A"


class _Model:
    def generate(self, input_ids=None, **kw):
        return torch.ones(input_ids.shape[0], input_ids.shape[1] + 1, dtype=torch.long)


def test_the_default_judge_cut_is_the_one_every_pass_on_record_used():
    long = "x" * 1500
    tok = _Tok()
    assert utility.judge_batch(_Model(), tok, [(long, long, "b")], "cpu") == ["A"]
    texts, max_length = tok.seen[-1]
    want = utility.JUDGE_TMPL.format(prompt=long[:1200], a=long[:1200], b="b")
    assert texts == [want] and max_length == 2048


def test_an_uncut_judge_sees_the_whole_text():
    long = "x" * 1500
    tok = _Tok()
    utility.judge_batch(_Model(), tok, [(long, long, "b")], "cpu", max_chars=0)
    texts, max_length = tok.seen[-1]
    assert texts == [utility.JUDGE_TMPL.format(prompt=long, a=long, b="b")] and max_length == 8192


def test_the_factuality_question_is_the_only_change_to_the_scorer_template():
    assert FACT_TMPL.replace("Is every factual claim in this response accurate?",
                             "Is this a helpful, coherent and well-written response to the instruction?") \
        == REWARD_TMPL


def test_pick_is_the_committed_replay_rule():
    assert pick([0.1, 0.5, 0.5]) == 1 and pick([2.0]) == 0
