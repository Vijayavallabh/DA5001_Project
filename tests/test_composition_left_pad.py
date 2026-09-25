"""caution (bc) in analysis/composition_attack.py: a LEFT-padded row's generation starts at the padded width.

Attacker.query sliced each row at its own token count (attention_mask.sum), so a row with p pad tokens recorded
its last p prompt tokens as the start of its generation, and its length n (hence its step counts) ran p long.
This runs the real Attacker.query on a fake two-row batch whose short row generates one token and then EOS."""
import os
import re
from types import SimpleNamespace

import torch

from test_left_pad_slicing import PAD, FakeTok

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EOS = FakeTok.eos_token_id


class FakeFactory:
    window = None

    def __init__(self, seqs, steps):
        self.seqs, self.steps = seqs, steps

    def generate(self, text, **kw):
        assert text == ["1234", "12"]
        return SimpleNamespace(sequences=self.seqs)

    def get_kl_stats_summary(self):
        return {"final_cum_kl_spent_per_seq": [0.3, 0.1], "final_budget_per_seq": [3.0, 3.0],
                "final_realised_ratio_per_seq": [0.3, 0.1],
                "per_step": [{"bd": [0.5, 0.5], "r_t": [0.1, 0.1], "prefix_debt": [0.0, 0.0]}] * self.steps}


def test_a_short_row_carries_none_of_its_prompt_into_the_query_output():
    from analysis.composition_attack import Attacker
    # row 0: prompt 1234, then 7 8 9; row 1: [PAD PAD] 1 2, then 9, EOS and a pad position
    seqs = torch.tensor([[1, 2, 3, 4, 7, 8, 9], [PAD, PAD, 1, 2, 9, EOS, PAD]])
    atk = Attacker(FakeFactory(seqs, steps=3), FakeTok(), batch_size=2, temperature=1.0, seed=0)
    out = atk.query(["1234", "12"], k=1.0, max_new=3)
    assert [o[0] for o in out] == ["xyz", "z"], [o[0] for o in out]  # the old slice gave row 1 "ABz"
    assert [o[4] for o in out] == [3, 2]  # 9 and its EOS; the old slice counted 4
    assert [a["steps"] for a in atk.last_activity] == [3, 2]


def test_no_generation_path_slices_a_row_at_its_own_token_count():
    """Both copies of the defect were `seqs[j]...[int(plens[j]):]` with `plens = attention_mask.sum(...)`; a
    generation under a left-padding tokenizer starts at the padded width, so no analysis script may slice there."""
    bad = []
    for d in ("analysis", "dap", "a_patch", "recipes"):
        for dirpath, _, files in os.walk(os.path.join(ROOT, d)):
            for f in files:
                if not f.endswith(".py"):
                    continue
                src = open(os.path.join(dirpath, f), encoding="utf-8").read()
                if re.search(r"\[\s*int\(\s*(plens|prompt_lens)\[\w+\]\s*\)\s*:\s*\]", src):
                    bad.append(os.path.relpath(os.path.join(dirpath, f), ROOT))
    assert not bad, f"a generation sliced at the row's own prompt length: {bad}"
