"""caution (bc): the generation of a LEFT-padded row starts at the padded prompt length.

Slicing at the row's own token count (attention_mask.sum) prepended the row's last `pad` prompt
tokens to its generation text -- the "echo" the MMLU, LAMBADA and host-B arms attributed to the
models. This runs the real record builder on a fake two-row batch and checks both rows."""
from types import SimpleNamespace

import torch

from dap import e1

PAD = 0
VOCAB = {1: "A", 2: "B", 3: "C", 4: "D", 7: "x", 8: "y", 9: "z"}


class Cfg(SimpleNamespace):
    """Any setting the builder reads and this test does not care about is None."""
    def __getattr__(self, name):
        return None


class FakeTok:
    pad_token_id = PAD
    eos_token_id = 99

    def __call__(self, texts, return_tensors=None, padding=None):
        ids = [[int(c) for c in t] for t in texts]
        width = max(map(len, ids))
        padded = [[PAD] * (width - len(r)) + r for r in ids]
        mask = [[0] * (width - len(r)) + [1] * len(r) for r in ids]
        return SimpleNamespace(input_ids=torch.tensor(padded), attention_mask=torch.tensor(mask))

    def decode(self, ids, skip_special_tokens=True):
        return "".join(VOCAB.get(i, "") for i in ids)


def test_a_short_row_carries_none_of_its_prompt_into_the_generation():
    runner = e1.H1AuditRunner.__new__(e1.H1AuditRunner)
    runner.tokenizer = FakeTok()
    runner.eos_ids = 99
    runner.config = Cfg(save_full_trajectories=False, use_chat_template=False,
                        model_pair="t", level="token", constraint="kl", k_values=[1.0], max_new_tokens=3,
                        initial_bank=0.0, t_max=3, delta=0.05)
    jobs = []
    for pid, text in (("long", "1234"), ("short", "12")):
        prompt = SimpleNamespace(prompt_id=pid, prompt_text=text, domain="d", split="s",
                                 novel_source=None, reference="", raw={})
        jobs.append({"prompt": prompt, "seed": 0, "trajectory_id": 0})
    # row 0: prompt 1234 then generates 789; row 1: [PAD PAD] 12 then generates 987
    seqs = torch.tensor([[1, 2, 3, 4, 7, 8, 9], [PAD, PAD, 1, 2, 9, 8, 7]])
    recs = runner._records_from_batch(jobs, SimpleNamespace(sequences=seqs), {}, 1.0)
    enc = runner.tokenizer(["1234", "12"], return_tensors="pt", padding=True)
    start = int(enc.input_ids.shape[1])
    assert runner.tokenizer.decode(seqs[1].tolist()[start:]) == "zyx"
    # the old slice, row 1's own token count, starts inside the prompt
    old = int(enc.attention_mask.sum(dim=1)[1])
    assert runner.tokenizer.decode(seqs[1].tolist()[old:]) == "ABzyx"
    gens = [r["aggregate"]["generation"] for r in recs]
    assert gens == ["xyz", "zyx"], gens
    assert [r["prefix_analysis"]["prefix_length_tokens"] for r in recs] == [4, 2]


def test_the_runner_and_the_search_harness_slice_at_the_padded_length():
    import inspect
    from dap.e2 import evaluator
    src = inspect.getsource(e1.H1AuditRunner._records_from_batch)
    assert "gen_start = int(enc.input_ids.shape[1])" in src
    assert "full_ids[gen_start:]" in src and "full_ids[prompt_len:]" not in src
    esrc = inspect.getsource(evaluator)
    assert "attention_mask.sum" not in esrc and "gen_start = int(enc.input_ids.shape[1])" in esrc
