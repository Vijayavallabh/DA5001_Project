"""A cap of 0 means 'none of this class'. It used to crash the stratified samplers, which is how a
plan-v4 run that wanted only the three ordinary classes died at startup."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dap.sampling import stratified_attack_sample, stratified_factual_sample, apply_e1_sampling
from dap.shared import load_prompt_corpus

PROMPTS = load_prompt_corpus("data", "factscore_prompt")
BY_SPLIT = {}
for _p in PROMPTS:
    BY_SPLIT.setdefault(_p.split, []).append(_p)


def test_zero_cap_returns_nothing():
    assert stratified_attack_sample(BY_SPLIT["attack_train"], 0) == []
    assert stratified_factual_sample(BY_SPLIT["factual"], 0) == []


def test_empty_pool_never_indexes_an_empty_quartile():
    assert stratified_attack_sample([], 0) == []
    assert stratified_attack_sample([], 5) == []


def test_nonzero_caps_still_sample():
    assert len(stratified_attack_sample(BY_SPLIT["attack_train"], 7)) == 7
    assert len(stratified_factual_sample(BY_SPLIT["factual"], 9)) == 9


def test_ordinary_only_workload_excludes_the_book_splits():
    out = apply_e1_sampling(PROMPTS, 60, 0, 0, 0, 45, 45)
    got = {p.split for p in out}
    assert got == {"neutral", "factual", "creative"}, got
    assert len(out) == 150
