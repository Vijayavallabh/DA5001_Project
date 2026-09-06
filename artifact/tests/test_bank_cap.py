"""feat-021: the banking rule is a token bucket; a depth cap bounds how much unspent allowance can be carried into a burst."""
import torch
from a_patch.bank import bucket_step


def test_uncapped_bucket_equals_the_cumulative_rule():
    # bank_0 = -delta; after t+1 refills and spends the allowance is (t+1)k - delta - sum(a)
    k, delta = 3.0, 6.0
    bank = torch.tensor([-delta])
    spends = [0.5, 4.0, 0.0, 2.5]
    total = 0.0
    for t, a in enumerate(spends):
        bank, allowance = bucket_step(bank, k, cap=None)
        assert torch.isclose(allowance, torch.tensor([max(0.0, (t + 1) * k - delta - total)]))
        bank = bank - a
        total += a


def test_capped_bucket_never_offers_more_than_the_cap():
    k, cap = 3.0, 4.0
    bank = torch.tensor([0.0])
    for _ in range(50):  # never spend: an uncapped bank would reach 150
        bank, allowance = bucket_step(bank, k, cap=cap)
        assert allowance.item() <= cap + 1e-6
    assert bank.item() == cap


def test_capped_bucket_still_owes_the_prefix_debt():
    bank = torch.tensor([-10.0])
    bank, allowance = bucket_step(bank, 3.0, cap=4.0)
    assert allowance.item() == 0.0 and bank.item() == -7.0
