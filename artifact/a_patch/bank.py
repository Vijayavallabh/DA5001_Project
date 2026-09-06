"""feat-021: the banking rule of Anchored Decoding as a token bucket (Turner, 1986).

Without a cap the bucket reproduces the cumulative rule k_t = max{0, (t+1)k - delta - sum_{i<t} a_i}: the bank starts at
-delta (the prefix debt), gains k per step, and pays each step's spend. A depth cap D discards allowance above D at refill time,
so no burst can spend more than W*k + D over any W consecutive steps.
"""
from typing import Optional, Tuple

import torch


def bucket_step(bank: torch.Tensor, k: float, cap: Optional[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Refill the bank by k (clipped at cap) and return (new bank, allowance = max(bank, 0))."""
    bank = bank + float(k)
    if cap is not None:
        bank = bank.clamp(max=float(cap))
    return bank, bank.clamp(min=0.0)
