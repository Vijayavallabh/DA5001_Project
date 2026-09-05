import math
from typing import List, Optional, Tuple

import numpy as np


def stable_hash(text: str) -> int:
    h = 2166136261
    for ch in text:
        h = (h ^ ord(ch)) * 16777619
        h &= 0xFFFFFFFF
    return h


def build_trajectory_seeds(prompt_id: str, base_seeds: Tuple[int, ...], n: int, start: int = 0) -> List[int]:
    """Seed for trajectory index j of a prompt: (hash(prompt_id, base_seeds) mod 2^16) << 16 | j.

    Seeds of one prompt are distinct for all j < 2^16, so a stage-1 call (start=0, n=n0) and a
    top-up call (start=n0) never collide. The old scheme (base[i % 3] + offset + i) made stage-2
    index 0 equal stage-1 index 2 whenever the top-up shifted seed values by n0 (feat-003).
    """
    assert 0 <= start and start + n <= 1 << 16, "trajectory index space is [0, 65536)"
    h = stable_hash(f"{prompt_id}|{tuple(base_seeds)}") & 0xFFFF
    return [(h << 16) | j for j in range(start, start + n)]


def rouge_l_score(hypothesis: str, reference: Optional[str]) -> float:
    if not reference:
        return 0.0
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()
    if not hyp or not ref:
        return 0.0
    dp = [[0] * (len(ref) + 1) for _ in range(len(hyp) + 1)]
    for i in range(1, len(hyp) + 1):
        for j in range(1, len(ref) + 1):
            if hyp[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    prec = lcs / len(hyp)
    rec = lcs / len(ref)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


def minhash_5gram_score(hypothesis: str, reference: Optional[str]) -> float:
    if not reference:
        return 0.0
    h = hypothesis.lower().split()
    r = reference.lower().split()
    if len(h) < 5 or len(r) < 5:
        return 0.0
    h5 = set(tuple(h[i:i + 5]) for i in range(len(h) - 4))
    r5 = set(tuple(r[i:i + 5]) for i in range(len(r) - 4))
    if not h5 or not r5:
        return 0.0
    return len(h5 & r5) / len(h5 | r5)


def ebb_upper_bound_chapman(samples: List[float], R: float, delta: float) -> float:
    if not samples:
        return float("inf")
    arr = np.asarray(samples, dtype=np.float64)
    M = len(arr)
    mean_z = float(arr.mean())
    var_z = float(arr.var(ddof=1)) if M > 1 else 0.0
    empirical_R = float(arr.max() - arr.min())
    R_eff = min(R, max(empirical_R, 1.0))
    log_term = math.log(2.0 / delta)
    width = math.sqrt((2.0 * var_z * log_term) / M) + (3.0 * R_eff * log_term) / M
    return mean_z + width
