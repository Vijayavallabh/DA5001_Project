import math
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

import numpy as np


def stable_hash(text: str) -> int:
    h = 2166136261
    for ch in text:
        h = (h ^ ord(ch)) * 16777619
        h &= 0xFFFFFFFF
    return h


def build_trajectory_seeds(prompt_id: str, base_seeds: Tuple[int, ...], n: int, start: int = 0) -> List[int]:
    """Seed for trajectory index j: (hash(base_seeds) mod 2^16) << 16 | j.

    Seeds are distinct across trajectory indices j < 2^16, so a stage-1 call (start=0, n=n0) and a
    top-up call (start=n0) never collide. The old scheme (base[i % 3] + offset(prompt) + i) made
    stage-2 index 0 equal stage-1 index 2 whenever the top-up shifted seed values by n0 (feat-003).
    Seeds do not depend on prompt_id (kept for API compatibility): E1 and E2 batch prompts that share
    a seed into one generate() call, so per-prompt seeds would force batch size 1 (feat-004).
    """
    assert 0 <= start and start + n <= 1 << 16, "trajectory index space is [0, 65536)"
    h = stable_hash(f"traj|{tuple(base_seeds)}") & 0xFFFF
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


def _words(text: str) -> List[str]:
    return text.lower().split()


def _blocks(a, b):
    """Non-overlapping, in-order matching blocks (i in a, j in b, size), longest-first greedy (difflib)."""
    return [m for m in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks() if m.size > 0]


def lcs_word(hypothesis: str, reference: Optional[str]) -> int:
    """Longest common substring in words (He et al. 2026 / CopyBench exact-match metric)."""
    if not reference:
        return 0
    return max((m.size for m in _blocks(_words(reference), _words(hypothesis))), default=0)


def lcs_char(hypothesis: str, reference: Optional[str]) -> int:
    """Longest common substring in characters (lower-cased, whitespace-normalised)."""
    if not reference:
        return 0
    a, b = " ".join(_words(reference)), " ".join(_words(hypothesis))
    return max((m.size for m in _blocks(a, b)), default=0)


def acs_word(hypothesis: str, reference: Optional[str], min_len: int = 5) -> int:
    """Accumulated common substring: total words in common word-substrings of length >= min_len (near-duplicate copying)."""
    if not reference:
        return 0
    return sum(m.size for m in _blocks(_words(reference), _words(hypothesis)) if m.size >= min_len)


def nv_recall(hypothesis: str, reference: Optional[str], min_len: int = 20, tau_gap: int = 2, tau_align: int = 1) -> float:
    """Near-verbatim recall (Ahmed et al. 2026, Eq. 7): fraction of reference words inside in-order near-verbatim
    spans. Greedy word-level common-substring blocks are merged when separated by <= tau_gap unmatched words on
    both sides with alignment offset drift <= tau_align; merged spans shorter than min_len words are dropped.
    Ahmed et al. use l=20 then l=100 on whole books; CopyBench references are ~50-70 words, so min_len=20 here."""
    if not reference:
        return 0.0
    ref, hyp = _words(reference), _words(hypothesis)
    if not ref:
        return 0.0
    merged = []
    for m in _blocks(ref, hyp):
        if merged:
            i0, j0, n0 = merged[-1]
            gap_ref, gap_hyp = m.a - (i0 + n0), m.b - (j0 + n0)
            if 0 <= gap_ref <= tau_gap and 0 <= gap_hyp <= tau_gap and abs((m.a - m.b) - (i0 - j0)) <= tau_align:
                merged[-1] = (i0, j0, m.a + m.size - i0)
                continue
        merged.append((m.a, m.b, m.size))
    matched = sum(n for _, _, n in merged if n >= min_len)
    return matched / len(ref)


def copying_metrics(hypothesis: str, reference: Optional[str]) -> dict:
    return {
        "rouge_l": rouge_l_score(hypothesis, reference),
        "minhash_5gram": minhash_5gram_score(hypothesis, reference),
        "lcs_word": lcs_word(hypothesis, reference),
        "lcs_char": lcs_char(hypothesis, reference),
        "acs_word": acs_word(hypothesis, reference),
        "nv_recall": nv_recall(hypothesis, reference),
    }


def anytime_valid_cs(samples: List[float], alpha: float = 0.05, c: float = 0.75) -> Tuple[float, float]:
    """Predictable-plug-in empirical-Bernstein confidence sequence for the mean of [0,1]-valued samples
    (Waudby-Smith and Ramdas, JRSSB 2024, Theorem 2, "PrPl-EB"). Valid at every stopping time: the
    interval after n samples contains the mean with probability >= 1 - alpha simultaneously over all n,
    so an audit may keep sampling until it is tight and stop whenever it likes (feat-007).

    Scale quantities to [0,1] first (e.g. Z / K, or an indicator such as 1[L(y) > K]).
    Returns (lower, upper) after all samples, clipped to [0,1]; see anytime_valid_cs_path for every n."""
    return anytime_valid_cs_path(samples, alpha, c)[-1] if samples else (0.0, 1.0)


def anytime_valid_cs_path(samples: List[float], alpha: float = 0.05, c: float = 0.75) -> List[Tuple[float, float]]:
    log_term = math.log(2.0 / alpha)
    mu_prev, var_prev = 0.5, 0.25  # priors: mu_0 = 1/2, sigma_0^2 = 1/4
    sum_x, sum_sq = 0.0, 0.0
    s_lam, s_lam_x, s_pen = 0.0, 0.0, 0.0
    out = []
    for t, x in enumerate(samples, start=1):
        if not 0.0 <= x <= 1.0:
            raise ValueError(f"samples must lie in [0,1]; got {x}")
        lam = min(math.sqrt(2.0 * log_term / (var_prev * t * math.log1p(t))), c)
        v = 4.0 * (x - mu_prev) ** 2
        psi = (-math.log1p(-lam) - lam) / 4.0
        s_lam += lam
        s_lam_x += lam * x
        s_pen += v * psi
        centre = s_lam_x / s_lam
        width = (log_term + s_pen) / s_lam
        out.append((max(0.0, centre - width), min(1.0, centre + width)))
        sum_x += x
        mu_t = (0.5 + sum_x) / (t + 1)
        sum_sq += (x - mu_t) ** 2
        mu_prev, var_prev = mu_t, (0.25 + sum_sq) / (t + 1)
    return out


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
