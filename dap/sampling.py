from collections import Counter, defaultdict
from typing import List

from .shared import PromptRecord


def _stable_sort_key(p: PromptRecord):
    return p.prompt_id


def _take_first_n(prompts: List[PromptRecord], n: int) -> List[PromptRecord]:
    return sorted(prompts, key=_stable_sort_key)[:n]


def _quartile_bucket(value: float, cuts: List[float]) -> int:
    if value <= cuts[0]:
        return 0
    if value <= cuts[1]:
        return 1
    if value <= cuts[2]:
        return 2
    return 3


def stratified_attack_sample(prompts: List[PromptRecord], target_n: int) -> List[PromptRecord]:
    with_debt = [p for p in prompts if p.debt_init_estimated is not None]
    without_debt = [p for p in prompts if p.debt_init_estimated is None]
    if len(with_debt) < target_n:
        return sorted(with_debt + without_debt, key=_stable_sort_key)[:target_n]

    debt_vals = sorted(float(p.debt_init_estimated) for p in with_debt)
    q1 = debt_vals[len(debt_vals) // 4]
    q2 = debt_vals[len(debt_vals) // 2]
    q3 = debt_vals[(3 * len(debt_vals)) // 4]
    cuts = [q1, q2, q3]

    buckets = {0: [], 1: [], 2: [], 3: []}
    for p in with_debt:
        b = _quartile_bucket(float(p.debt_init_estimated), cuts)
        buckets[b].append(p)

    per_bucket = target_n // 4
    sampled = []
    for b in range(4):
        sampled.extend(sorted(buckets[b], key=_stable_sort_key)[:per_bucket])

    if len(sampled) < target_n:
        used = {p.prompt_id for p in sampled}
        remainder = [p for p in sorted(with_debt, key=_stable_sort_key) if p.prompt_id not in used]
        sampled.extend(remainder[: target_n - len(sampled)])

    return sampled[:target_n]


def stratified_factual_sample(prompts: List[PromptRecord], target_n: int) -> List[PromptRecord]:
    by_source = defaultdict(list)
    for p in prompts:
        by_source[p.novel_source or "unknown"].append(p)

    groups = sorted(by_source.items(), key=lambda x: x[0])
    if not groups:
        return []

    base = target_n // len(groups)
    extra = target_n % len(groups)
    sampled = []

    for i, (_, items) in enumerate(groups):
        quota = base + (1 if i < extra else 0)
        sampled.extend(sorted(items, key=_stable_sort_key)[:quota])

    if len(sampled) < target_n:
        used = {p.prompt_id for p in sampled}
        flat = []
        for _, items in groups:
            flat.extend(sorted(items, key=_stable_sort_key))
        sampled.extend([p for p in flat if p.prompt_id not in used][: target_n - len(sampled)])

    return sampled[:target_n]


def apply_e1_sampling(prompts: List[PromptRecord], cap_neutral: int, cap_val: int, cap_test: int, cap_attack_train: int, cap_factual: int, cap_creative: int) -> List[PromptRecord]:
    grouped = defaultdict(list)
    for p in prompts:
        grouped[p.split].append(p)

    sampled = []
    sampled.extend(_take_first_n(grouped["neutral"], cap_neutral))
    sampled.extend(_take_first_n(grouped["val"], cap_val))
    sampled.extend(_take_first_n(grouped["test"], cap_test))
    sampled.extend(stratified_attack_sample(grouped["attack_train"], cap_attack_train))
    sampled.extend(stratified_factual_sample(grouped["factual"], cap_factual))
    creative_pool = [p for p in grouped["creative"] if p.cleaning_passed is not False and (p.score is None or float(p.score) >= 10)]
    sampled.extend(_take_first_n(creative_pool, cap_creative))
    return sampled


def validate_sample_counts(counts: Counter, caps: dict) -> None:
    for split, exp in caps.items():
        got = counts.get(split, 0)
        if got != exp:
            raise ValueError(
                f"Sample count mismatch for split='{split}': expected {exp}, got {got}. "
                "This usually means the filtered source pool is smaller than the configured cap."
            )
    total_expected = sum(caps.values())
    total_got = sum(counts.values())
    if total_got != total_expected:
        raise ValueError(f"Total sampled prompt count mismatch: expected {total_expected}, got {total_got}.")
