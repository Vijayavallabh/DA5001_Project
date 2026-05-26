import random
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ngrams(text: str, n: int = 4) -> set:
    toks = text.lower().split()
    if len(toks) < n:
        return set()
    return set(tuple(toks[i:i+n]) for i in range(len(toks) - n + 1))


def ngram_jaccard(a: str, b: str, n: int = 4) -> float:
    A = ngrams(a, n)
    B = ngrams(b, n)
    if not A and not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))


def rough_structural_tag(text: str) -> str:
    t = text.lower()
    if "continue" in t or "complete" in t:
        return "continuation"
    if ":" in text:
        return "instructional"
    if '"' in text:
        return "dialogue"
    if "chapter" in t or "excerpt" in t:
        return "excerpt"
    return "semantic"


def token_count(tokenizer, text: str) -> int:
    return int(tokenizer(text, return_tensors="pt").input_ids.shape[1])


def _canon_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _token_ngrams(text: str, n: int = 4):
    toks = _canon_text(text).split()
    if not toks:
        return set()
    if len(toks) < n:
        return {" ".join(toks)}
    return {" ".join(toks[i:i+n]) for i in range(len(toks) - n + 1)}


def _jaccard(a, b) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / max(1, len(u))


def dedupe_candidates(cands: List, history_texts: List[str], near_threshold: float = 0.92) -> List:
    out = []
    history_exact = {_canon_text(t) for t in history_texts if t and t.strip()}
    history_ngrams = [_token_ngrams(t, 4) for t in history_exact]

    seen_exact = set()
    seen_ngrams = []

    for c in cands:
        raw = c.prompt_text
        canon = _canon_text(raw)
        if not canon:
            continue

        if canon in history_exact:
            print(f"Removed candidate {c.candidate_id}: exact history duplicate")
            continue
        if canon in seen_exact:
            print(f"Removed candidate {c.candidate_id}: exact same-gen duplicate")
            continue

        cand_ngrams = _token_ngrams(canon, 4)
        near = False

        for old in history_ngrams:
            if _jaccard(cand_ngrams, old) >= near_threshold:
                near = True
                print(f"Removed candidate {c.candidate_id}: near duplicate (history)")
                break

        if not near:
            for old in seen_ngrams:
                if _jaccard(cand_ngrams, old) >= near_threshold:
                    near = True
                    print(f"Removed candidate {c.candidate_id}: near duplicate (same-gen)")
                    break

        if near:
            continue

        out.append(c)
        seen_exact.add(canon)
        seen_ngrams.append(cand_ngrams)

    return out


def filter_by_length(cands: List, tokenizer, min_tok: int, max_tok: int) -> List:
    out = []
    for c in cands:
        n = token_count(tokenizer, c.prompt_text)
        if min_tok <= n <= max_tok:
            out.append(c)
    return out


def lineage_id_for_candidate(c, archive: List) -> str:
    if c.parent_lineage_ids:
        if len(c.parent_lineage_ids) == 1:
            return c.parent_lineage_ids[0]
        return "x_" + "_".join(sorted(set(c.parent_lineage_ids)))

    nearest = None
    best = -1.0
    for a in archive:
        sim = ngram_jaccard(c.prompt_text, a.prompt_text, 4)
        if sim > best:
            best = sim
            nearest = a.lineage_id
    return nearest or f"lineage_{hash(c.prompt_text)}"


def k_dpp_select(embeddings: np.ndarray, quality: np.ndarray, k: int) -> Tuple[List[int], Dict[str, Any]]:
    n = len(embeddings)
    if n == 0:
        return [], {"mode": "empty"}
    if n <= k:
        return list(range(n)), {"mode": "all"}

    quality = np.asarray(quality, dtype=np.float64)
    quality = np.maximum(quality, 1e-8)

    dists = np.sqrt(((embeddings[:, None, :] - embeddings[None, :, :]) ** 2).sum(-1))
    q_min, q_max = float(quality.min()), float(quality.max())

    if q_max > q_min:
        q_norm = (quality - q_min) / (q_max - q_min)
    else:
        q_norm = np.ones(n, dtype=np.float64)

    selected = [int(np.argmax(q_norm))]
    remaining = set(range(n)) - set(selected)

    while remaining and len(selected) < k:
        best_i = None
        best_score = -1e18

        for i in remaining:
            min_dist = float(np.min(dists[i, selected])) if selected else 0.0
            score = 0.5 * q_norm[i] + 0.5 * min_dist
            if score > best_score:
                best_score = score
                best_i = i

        selected.append(best_i)
        remaining.remove(best_i)

    return selected, {"mode": "greedy_quality_diversity", "quality_weight": 0.5, "diversity_weight": 0.5}
