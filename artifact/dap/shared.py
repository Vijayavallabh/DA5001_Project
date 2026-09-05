import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()


CLASS_ORDER = ["neutral", "val", "test", "attack_train", "factual", "creative"]

# Llama-3.1-Instruct chat template output for a single user turn (BOS omitted: the tokenizer adds it).
LLAMA3_CHAT = (
    "<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)


def wrap_chat(text: str, tokenizer) -> str:
    """Wrap a raw prompt as one user turn (feat-003 --use-chat-template).

    Uses the tokenizer's own template when it has one, else the Llama-3.1 template above
    (the shared TinyComma/Llama-3 tokenizer ships without a template)."""
    if getattr(tokenizer, "chat_template", None):
        s = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        bos = getattr(tokenizer, "bos_token", None)
        return s[len(bos):] if bos and s.startswith(bos) else s
    return LLAMA3_CHAT.format(content=text)


def true_gen_len(gen_ids: List[int], eos_ids) -> int:
    """Generated tokens up to and including the first EOS; batched outputs are padded with EOS/pad to the batch length."""
    eos = set(eos_ids) if isinstance(eos_ids, (list, tuple, set)) else {eos_ids}
    for j, tok in enumerate(gen_ids):
        if tok in eos:
            return j + 1
    return len(gen_ids)


def chat_eos_ids(tokenizer) -> List[int]:
    """EOS ids for chat-formatted generation: the model's EOS plus <|eot_id|> when the vocab has it."""
    ids = [tokenizer.eos_token_id]
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot >= 0 and eot != tokenizer.eos_token_id and eot != getattr(tokenizer, "unk_token_id", None):
        ids.append(eot)
    return ids

SOURCE_FILES = {
    "copybench_attack_train.jsonl": ("copyright", "attack_train"),
    "copybench_test.jsonl": ("copyright", "test"),
    "copybench_val.jsonl": ("copyright", "val"),
    "neutral.jsonl": ("copyright", "neutral"),
    "creative.jsonl": ("creative", "creative"),
    "factscore.jsonl": ("factual", "factual"),
}


@dataclass
class PromptRecord:
    prompt_id: str
    domain: str
    split: str
    prompt_text: str
    novel_source: Optional[str] = None
    reference: Optional[str] = None
    expected_answer: Optional[str] = None
    question_type: Optional[str] = None
    truncation_type: Optional[str] = None
    debt_init_estimated: Optional[float] = None
    atomic_fact_source: Optional[str] = None
    reddit_id: Optional[str] = None
    score: Optional[int] = None
    source_file: Optional[str] = None
    cleaning_passed: Optional[bool] = None
    raw: Optional[Dict[str, Any]] = None


class PromptNormalizer:
    def __init__(self, factscore_field: str = "factscore_prompt"):
        self.factscore_field = factscore_field

    def normalize(self, raw: Dict[str, Any], filename: str, idx: int) -> PromptRecord:
        if filename not in SOURCE_FILES:
            raise ValueError(f"Unsupported file: {filename}")
        domain, split = SOURCE_FILES[filename]
        if filename.startswith("copybench_") or filename == "neutral.jsonl":
            return self._normalize_copybench_like(raw, filename, idx, domain, split)
        if filename == "creative.jsonl":
            return self._normalize_creative(raw, filename, idx, domain, split)
        if filename == "factscore.jsonl":
            return self._normalize_factscore(raw, filename, idx, domain, split)
        raise ValueError(f"No normalizer registered for {filename}")

    def _normalize_copybench_like(self, raw, filename, idx, domain, split):
        prompt_id = str(raw.get("prompt_id") or raw.get("source_excerpt_id") or f"{split}_{idx:05d}")
        prompt_text = raw.get("prefix") or raw.get("raw_text") or raw.get("prompt_text") or raw.get("prompt")
        if prompt_text is None:
            raise ValueError(f"Missing prefix/raw_text/prompt for {filename} line {idx + 1}")
        prompt_text = "Complete the prefix:\n" + str(prompt_text)
        return PromptRecord(
            prompt_id=prompt_id,
            domain=str(raw.get("domain") or domain),
            split=str(raw.get("split") or split),
            prompt_text=prompt_text,
            novel_source=raw.get("novel_source") or raw.get("source_novel"),
            reference=raw.get("reference") or raw.get("reference_text") or "",
            expected_answer=raw.get("reference") or raw.get("reference_text") or "",
            question_type=raw.get("question_type"),
            truncation_type=raw.get("truncation_type") or ("cliffhanger" if split == "attack_train" else None),
            debt_init_estimated=raw.get("debt_init_estimated"),
            source_file=filename,
            raw=raw,
        )

    def _normalize_creative(self, raw, filename, idx, domain, split):
        meta = raw.get("metadata") or {}
        prompt_text = raw.get("prompt_text") or raw.get("input") or meta.get("title")
        if prompt_text is None:
            raise ValueError(f"Missing input/prompt_text for {filename} line {idx + 1}")
        prompt_id = str(raw.get("prompt_id") or meta.get("submission_id") or f"creative_{idx:05d}")
        prompt_text = "Complete the prefix:\n" + str(prompt_text)
        return PromptRecord(
            prompt_id=prompt_id,
            domain=str(raw.get("domain") or domain),
            split=str(raw.get("split") or split),
            prompt_text=prompt_text,
            novel_source=raw.get("novel_source"),
            reference=raw.get("reference"),
            expected_answer=None,
            reddit_id=raw.get("reddit_id") or meta.get("submission_id"),
            score=raw.get("score") or meta.get("score"),
            source_file=filename,
            cleaning_passed=raw.get("cleaning_passed", True),
            raw=raw,
        )

    def _normalize_factscore(self, raw, filename, idx, domain, split):
        prompt_text = (
            raw.get("prompt_text")
            or raw.get(self.factscore_field)
            or raw.get("factscore_prompt")
            or raw.get("hundredw_prompt")
            or raw.get("around_100")
            or raw.get("one_fact_prompt")
        )
        if prompt_text is None:
            raise ValueError(f"Missing factual prompt field for {filename} line {idx + 1}")
        entity = raw.get("entity") or f"entity_{idx:05d}"
        return PromptRecord(
            prompt_id=str(raw.get("prompt_id") or f"fact_{idx:05d}_{entity}"),
            domain=str(raw.get("domain") or domain),
            split=str(raw.get("split") or split),
            prompt_text=str(prompt_text).strip(),
            novel_source=raw.get("novel_source") or raw.get("source_novel") or str(entity),
            reference=raw.get("expected_answer") or raw.get("wikipedia_text"),
            expected_answer=raw.get("expected_answer") or raw.get("wikipedia_text"),
            question_type=raw.get("question_type") or "plot_event",
            atomic_fact_source=raw.get("atomic_fact_source") or "factscore_books",
            source_file=filename,
            raw=raw,
        )


def load_prompt_corpus(data_dir: str, factscore_field: str) -> List[PromptRecord]:
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    normalizer = PromptNormalizer(factscore_field=factscore_field)
    prompts: List[PromptRecord] = []
    for filename in SOURCE_FILES:
        path = base / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        print(f"[stage] reading {path}", flush=True)
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                prompts.append(normalizer.normalize(raw, filename, idx))
    return prompts


def first_of(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            v = d[k]
            if isinstance(v, list):
                return v[0] if v else default
            return v
    return default
