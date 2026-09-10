"""Read a standalone protected corpus from one JSONL file (plan v5 / feat-080).

Every extraction number in this paper comes from CopyBench, and dap.shared.load_prompt_corpus reads
that by fixed filename. A second corpus needs a second reader, and it deliberately does not go
through the loader: the committed prompt sets under data/ are not to be modified, and adding a file
to SOURCE_FILES would change what every other script sees. This reads exactly the records
analysis/build_gutenberg_excerpts.py writes and returns them in the shape the rest of the pipeline
expects, so `--corpus-file` is a strictly additive option on the two scripts that take it.

The prefix carries the same "Complete the prefix:" instruction load_prompt_corpus prepends, because
the memoriser is trained on that form and the sweep must score the same one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FileRecord:
    prompt_id: str
    split: str
    prompt_text: str
    reference: str
    novel_source: Optional[str] = None


def load_corpus_file(path: str, instruction: str = "Complete the prefix:\n") -> List[FileRecord]:
    out = []
    for i, line in enumerate(open(path, encoding="utf-8")):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        prefix = r.get("raw_text") or r.get("prefix") or r.get("prompt_text")
        ref = r.get("reference_text") or r.get("reference") or ""
        if prefix is None or not ref:
            raise ValueError(f"{path} line {i + 1}: needs raw_text and reference_text")
        out.append(FileRecord(
            prompt_id=str(r.get("prompt_id") or f"file_{i:05d}"),
            split=str(r.get("split") or "file"),
            prompt_text=instruction + str(prefix),
            reference=str(ref),
            novel_source=r.get("source_novel")))
    if not out:
        raise ValueError(f"{path} is empty")
    return out
