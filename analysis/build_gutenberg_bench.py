"""feat-176: a public-domain COMPLETION workload, through the factual slot.

The workload split's last live candidate is the task TYPE. Every external benchmark measured so far
is instruction-following (AlpacaEval, MT-Bench) or reading comprehension (CoTaEval-QA); our own
corpus is prefix completion. Anchor competence is refuted (feat-173 B4), the support ceiling is
retracted, and the prompt template is excluded (results/prompt_header_audit.csv), so "completion
versus not" is what is left -- and it has never been tested on a corpus we did not choose.

data/gutenberg/excerpts.jsonl is 600 excerpts of 50 public-domain books, built by
analysis/build_gutenberg_excerpts.py for the onset work. This routes 500 of them through the
FACTUAL slot, which is the slot AlpacaEval and MT-Bench use and the one dap/shared.py does NOT
prepend `Complete the prefix:` to -- so the new arm differs from AlpacaEval in content alone.

The prompt is the excerpt's `raw_text` AS BUILT -- 146 to 209 words, median 170 -- and its
`reference_text` is the continuation that already exists beside it, so nothing is truncated and no
prefix length is chosen here. That length is the same scale as our own protected prompts (930
characters, about 160 words), which is the comparison this arm is for.

Usage:
  .venv/bin/python analysis/build_gutenberg_bench.py --limit 500 --words 100
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/gutenberg/excerpts.jsonl")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--words", type=int, default=0,
                    help="0 (default) uses the excerpt as built; a positive value truncates")
    ap.add_argument("--name", default="gutenberg")
    a = ap.parse_args()

    out_file = f"data/bench/{a.name}_factual.jsonl"
    out_dir = f"data/bench/{a.name}"
    rows, books = [], set()
    for line in open(a.src, encoding="utf-8"):
        r = json.loads(line)
        text = (r.get("raw_text") or "").strip()
        # raw_text IS the prefix and reference_text is its continuation; an excerpt with no
        # continuation is not a completion prompt.
        if not text or not (r.get("reference_text") or "").strip():
            continue
        w = text.split()
        rows.append(dict(prompt_id=r["prompt_id"],
                         source_novel=r.get("source_novel") or "gutenberg",
                         split="factual",
                         prompt_text=" ".join(w[:a.words]) if a.words else text,
                         expected_answer=""))
        books.add(rows[-1]["source_novel"])
        if len(rows) >= a.limit:
            break
    assert len(rows) == a.limit, f"only {len(rows)} usable excerpts of {a.limit} requested"

    with open(out_file, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # The bench corpus is a directory of symlinks to the five committed files plus one real one,
    # so a run on it is the same h1.py --data-dir code path as every run on record.
    os.makedirs(out_dir, exist_ok=True)
    here = os.path.abspath(".")
    links = {"copybench_attack_train.jsonl": "data/copybench_attack_train.jsonl",
             "copybench_test.jsonl": "data/copybench_test.jsonl",
             "copybench_val.jsonl": "data/copybench_val.jsonl",
             "creative.jsonl": "data/creative.jsonl",
             "neutral.jsonl": "data/neutral.jsonl",
             "factscore.jsonl": out_file}
    for name, target in links.items():
        p = os.path.join(out_dir, name)
        if os.path.islink(p) or os.path.exists(p):
            os.remove(p)
        os.symlink(os.path.join(here, target), p)

    # CAUTION (w): print the corpus you selected, and read it back.
    n = sum(len(r["prompt_text"].split()) for r in rows) / len(rows)
    print(f"wrote {out_file}: {len(rows)} prompts from {len(books)} books, "
          f"mean prompt {n:.1f} words")
    print(f"wrote {out_dir}/ as {len(links)} symlinks")
    print("first prompt:", repr(rows[0]["prompt_text"][:100]))


if __name__ == "__main__":
    main()
