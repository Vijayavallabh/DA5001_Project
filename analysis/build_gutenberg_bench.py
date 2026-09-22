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


def select(rows, limit, mode):
    """`limit` rows, spread over books unless `mode` is "first".

    Extracted so it can be tested without a corpus: `first` is what produced a six-book "27-book"
    workload, and the only way to see that a round robin fixes it is to run it on an input whose
    book sizes are lopsided."""
    if mode == "first":
        return rows[:limit]
    by_book = {}
    for r in rows:
        by_book.setdefault(r["source_novel"], []).append(r)
    picked, i = [], 0
    while len(picked) < limit:
        added = False
        for b in sorted(by_book):
            if i < len(by_book[b]):
                picked.append(by_book[b][i])
                added = True
                if len(picked) >= limit:
                    break
        if not added:
            break
        i += 1
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/gutenberg/excerpts.jsonl")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--words", type=int, default=0,
                    help="0 (default) uses the excerpt as built; a positive value truncates")
    ap.add_argument("--name", default="gutenberg")
    ap.add_argument("--select", default="roundrobin", choices=("first", "roundrobin"),
                    help="roundrobin (default) takes an even spread over books; `first` is the "
                         "original prefix behaviour and is kept ONLY so the feat-176 Gutenberg "
                         "arm's corpus file still reproduces the corpus it ran on")
    a = ap.parse_args()

    out_file = f"data/bench/{a.name}_factual.jsonl"
    out_dir = f"data/bench/{a.name}"
    rows, books, src = [], set(), {}
    for line in open(a.src, encoding="utf-8"):
        r = json.loads(line)
        text = (r.get("raw_text") or "").strip()
        # raw_text IS the prefix and reference_text is its continuation; an excerpt with no
        # continuation is not a completion prompt.
        if not text or not (r.get("reference_text") or "").strip():
            continue
        # THE EXCERPTS ARE RAW CHARACTER SLICES AND SOME BEGIN MID-WORD ("rk hall, and wander
        # about..." is Alice's "dark hall"). Mid-SENTENCE is right -- our own protected prompts
        # are 930 characters cut out of a novel and start that way too -- but mid-word is a
        # different thing, and a prompt nobody can parse depresses every arm's completion and
        # costs the comparison power. Drop one leading token where the slice did not start at a
        # word boundary; the rule is uniform and is fixed here, before any generation.
        if text[0].islower():
            text = text.split(None, 1)[1] if " " in text else text
        w = text.split()
        rows.append(dict(prompt_id=r["prompt_id"],
                         source_novel=r.get("source_novel") or "gutenberg",
                         split="factual",
                         prompt_text=" ".join(w[:a.words]) if a.words else text,
                         expected_answer=""))
        books.add(rows[-1]["source_novel"])
        src[r["prompt_id"]] = r

    # `--limit N` MUST NOT MEAN "the first N". These files are ordered by book, so a prefix of
    # them is a few books repeated -- 500 rows of the BookMIA unseen half gave 6 books where the
    # file holds 27. That is caution (w) exactly, the defect that made feat-109's corpus Fifty
    # Shades instead of Harry Potter. Take a round robin over books instead, which is a prefix
    # only when one book supplies everything.
    rows = select(rows, a.limit, a.select)
    books = {r["source_novel"] for r in rows}
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

    # A LEAKAGE PROBE MUST SEE THE SAME PASSAGES THE WORKLOAD USES. The vetting instrument reads a
    # copybench-shaped slot, so the SELECTED records are written back in their original schema and
    # symlinked into one. Without this, G3 would measure the first N passages of the source file --
    # caution (w) again, in the gate rather than the corpus.
    leak_file = f"data/bench/{a.name}_leak.jsonl"
    leak_dir = f"data/bench/{a.name}_leak"
    with open(leak_file, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(dict(src[r["prompt_id"]], split="attack_train"),
                               ensure_ascii=False) + "\n")
    os.makedirs(leak_dir, exist_ok=True)
    # Only the attack_train slot holds the probe corpus. The other two keep the committed files,
    # whose records carry their own `split`, so `--split attack_train` reads the 500 selected
    # passages and nothing else -- pointing all three at the leak file would triple them.
    for name, target in {"copybench_attack_train.jsonl": leak_file,
                         "copybench_test.jsonl": "data/copybench_test.jsonl",
                         "copybench_val.jsonl": "data/copybench_val.jsonl",
                         "creative.jsonl": "data/creative.jsonl",
                         "neutral.jsonl": "data/neutral.jsonl",
                         "factscore.jsonl": out_file}.items():
        q = os.path.join(leak_dir, name)
        if os.path.islink(q) or os.path.exists(q):
            os.remove(q)
        os.symlink(os.path.join(os.path.abspath("."), target), q)
    print(f"wrote {leak_file} and {leak_dir}/ for the leakage probe")

    # CAUTION (w): print the corpus you selected, and read it back.
    n = sum(len(r["prompt_text"].split()) for r in rows) / len(rows)
    print(f"wrote {out_file}: {len(rows)} prompts from {len(books)} books, "
          f"mean prompt {n:.1f} words  (--select {a.select})")
    print(f"wrote {out_dir}/ as {len(links)} symlinks")
    print("first prompt:", repr(rows[0]["prompt_text"][:100]))


if __name__ == "__main__":
    main()
