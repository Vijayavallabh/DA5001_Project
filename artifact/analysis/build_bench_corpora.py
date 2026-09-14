"""Build the two scale-up corpora the 2026-09-12 token unblocked, WITHOUT touching data/.

The committed prompt sets are read-only (AGENTS.md), and h1.py takes a `--data-dir`, so each new
corpus is a directory of symlinks to the five files that do not change plus one real file that does.
Nothing under data/*.jsonl is written, and a run on a new corpus is the same code path as every run
on record.

  data/bench/alpaca/    neutral.jsonl  <- AlpacaEval's 805 instructions (the other five symlinked)
  data/bench/bookmia100/       <- the 50 BookMIA books marked seen (label = 1)
  data/bench/bookmia100unseen/ <- the 50 marked unseen (label = 0), a negative control

AlpacaEval is the standard instruction-following set; the paper's judged utility has until now been
measured only on 500 in-house prompts, which is the "no standard benchmark" gap.

BookMIA-100 is the corpus our 16 novels were drawn from. Passages are the `label = 1` snippets --
the ones the benchmark marks as seen -- cut to the same shape as the committed CopyBench files: a
`raw_text` prefix for conditioning and a `reference_text` continuation to score recall against. The
split assignment is BY BOOK and deterministic in a hash of the title, so attack_train / val / test
stay disjoint in novel, which caution (h) requires: a memoriser is fine-tuned on attack_train + val
and must never have seen test.

Usage: .venv/bin/python analysis/build_bench_corpora.py
"""
import hashlib
import json
import os

BENCH = "data/bench"
COMMITTED = ["copybench_attack_train.jsonl", "copybench_test.jsonl", "copybench_val.jsonl",
             "neutral.jsonl", "creative.jsonl", "factscore.jsonl"]
# the committed files carry a 930-character prefix and a ~250-character continuation; match them so
# recall numbers are comparable across corpora rather than confounded by how much context is given.
PREFIX_CHARS, REF_CHARS = 930, 250


def cut(text, n):
    """Cut at the last word boundary at or before n characters.

    The committed files are cut on boundaries, not on a character count (median prefix 886, range
    584-1123). Slicing mid-word would change what the adversary is handed and perturb recall for a
    reason that has nothing to do with the budget, so match the convention rather than the number."""
    if len(text) <= n:
        return text
    head = text[:n]
    i = head.rfind(" ")
    return head[:i] if i > n // 2 else head


def link_dir(name, replace):
    """A --data-dir that is the committed corpus with `replace` (filename -> path) swapped in."""
    d = os.path.join(BENCH, name)
    os.makedirs(d, exist_ok=True)
    for f in COMMITTED:
        dst = os.path.join(d, f)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        if f in replace:
            os.symlink(os.path.abspath(replace[f]), dst)
        else:
            os.symlink(os.path.abspath(os.path.join("data", f)), dst)
    return d


def build_alpaca():
    src = os.path.join(BENCH, "alpaca_eval.json")
    rows = json.load(open(src))
    # The FACTUAL slot, not the neutral one: dap/shared.py wraps every copyright-domain prompt in
    # "Complete the prefix:", which is right for a passage and wrong for an instruction. The
    # factscore normaliser passes prompt_text through verbatim, which is what an instruction
    # benchmark needs if the number is to mean what AlpacaEval means.
    out = os.path.join(BENCH, "alpaca_factual.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            fh.write(json.dumps({
                "prompt_id": f"alpaca_{i:04d}",
                "source_novel": r["dataset"],          # selfinstruct / oasst / koala / ...
                "split": "factual",
                "prompt_text": r["instruction"],
                "expected_answer": "",                 # an instruction set has no protected target
            }) + "\n")
    d = link_dir("alpaca", {"factscore.jsonl": out})
    print(f"alpaca: {len(rows)} instructions -> {out}; data-dir {d}")
    return len(rows)


def build_bookmia(label=1, tag="bookmia100"):
    src = os.path.join(BENCH, "bookmia.jsonl")
    rows = [json.loads(l) for l in open(src)]
    seen = [r for r in rows if r.get("label") == label
            and len(r.get("snippet", "")) >= PREFIX_CHARS + 60]
    by_book = {}
    for r in seen:
        by_book.setdefault(r["book"], []).append(r)
    # split by BOOK, deterministically: caution (h), the splits must be disjoint in novel
    def bucket(book):
        h = int(hashlib.sha256(book.encode()).hexdigest(), 16) % 10
        return "attack_train" if h < 6 else ("val" if h < 8 else "test")
    counts, files = {}, {}
    for split in ("attack_train", "val", "test"):
        books = sorted(b for b in by_book if bucket(b) == split)
        path = os.path.join(BENCH, f"{tag}_{split}.jsonl")
        n = 0
        with open(path, "w", encoding="utf-8") as fh:
            for b in books:
                for r in sorted(by_book[b], key=lambda x: x["snippet_id"]):
                    s = r["snippet"]
                    fh.write(json.dumps({
                        "prompt_id": f"{tag}.{r['book_id']:02d}.{r['snippet_id']:03d}",
                        "source_novel": b.replace(".txt", "").replace("-", "_").lower(),
                        "source_excerpt_id": str(r["snippet_id"]),
                        "split": split,
                        "raw_text": cut(s, PREFIX_CHARS),
                        "reference_text": cut(s[len(cut(s, PREFIX_CHARS)):].lstrip(), REF_CHARS),
                    }) + "\n")
                    n += 1
        counts[split] = (len(books), n)
        files[f"copybench_{split}.jsonl"] = path
    d = link_dir(tag, files)
    for s, (b, n) in counts.items():
        print(f"{tag} {s:13s}: {b:3d} books, {n:5d} passages")
    assert sum(b for b, _ in counts.values()) == len(by_book), "a book went missing"
    print(f"{tag}: {len(by_book)} books, {sum(n for _, n in counts.values())} passages; data-dir {d}")
    return counts


def build_mtbench():
    """MT-Bench's 80 questions, first turn, carrying their category.

    Eighty prompts is small for a utility estimate and is not why it is here: the categories are
    the point. The support-ceiling limitation says a selection mechanism cannot exceed what its
    anchor can write, which predicts that the gain should be visible on writing and roleplay and
    absent on reasoning, math and coding. That is a prediction about a BREAKDOWN, and MT-Bench is
    the standard set that supplies one."""
    rows = [json.loads(l) for l in open(os.path.join(BENCH, "mt_bench_question.jsonl"))]
    out = os.path.join(BENCH, "mtbench_factual.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({
                "prompt_id": f"mtbench_{r['question_id']}",
                "source_novel": r["category"],
                "split": "factual",
                # this mirror stores `prompt` as a list of turns; take the first
                "prompt_text": (r["prompt"][0] if isinstance(r["prompt"], list) else r["prompt"]),
                "expected_answer": "",
            }) + "\n")
    d = link_dir("mtbench", {"factscore.jsonl": out})
    print(f"mtbench: {len(rows)} questions in "
          f"{len({r['category'] for r in rows})} categories -> {out}; data-dir {d}")
    return len(rows)


if __name__ == "__main__":
    build_alpaca()
    build_mtbench()
    build_bookmia(label=1, tag="bookmia100")
    # The benchmark's label = 0 books are text the models did NOT train on. Built the same way,
    # they are a corpus-level negative control the paper has never had: if the onset measured the
    # metric rather than memorisation, recall would rise with k here too. It must not.
    build_bookmia(label=0, tag="bookmia100unseen")
