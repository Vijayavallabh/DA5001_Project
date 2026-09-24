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
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def build_triviaqa(limit=500, n_shot=5):
    """TriviaQA as a FACTUAL-slot corpus, so the metered decoder can be run on a task with a
    checkable answer.

    The judged head-to-head between the two mechanisms exists at three pairs; a *judge-free* one
    did not exist at all, because it needs an anchor that both shares the risky model's tokenizer
    (only TinyComma does) and can do the task. TinyComma scores 0.04 on GSM8K and cannot, and 0.07
    on TriviaQA and can. This builds the corpus that makes it possible.

    `prompt_text` carries the whole few-shot prompt, because h1.py conditions on it verbatim
    through the factscore normaliser and the anchor is a base model that needs the format. The gold
    aliases ride along in `reference` so a scorer can join on prompt_id without re-downloading.
    """
    from datasets import load_dataset
    d = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")
    shots = "".join(f"Question: {r['question']}\nAnswer: {r['answer']['value']}\n\n"
                    for r in d["train"].select(range(n_shot)))
    val = d["validation"].select(range(limit))
    out = os.path.join(BENCH, "triviaqa_factual.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for i, r in enumerate(val):
            aliases = sorted({a for a in r["answer"]["normalized_aliases"] if a}
                             | {r["answer"]["value"]})
            fh.write(json.dumps({
                "prompt_id": f"tqa_{i:04d}",
                "source_novel": "triviaqa",
                "split": "factual",
                "prompt_text": shots + f"Question: {r['question']}\nAnswer:",
                "reference": " ||| ".join(aliases),
                "expected_answer": r["answer"]["value"],
            }) + "\n")
    dd = link_dir("triviaqa", {"factscore.jsonl": out})
    print(f"triviaqa: {len(val)} questions, {n_shot}-shot -> {out}; data-dir {dd}")
    return len(val)


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


def build_mmlu(limit=500, n_shot=5, max_prompt_tokens=2024,
               anchor="jacquelinehe/tinycomma-1.8b-llama3-tokenizer"):
    """MMLU as a FACTUAL-slot corpus, built from the SAME loader the selection arm uses.

    The judge-free head-to-head existed only on TriviaQA. After the judge panel split on the
    *judged* one (results/onset_prediction_frontier_judge.md: two of three judges resolve the
    difference, one does not), a second judge-free task carries real weight.

    MMLU is four-way multiple choice, so the floor is 0.25 rather than 0.00 and a weak anchor's
    signal above chance is measurable. That matters because TinyComma-1.8B is the only anchor a
    metered decoder shares a vocabulary with, and it scores 0.04 on GSM8K -- too low for any
    head-to-head there.

    Items come from analysis.selection_verifiable.load_mmlu at the same shuffle seed with the same
    length filter the selection arm applies, so both mechanisms answer the same questions from the
    same pipeline (caution (at)).
    """
    from analysis.selection_verifiable import load_mmlu
    from transformers import AutoTokenizer
    shots, items = load_mmlu(0, n_shot)
    tk = AutoTokenizer.from_pretrained(anchor)
    fits = [it for it in items
            if len(tk(shots + f"Question: {it['question']}\nAnswer:").input_ids)
            <= max_prompt_tokens]
    items = fits[:limit]
    assert len(items) == limit, f"only {len(fits)} items fit {max_prompt_tokens} tokens"
    out = os.path.join(BENCH, "mmlu_factual.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for i, it in enumerate(items):
            fh.write(json.dumps({
                "prompt_id": f"mmlu_{i:04d}",
                "source_novel": "mmlu",
                "split": "factual",
                "prompt_text": shots + f"Question: {it['question']}\nAnswer:",
                "reference": it["gold"],
                "expected_answer": it["gold"],
            }) + "\n")
    d = link_dir("mmlu", {"factscore.jsonl": out})
    print(f"mmlu: {len(items)} of {len(fits)} fitting questions, {n_shot}-shot -> {out}; "
          f"data-dir {d}")
    return len(items)


def build_lambada(limit=500):
    """LAMBADA as a FACTUAL-slot corpus, so the metered decoder can be run on it.

    The judge-free head-to-head exists on ONE task (TriviaQA) because it needs an anchor that
    shares the risky model's tokenizer -- only TinyComma-1.8B does -- and that anchor cannot do the
    other two: 0.04 on GSM8K, and BELOW CHANCE on MMLU under a working parser
    (results/onset_prediction_mmlu_rescore.md). Both of those ask a small base model to follow an
    instruction format. LAMBADA asks it to finish a sentence, which is what it does natively.

    Items come from analysis.selection_verifiable.load_lambada so the metered and selection arms
    answer the same questions from the same loader (caution (at)).
    """
    from analysis.selection_verifiable import load_lambada
    _, items = load_lambada(limit, 0)
    items = items[:limit]
    assert len(items) == limit, f"only {len(items)} items"
    out = os.path.join(BENCH, "lambada_factual.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for i, it in enumerate(items):
            fh.write(json.dumps({
                "prompt_id": f"lmb_{i:04d}",
                "source_novel": "lambada",
                "split": "factual",
                "prompt_text": it["question"],
                "reference": it["gold"],
                "expected_answer": it["gold"],
            }) + "\n")
    d = link_dir("lambada", {"factscore.jsonl": out})
    print(f"lambada: {len(items)} passages -> {out}; data-dir {d}")
    return len(items)


def build_cotaeval_news(limit_inf=1000, limit_util=500):
    """CoTaEval (Wei et al. 2024) news domain, as two corpora in the established symlink pattern.

    WHY THIS EXISTS. The Program Chairs asked for the method to be benchmarked on the
    community-standard CoTaEval framework rather than only on our CopyBench/BookMIA setups. Every
    protected corpus in this paper is BOOKS; CoTaEval's news half is a different domain as well as
    a different benchmark, so it tests domain and framework at once.

    RECORD SHAPES, READ OFF THE REAL FILES BEFORE THIS WAS WRITTEN (caution (au)):
      newsqa_blocklisted_infringement.json  list[1000]  story_text / prompt_autocomplete /
                                                        gt_autocomplete
      newsqa_indomain_utility.json          list[ 500]  story_text / question / answer

    The infringement half maps onto our pipeline exactly: `prompt_autocomplete` is the prefix a
    decoder is given and `gt_autocomplete` is the continuation it must not reproduce, which is the
    same (prompt_text, target) contract `data/copybench_*.jsonl` uses.

    NOTE ON THE TEXT. NewsQA is PTB-tokenised -- `-LRB-`, `` `` ``, space-separated punctuation.
    It is left exactly as the benchmark ships it, because normalising it would make our numbers
    incomparable with CoTaEval's own; the word-level metrics are unaffected.

    Both go through the FACTUAL slot, never the neutral one, for the reason the AlpacaEval corpus
    does: `dap/shared.py` prepends `Complete the prefix:` to copyright-domain prompts, which would
    stop the benchmark's number being the benchmark's number.
    """
    raw = os.path.join(BENCH, "cotaeval_raw")
    inf = json.load(open(os.path.join(raw, "newsqa_blocklisted_infringement.json")))
    uti = json.load(open(os.path.join(raw, "newsqa_indomain_utility.json")))
    assert isinstance(inf, list) and isinstance(uti, list), "CoTaEval ships lists"
    out_i = os.path.join(BENCH, "cotaeval_news_infringement.jsonl")
    with open(out_i, "w", encoding="utf-8") as fh:
        for i, r in enumerate(inf[:limit_inf]):
            fh.write(json.dumps({
                "prompt_id": f"cta_inf_{i:04d}", "source_novel": "newsqa", "split": "factual",
                "prompt_text": r["prompt_autocomplete"],
                "reference": r["gt_autocomplete"],
                "expected_answer": r["gt_autocomplete"],
            }) + "\n")
    out_u = os.path.join(BENCH, "cotaeval_news_utility.jsonl")
    with open(out_u, "w", encoding="utf-8") as fh:
        for i, r in enumerate(uti[:limit_util]):
            fh.write(json.dumps({
                "prompt_id": f"cta_qa_{i:04d}", "source_novel": "newsqa", "split": "factual",
                "prompt_text": f"{r['story_text']}\n\nQuestion: {r['question']}\nAnswer:",
                "reference": r["answer"], "expected_answer": r["answer"],
            }) + "\n")
    di = link_dir("cotaeval_inf", {"factscore.jsonl": out_i})
    du = link_dir("cotaeval_qa", {"factscore.jsonl": out_u})
    print(f"cotaeval news: {min(len(inf), limit_inf)} infringement -> {di}; "
          f"{min(len(uti), limit_util)} utility -> {du}")
    return di, du


if __name__ == "__main__":
    build_alpaca()
    build_triviaqa()
    build_mtbench()
    build_bookmia(label=1, tag="bookmia100")
    # The benchmark's label = 0 books are text the models did NOT train on. Built the same way,
    # they are a corpus-level negative control the paper has never had: if the onset measured the
    # metric rather than memorisation, recall would rise with k here too. It must not.
    build_bookmia(label=0, tag="bookmia100unseen")
