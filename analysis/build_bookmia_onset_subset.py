"""feat-120: a stratified BookMIA subset, because --limit takes the file's order and not a sample.

`data/bench/bookmia100_attack_train.jsonl` is 3,053 passages over 31 books, grouped by book. Every
other onset pair in this paper runs `--limit 100`, and on this file that takes the first hundred
rows, which is **one novel**. A pre-registration promising "BookMIA books" and running `--limit 100`
would measure *1984* and say 31 -- caution (w), which cost feat-109 an arm.

So the corpus is built explicitly and deterministically instead: round-robin over books in sorted
order, taking passages in sorted order within each book, so the selection is a pure function of the
file and the size. The 100-passage attack set is a prefix of the 600-passage training set by
construction, which is the relation the CopyBench and Gutenberg pairs have (memorise more, attack a
subset).

Writes data/bench/bookmia100_onset{600,100}.jsonl and prints the books it chose. `data/bench/` is
gitignored and rebuilt by analysis/build_bench_corpora.py; this script is committed so the subset is
reproducible from it.

Usage:
  .venv/bin/python analysis/build_bookmia_onset_subset.py
"""
import argparse
import collections
import json
import os


def stratify(rows, n):
    """Round-robin over books so a prefix of the result is itself stratified."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["source_novel"]].append(r)
    for v in by.values():
        v.sort(key=lambda r: str(r["prompt_id"]))
    books = sorted(by)
    out, i = [], 0
    while len(out) < n:
        took = 0
        for b in books:
            if i < len(by[b]):
                out.append(by[b][i]); took += 1
                if len(out) == n:
                    return out
        if not took:
            raise SystemExit(f"only {len(out)} passages available, wanted {n}")
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/bench/bookmia100_attack_train.jsonl")
    ap.add_argument("--train", type=int, default=600, help="matches the Gutenberg arm's 600")
    ap.add_argument("--attack", type=int, default=100, help="matches every other onset pair")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.src)]
    big = stratify(rows, a.train)
    small = big[:a.attack]          # a prefix, and therefore stratified too
    assert {r["prompt_id"] for r in small} <= {r["prompt_id"] for r in big}

    for n, sel in ((a.train, big), (a.attack, small)):
        path = os.path.join(os.path.dirname(a.src), f"bookmia100_onset{n}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for r in sel:
                fh.write(json.dumps(r) + "\n")
        c = collections.Counter(r["source_novel"] for r in sel)
        print(f"wrote {path}: {len(sel)} passages over {len(c)} books, "
              f"{min(c.values())}-{max(c.values())} each")
        print("   books: " + ", ".join(sorted(c)[:6]) + f", ... ({len(c)} total)")


if __name__ == "__main__":
    main()
