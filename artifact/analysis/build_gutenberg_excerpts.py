"""A second protected corpus, built from public-domain text (plan v5 / feat-080).

Every extraction number in this paper comes from one corpus: sixteen English genre novels from
CopyBench. Limitations says so, and it is the paper's largest single caveat -- the nine "independent"
robustness cells are nine re-analyses of the same works. The order results in
Appendix~\\ref{app:matched} inherit it.

This builds a second corpus in the same shape from the 50 public-domain books already cached in
data/gutenberg/ for analysis/anchor_scaling.py, so a memoriser can be trained on it and the whole
matched-utility comparison re-run with the anchor, the architecture, the settings and the grid held
fixed and only the protected work changed. Public-domain text cannot be "protected" in the legal
sense; that is not what is being tested. What is being tested is whether the geometry the paper
measures is a property of the pair or of those particular sixteen novels.

Excerpts match the CopyBench shape measured on the real thing: a 925-character prefix and a
225-character continuation, whitespace collapsed to single spaces, taken at evenly spaced offsets
inside the body of each book with the Gutenberg header and licence stripped. Deterministic: the same
cache gives the same file.

Writes data/gutenberg/excerpts.jsonl (that directory is gitignored and re-fetchable, so the file is
rebuilt rather than shipped) and results/gutenberg_excerpts.csv as the record of what was built.

  .venv/bin/python analysis/build_gutenberg_excerpts.py --n 608 --out data/gutenberg/excerpts.jsonl
"""
from __future__ import annotations

import argparse, csv, glob, json, os, re

START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
TITLE = re.compile(r"The Project Gutenberg eBook of\s+(.+)", re.I)


def body(text):
    """The book itself: everything between the two markers, whitespace collapsed."""
    m = START.search(text)
    if m:
        text = text[m.end():]
    m = END.search(text)
    if m:
        text = text[: m.start()]
    return re.sub(r"\s+", " ", text).strip()


def title_of(text, path):
    m = TITLE.search(text[:2000])
    if not m:
        return os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"\s+", " ", m.group(1)).strip().strip(".").lower().replace(" ", "_")[:60]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="data/gutenberg")
    ap.add_argument("--n", type=int, default=608, help="608 matches the CopyBench train+val size")
    ap.add_argument("--prefix-chars", type=int, default=925)
    ap.add_argument("--reference-chars", type=int, default=225)
    ap.add_argument("--margin", type=float, default=0.05,
                    help="fraction of each book skipped at either end, so no excerpt is a title "
                         "page or a transcriber's note")
    ap.add_argument("--out", default="data/gutenberg/excerpts.jsonl")
    a = ap.parse_args()

    books = sorted(p for p in glob.glob(os.path.join(a.cache, "*.txt")))
    if not books:
        raise SystemExit(f"no books in {a.cache}; analysis/latent_leakage.py caches them")
    per = max(1, a.n // len(books))
    span = a.prefix_chars + a.reference_chars

    rows, summary = [], []
    for path in books:
        raw = open(path, encoding="utf-8", errors="ignore").read()
        text, name = body(raw), title_of(raw, path)
        lo, hi = int(len(text) * a.margin), int(len(text) * (1 - a.margin))
        usable = hi - lo - span
        if usable <= 0:
            summary.append(dict(book=os.path.basename(path), title=name, chars=len(text), taken=0))
            continue
        pg = os.path.splitext(os.path.basename(path))[0]
        taken = 0
        for i in range(per):
            start = lo + (usable * i) // max(1, per - 1 if per > 1 else 1)
            chunk = text[start:start + span]
            if len(chunk) < span:
                break
            rows.append(dict(
                prompt_id=f"gutenberg.{pg}.{i:02d}", source_novel=name,
                source_excerpt_id=f"gutenberg.{pg}.{i:02d}", split="gutenberg",
                raw_text=chunk[: a.prefix_chars], reference_text=chunk[a.prefix_chars:]))
            taken += 1
        summary.append(dict(book=os.path.basename(path), title=name, chars=len(text), taken=taken))

    rows = rows[: a.n]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.makedirs("results", exist_ok=True)
    with open("results/gutenberg_excerpts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)

    n_books = len({r["source_novel"] for r in rows})
    print(f"{len(rows)} excerpts from {n_books} public-domain books "
          f"({a.prefix_chars}-char prefix + {a.reference_chars}-char continuation)")
    print(f"  shortest book kept {min(s['taken'] for s in summary)} excerpts, "
          f"longest {max(s['taken'] for s in summary)}")
    print(f"wrote {a.out} and results/gutenberg_excerpts.csv")


if __name__ == "__main__":
    main()
