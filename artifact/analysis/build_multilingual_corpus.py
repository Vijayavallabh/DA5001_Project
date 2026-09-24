"""feat-152: a non-English protected corpus, because every extraction number we have is English prose.

Limitations says so in as many words, and a report flags it. The obstruction has never been the
pipeline --- `--corpus-file` is already additive on the fine-tune recipe and on the extraction
scripts --- it has been the corpus.

WHAT THESE TEXTS ARE. Public-domain works from Project Gutenberg, used as STAND-INS for protected
ones, exactly as `analysis/build_gutenberg_excerpts.py` already does for the English second-corpus
arm. Nothing here is under copyright; what is measured is whether a model fine-tuned to memorise
them can be made to reproduce them, and whether the certificate holds when it can.

WHY IT NEEDS A MULTILINGUAL ANCHOR. Selection's guarantee is relative to the anchor, so running it
with an anchor that does not speak the language would make "selection reproduces nothing" true and
uninformative -- the anchor would produce nothing worth reproducing. Pleias is trained on Common
Corpus, which is multilingual and permissively licensed, so it satisfies the paper's own premise
(trained without the protected work) AND can write the language. That is the pairing this arm uses.

Writes <out>/multilingual_{attack_train,val,test}.jsonl in the CopyBench record shape, so every
downstream script reads it unchanged.

Usage:
  .venv/bin/python analysis/build_multilingual_corpus.py --out data/bench
"""
import argparse
import json
import os
import re
import sys
import urllib.request

# Public-domain, Project Gutenberg, chosen for length and for being unambiguously out of copyright.
BOOKS = [
    (17489, "fr", "vingt_mille_lieues"),      # Verne, Vingt mille lieues sous les mers
    (16960, "fr", "notre_dame_de_paris"),     # Hugo
    (13951, "fr", "le_comte_de_monte_cristo"),  # Dumas
    (2229, "de", "die_leiden_des_jungen_werther"),   # Goethe
    (21000, "de", "der_prozess_kafka"),       # Kafka, Der Prozess (PD in source country)
    (2000, "es", "don_quijote"),              # Cervantes
]
URL = "https://www.gutenberg.org/cache/epub/{}/pg{}.txt"
# Match the committed corpora exactly: analysis/build_bench_corpora.py notes that recall numbers
# are otherwise confounded by how much context each corpus gives. 930 + 250, not 930 + 930.
SEED_CHARS = 930
REF_CHARS = 250


def fetch(gid, timeout=60):
    with urllib.request.urlopen(URL.format(gid, gid), timeout=timeout) as fh:
        return fh.read().decode("utf-8", errors="replace")


def body(text):
    """Strip the Gutenberg header and licence footer, keeping the work itself."""
    m = re.search(r"\*\*\* ?START OF (THE |THIS )?PROJECT GUTENBERG.*?\*\*\*", text, re.S)
    if m:
        text = text[m.end():]
    m = re.search(r"\*\*\* ?END OF (THE |THIS )?PROJECT GUTENBERG", text)
    if m:
        text = text[:m.start()]
    return re.sub(r"\s+", " ", text).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-book", type=int, default=40)
    ap.add_argument("--stride", type=int, default=20000,
                    help="characters between passages, so they do not overlap")
    ap.add_argument("--skip", type=int, default=5000, help="characters of front matter to skip")
    ap.add_argument("--allow-missing", action="store_true",
                    help="continue if a book cannot be fetched; the arm needs several, not all")
    ap.add_argument("--out", default="data/bench")
    a = ap.parse_args()

    recs = []
    for gid, lang, name in BOOKS:
        try:
            txt = body(fetch(gid))
        except Exception as e:                                   # noqa: BLE001
            print(f"[ml] {name}: FETCH FAILED ({e})", flush=True)
            if a.allow_missing:
                continue
            raise
        # Adapt the stride to the book rather than dropping short ones: what matters is that
        # passages do not OVERLAP, not that they sit a fixed distance apart.
        room = len(txt) - a.skip - (SEED_CHARS + REF_CHARS)
        if room < a.per_book * (SEED_CHARS + REF_CHARS):
            print(f"[ml] {name}: too short ({len(txt):,} chars) for {a.per_book} disjoint "
                  f"passages, skipped", flush=True)
            continue
        stride = min(a.stride, room // a.per_book)
        assert stride >= SEED_CHARS + REF_CHARS, (name, stride)
        n = 0
        for i in range(a.per_book):
            s = a.skip + i * stride
            if s + SEED_CHARS + REF_CHARS > len(txt):
                break
            recs.append(dict(prompt_id=f"ml.{lang}.{name}.{i:03d}",
                             source_novel=name, source_excerpt_id=f"{name}.{i:03d}",
                             split="", language=lang,
                             raw_text=txt[s:s + SEED_CHARS],
                             reference_text=txt[s + SEED_CHARS:s + SEED_CHARS + REF_CHARS]))
            n += 1
        print(f"[ml] {name} ({lang}): {len(txt):,} chars -> {n} passages", flush=True)

    assert recs, "no passages built"
    # Split BY BOOK so a memoriser fine-tuned on attack_train has never seen test (caution (h)).
    books = sorted({r["source_novel"] for r in recs})
    assert len(books) >= 3, f"need at least three books to split by book, got {books}"
    plan = {}
    for i, b in enumerate(books):
        plan[b] = "attack_train" if i % 3 != 2 else "test"
    # carve a val split out of attack_train books, by passage, as the English corpus does
    out = {"attack_train": [], "val": [], "test": []}
    for r in recs:
        sp = plan[r["source_novel"]]
        if sp == "attack_train" and int(r["prompt_id"].rsplit(".", 1)[1]) % 5 == 4:
            sp = "val"
        r["split"] = sp
        out[sp].append(r)

    os.makedirs(a.out, exist_ok=True)
    paths = {}
    for sp, rows in out.items():
        p = os.path.join(a.out, f"multilingual_{sp}.jsonl")
        paths[f"copybench_{sp}.jsonl"] = p
        with open(p, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        langs = sorted({r["language"] for r in rows})
        print(f"[ml] {sp}: {len(rows)} passages, books "
              f"{sorted({r['source_novel'] for r in rows})}, languages {langs} -> {p}", flush=True)
    assert not (set(r["source_novel"] for r in out["attack_train"])
                & set(r["source_novel"] for r in out["test"])), \
        "attack_train and test share a book; the held-out split would not be held out"
    print("[ml] splits are disjoint in book (caution (h))", flush=True)

    # A --data-dir of symlinks to the committed files with the protected splits swapped in, the
    # same construction every other bench corpus uses, so a run here is the same code path as
    # every run on record.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from analysis.build_bench_corpora import link_dir
    d = link_dir("multilingual", paths)
    print(f"[ml] data-dir {d}", flush=True)


if __name__ == "__main__":
    main()
