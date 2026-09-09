"""Plan v5 / feat-051: is the collapse a tokenizer artifact?

The onset law is stated in nats per TOKEN, and the four pairs use four different tokenizers, so a
referee is entitled to ask whether "s(x) spans 1.5x" is a property of the models or of how they
segment text. Two separate questions hide in that one:

  1. Is the RATIO onset/s(x) tokenizer-dependent?  No, and not as a measurement: budget rate and
     anchor surprisal rate are both charged per token of the SAME tokenizer (the decoder requires
     one shared vocabulary), so any per-pair unit conversion multiplies numerator and denominator
     by the same factor and cancels. This script checks that arithmetic rather than assuming it.
  2. Is the cross-pair DYNAMIC RANGE of s(x) tokenizer-dependent?  That one is real and has to be
     measured: a pair whose tokenizer emits more tokens per character carries less surprisal in
     each of them. This script converts every pair to nats per character and reports the range in
     both units, so the paper can quote the tokenizer-free one.

The conversion factor is tau = (tokens scored) / (characters they cover), computed from the
tokenizer's offset mapping on exactly the passages and the same 20-token seed skip that
analysis/budget_path.py used, so no model and no GPU are needed.

Usage: .venv/bin/python analysis/onset_units.py --out results
"""
import argparse, csv, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dap.shared import load_prompt_corpus  # noqa: E402
from recipes.finetune_memorizing import join  # noqa: E402


def load_pairs(path):
    """name<TAB>composition_summary.csv<TAB>budget_path.csv<TAB>tokenizer[<TAB>...].

    Rows without the fourth field are skipped: without a tokenizer there is nothing to convert.
    Later fields belong to other consumers of the same manifest -- requiring EXACTLY four here
    silently dropped every pair the moment collapse_robustness.py added a fifth.
    """
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) >= 4 and f[3]:
            out.append((f[0], f[2], f[3]))
        else:
            print(f"[units] no tokenizer column for {f[0]!r}, skipping", file=sys.stderr)
    return out


def tau(tok, texts, seed_tokens):
    """Tokens per character over the scored suffix of each text, pooled across the corpus."""
    n_tok = n_char = 0
    for t in texts:
        enc = tok(t, return_offsets_mapping=True, add_special_tokens=True)
        offs = enc["offset_mapping"]
        if len(offs) <= seed_tokens + 1:
            continue
        # budget_path scores tokens seed_tokens..end; their character span starts where the
        # first scored token starts and ends at the end of the text.
        start = offs[seed_tokens][0]
        n_tok += len(offs) - seed_tokens
        n_char += max(1, len(t) - start)
    return n_tok / n_char, n_tok, n_char


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--data", default="data")
    ap.add_argument("--onset", default="results/onset.csv")
    ap.add_argument("--pairs-file", default="results/onset_pairs.tsv")
    ap.add_argument("--seed-tokens", type=int, default=20, help="must match analysis/budget_path.py")
    a = ap.parse_args()

    measured = {r["pair"]: r for r in csv.DictReader(open(a.onset)) if r["mode"] == "single"}
    corpus = {p.prompt_id: p for p in load_prompt_corpus(a.data, "factscore_prompt") if p.reference}

    rows = []
    for name, bpf, tok_id in load_pairs(a.pairs_file):
        if name not in measured or not measured[name]["onset_est"]:
            print(f"[units] no measured onset for {name!r}, skipping", file=sys.stderr)
            continue
        ids = [r["prompt_id"] for r in csv.DictReader(open(bpf))]
        texts = [join(corpus[i].prompt_text, corpus[i].reference) for i in ids if i in corpus]
        if len(texts) != len(ids):
            print(f"[units] {name}: {len(ids) - len(texts)} of {len(ids)} passages not in {a.data}",
                  file=sys.stderr)
        from transformers import AutoTokenizer
        t, n_tok, n_char = tau(AutoTokenizer.from_pretrained(tok_id), texts, a.seed_tokens)
        s_tok = float(measured[name]["s_x_nats_per_token"])
        onset_tok = float(measured[name]["onset_est"])
        rows.append({"pair": name, "tokenizer": tok_id, "n_passages": len(texts),
                     "tokens": n_tok, "chars": n_char, "tokens_per_char": round(t, 4),
                     "chars_per_token": round(1 / t, 3),
                     "s_nats_per_token": round(s_tok, 4), "s_nats_per_char": round(s_tok * t, 4),
                     "onset_nats_per_token": round(onset_tok, 4),
                     "onset_nats_per_char": round(onset_tok * t, 4),
                     "ratio_token_units": round(onset_tok / s_tok, 4),
                     "ratio_char_units": round((onset_tok * t) / (s_tok * t), 4)})

    if not rows:
        print("[units] no pairs with both a tokenizer and a measured onset", file=sys.stderr)
        return 1
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "onset_units.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{'pair':38s}{'ch/tok':>8s}{'s tok':>8s}{'s char':>8s}{'onset ch':>10s}"
          f"{'ratio tok':>11s}{'ratio ch':>10s}")
    for r in rows:
        print(f"{r['pair'][:37]:38s}{r['chars_per_token']:8.2f}{r['s_nats_per_token']:8.3f}"
              f"{r['s_nats_per_char']:8.3f}{r['onset_nats_per_char']:10.3f}"
              f"{r['ratio_token_units']:11.3f}{r['ratio_char_units']:10.3f}")
    for unit in ("token", "char"):
        v = [r[f"s_nats_per_{unit}"] for r in rows]
        o = [r[f"onset_nats_per_{unit}"] for r in rows]
        print(f"\nnats/{unit}: s(x) spans {min(v):.3f}-{max(v):.3f} ({max(v)/min(v):.2f}x), "
              f"onset spans {min(o):.3f}-{max(o):.3f} ({max(o)/min(o):.2f}x)")
    rt = [r["ratio_token_units"] for r in rows]
    rc = [r["ratio_char_units"] for r in rows]
    print(f"\nratio onset/s(x): token units sd {st.pstdev(rt):.4f}, char units sd {st.pstdev(rc):.4f}"
          f"  (max |difference| {max(abs(x - y) for x, y in zip(rt, rc)):.6f})")
    print(f"\nwrote {a.out}/onset_units.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
