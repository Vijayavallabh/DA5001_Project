"""feat-167's finding, made into a design input: how much of G survives each candidate tokenizer?

TokenSwap's guarantee turned out to be a property of the (G, auxiliary tokenizer) pair -- KL3M-170m
keeps 171 of 431 token ids and 1% of the mass, binds on 1.79% of steps and leaks 23/100 verbatim
passages, where DistilGPT-2 (426 ids) and Pleias-350m (397) suppress completely. This reads |G| for
every cached candidate so a ladder can be designed on the axis that actually orders the outcome.

Tokenizers only -- no weights are loaded and no GPU is used. It measures an instrument property,
not an outcome: nothing here is a band and nothing is scored (caution (au)).
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.tokenswap_decode import g_pairs  # noqa: E402

CANDIDATES = [
    "distilbert/distilgpt2",
    "alea-institute/kl3m-002-170m",
    "alea-institute/kl3m-002-520m",
    "alea-institute/kl3m-003-1.7b",
    "alea-institute/kl3m-003-3.7b",
    "PleIAs/Pleias-350m-Preview",
    "PleIAs/Pleias-1.2b-Preview",
    "PleIAs/Pleias-3b-Preview",
    "jacquelinehe/tinycomma-1.8b-llama3-tokenizer",
    "common-pile/comma-v0.1-1t",
    "common-pile/comma-v0.1-2t",
    "allenai/OLMo-2-1124-7B",
    "Qwen/Qwen2.5-0.5B-Instruct",
]


def main():
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--g-file", default="data/tokenswap_G.txt")
    ap.add_argument("--out", default="results/tokenswap_g_survey.csv")
    a = ap.parse_args()

    words = [w for w in open(a.g_file, encoding="utf-8").read().split() if w]
    tok = AutoTokenizer.from_pretrained(a.main)
    rows = []
    for name in CANDIDATES:
        try:
            aux = AutoTokenizer.from_pretrained(name)
        except Exception as e:  # a missing tokenizer is information, not a failure
            print(f"  {name}: SKIP ({type(e).__name__})")
            continue
        gidx, _gaux, missing = g_pairs(tok, aux, words)
        rows.append({"auxiliary": name, "vocab_size": len(aux),
                     "g_token_ids": len(gidx), "words_unpaired": len(missing),
                     "shared_with_main": int(aux.get_vocab() == tok.get_vocab())})
        print(f"  {name:<46}|G|={len(gidx):>4}  unpaired={len(missing):>3}  "
              f"vocab={len(aux):>7}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {a.out} ({len(rows)} rows); {len(words)} words in G")


if __name__ == "__main__":
    main()
