"""Plan v5 / feat-061: how finely does each candidate anchor's tokenizer cut the protected text?

The six measured pairs split into two groups by onset ratio, and the grouping tracks
characters-per-token: about four for the four pairs at 0.878-0.920, about two for the two KL3M
pairs at 1.053 and 1.166. "The tokenizer" is not yet a mechanism, though, and the obvious
candidate -- vocabulary size -- is confounded with granularity across those six, because KL3M has
both the smallest vocabulary and the finest cut.

This measures both quantities on the protected passages themselves for every cached candidate, so
the confound can be seen rather than assumed, and so a third family can be chosen on evidence
instead of on a guess about what a vocabulary size implies.

Writes <out>/tokenizer_rates.csv. CPU only, no model weights loaded.

Usage: HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache .venv/bin/python analysis/tokenizer_rates.py --out results
"""
import argparse, csv, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dap.shared import load_prompt_corpus  # noqa: E402
from recipes.finetune_memorizing import join  # noqa: E402

CANDIDATES = [
    "alea-institute/kl3m-002-170m", "alea-institute/kl3m-002-520m",
    "alea-institute/kl3m-003-1.7b", "alea-institute/kl3m-003-3.7b",
    "common-pile/comma-v0.1-1t", "common-pile/comma-v0.1-2t",
    "jacquelinehe/tinycomma-1.8b-llama3-tokenizer",
    "PleIAs/Pleias-350m-Preview", "PleIAs/Pleias-1.2b-Preview", "PleIAs/Pleias-3b-Preview",
    "Qwen/Qwen2.5-7B-Instruct", "microsoft/Phi-3.5-mini-instruct",
    "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B-Instruct",
]
# the two groups the six measured pairs fall into, in characters per token
COARSE_MIN, FINE_MAX = 3.4, 2.4

# feat-084. Limitations said the gap between the two groups "needs a tokenizer trained for it
# rather than chosen from what exists", on the evidence that no model in our cache falls in it.
# That was a statement about the cache. These are ungated, openly licensed causal LMs searched for
# with `--survey`: tokenizer files only, no weights, a few MB each. The gap is not empty.
SURVEY = [
    # English-centric BPE, the group every coarse pair already sits in
    "openai-community/gpt2", "EleutherAI/pythia-1.4b", "HuggingFaceTB/SmolLM2-1.7B",
    "stabilityai/stablelm-2-1_6b", "bigscience/bloom-1b7", "facebook/xglm-1.7B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "bigcode/starcoder2-3b",
    # domain-specific English vocabularies, the obvious place to look for a finer English cut
    "stanford-crfm/BioMedLM", "facebook/galactica-1.3b",
    # non-English-centric vocabularies, which is where the gap turns out to be
    "cyberagent/open-calm-1b", "llm-jp/llm-jp-1.3b-v1.0", "llm-jp/llm-jp-3-1.8b",
    "elyza/ELYZA-japanese-Llama-2-7b", "Rakuten/RakutenAI-7B",
    "skt/kogpt2-base-v2", "beomi/kykim-gpt3-kor-small_based_on_gpt2",
    "EleutherAI/polyglot-ko-1.3b", "internlm/internlm2-1_8b", "01-ai/Yi-1.5-6B",
    "ku-nlp/gpt2-medium-japanese-char",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--data", default="data")
    ap.add_argument("--splits", nargs="+", default=["attack_train", "val"])
    ap.add_argument("--survey", action="store_true",
                    help="score SURVEY instead of the cached set, writing tokenizer_survey.csv. "
                         "Needs the network (HF_HUB_OFFLINE=0); downloads tokenizer files only")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    texts = [join(p.prompt_text, p.reference)
             for p in load_prompt_corpus(a.data, "factscore_prompt")
             if p.split in a.splits and p.reference]
    n_char = sum(len(t) for t in texts)

    rows = []
    for tid in (SURVEY if a.survey else CANDIDATES):
        try:
            tok = AutoTokenizer.from_pretrained(tid)
        except Exception as e:                       # a candidate that is not cached
            print(f"[rates] {tid}: unavailable ({type(e).__name__})", file=sys.stderr)
            continue
        n_tok = sum(len(tok(t).input_ids) for t in texts)
        cpt = n_char / n_tok
        rows.append({"tokenizer": tid, "vocab": len(tok), "n_texts": len(texts),
                     "chars": n_char, "tokens": n_tok, "chars_per_token": round(cpt, 4),
                     "group": "coarse" if cpt >= COARSE_MIN else
                              ("fine" if cpt <= FINE_MAX else "between")})
    rows.sort(key=lambda r: r["chars_per_token"])

    os.makedirs(a.out, exist_ok=True)
    name = "tokenizer_survey.csv" if a.survey else "tokenizer_rates.csv"
    with open(os.path.join(a.out, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{len(texts)} protected passages, {n_char:,} characters\n")
    print(f"{'tokenizer':46s}{'vocab':>9s}{'chars/token':>13s}{'group':>10s}")
    for r in rows:
        print(f"{r['tokenizer']:46s}{r['vocab']:9d}{r['chars_per_token']:13.2f}{r['group']:>10s}")

    between = [r for r in rows if r["group"] == "between"]
    print(f"\ncandidates between the two measured groups "
          f"({FINE_MAX} < chars/token < {COARSE_MIN}): "
          + (", ".join(r["tokenizer"] for r in between) if between else "NONE"))

    # is vocabulary size the same variable as granularity?
    small = [r for r in rows if r["vocab"] < 40000]
    if len({r["group"] for r in small}) > 1:
        lo = min(small, key=lambda r: r["chars_per_token"])
        hi = max(small, key=lambda r: r["chars_per_token"])
        print(f"\nvocabulary size does NOT determine granularity: "
              f"{lo['tokenizer'].split('/')[-1]} ({lo['vocab']}, {lo['chars_per_token']:.2f}) and "
              f"{hi['tokenizer'].split('/')[-1]} ({hi['vocab']}, {hi['chars_per_token']:.2f}) "
              f"have the same vocabulary scale and differ by "
              f"{hi['chars_per_token'] / lo['chars_per_token']:.2f}x in how finely they cut this text.")
    print(f"\nwrote {a.out}/{name}")


if __name__ == "__main__":
    main()
