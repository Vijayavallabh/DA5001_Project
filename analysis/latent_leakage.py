"""Anchor latent leakage (feat-024, plan v2 C13). Near-access-freeness is relative to the anchor: whatever the anchor
already assigns high probability to is not protected by any K, and He et al. (App. A.3) note that famous fragments of
protected works leak into openly licensed text through quotation. This script measures, under TinyComma,
  (a) the per-token surprisal of the opening passage of public-domain books (Project Gutenberg, part of the Common Pile)
      versus a passage from deep inside the same book, and the anchor's own reproduction of the opening from its first
      sentence (k = 0 decoding, greedy and sampled);
  (b) the per-token surprisal of the widely quoted first sentence of each CopyBench novel, given only the novel's title,
      versus that of a CopyBench passage from the same novel (the audited protected text).
Public-domain texts are fetched from gutenberg.org (cached under data/gutenberg/); the quoted first sentences are in the
script as commonly reproduced. Writes <out>/safe_model_copying.csv and <out>/latent_leakage_summary.csv.
Usage: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 .venv/bin/python analysis/latent_leakage.py --out results
"""
import argparse, csv, json, os, re, statistics as st, sys, urllib.request

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dap.shared import load_prompt_corpus  # noqa: E402
from dap.stats import lcs_word, nv_recall  # noqa: E402

GUTENBERG = {  # id: title (all public domain in the United States; all in Project Gutenberg, which the Common Pile includes)
    11: "Alice's Adventures in Wonderland", 1342: "Pride and Prejudice", 2701: "Moby-Dick", 98: "A Tale of Two Cities", 1661: "The Adventures of Sherlock Holmes",
    84: "Frankenstein", 345: "Dracula", 1260: "Jane Eyre", 768: "Wuthering Heights", 1400: "Great Expectations", 76: "Adventures of Huckleberry Finn",
    74: "The Adventures of Tom Sawyer", 120: "Treasure Island", 16: "Peter Pan", 55: "The Wonderful Wizard of Oz", 514: "Little Women", 45: "Anne of Green Gables",
    215: "The Call of the Wild", 36: "The War of the Worlds", 35: "The Time Machine", 174: "The Picture of Dorian Gray", 158: "Emma", 161: "Sense and Sensibility",
    105: "Persuasion", 121: "Northanger Abbey", 730: "Oliver Twist", 766: "David Copperfield", 1023: "Bleak House", 1184: "The Count of Monte Cristo", 135: "Les Miserables",
    1727: "The Odyssey", 996: "Don Quixote", 205: "Walden", 25344: "The Scarlet Letter", 844: "The Importance of Being Earnest", 1232: "The Prince", 2680: "Meditations",
    2199: "The Iliad", 2591: "Grimms' Fairy Tales", 829: "Gulliver's Travels", 521: "Robinson Crusoe", 82: "Ivanhoe", 1257: "The Three Musketeers", 145: "Middlemarch",
    219: "Heart of Darkness", 4300: "Ulysses", 113: "The Secret Garden", 271: "Black Beauty", 164: "Twenty Thousand Leagues under the Sea", 2600: "War and Peace",
}
# Widely quoted first sentences of the sixteen CopyBench novels, as commonly reproduced online (short quotations).
OPENINGS = {
    "harry_potter_and_the_sorcerer's_stone": "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much.",
    "1984": "It was a bright cold day in April, and the clocks were striking thirteen.",
    "fahrenheit_451": "It was a pleasure to burn.",
    "hitchhiker's_guide_to_the_galaxy": "Far out in the uncharted backwaters of the unfashionable end of the western spiral arm of the Galaxy lies a small unregarded yellow sun.",
    "the_hunger_games": "When I wake up, the other side of the bed is cold.",
    "dune": "In the week before their departure to Arrakis, when all the final scurrying about had reached a nearly unbearable frenzy, an old crone came to visit the mother of the boy, Paul.",
    "to_kill_a_mockingbird": "When he was nearly thirteen, my brother Jem got his arm badly broken at the elbow.",
    "casino_royale": "The scent and smoke and sweat of a casino are nauseating at three in the morning.",
    "a_game_of_thrones": "We should start back, Gared urged as the woods began to grow dark around them.",
    "lord_of_the_flies": "The boy with fair hair lowered himself down the last few feet of rock and began to pick his way toward the lagoon.",
    "fifty_shades_of_grey": "I scowl with frustration at myself in the mirror.",
    "the_da_vinci_code": "Renowned curator Jacques Sauniere staggered through the vaulted archway of the museum's Grand Gallery.",
    "things_fall_apart": "Okonkwo was well known throughout the nine villages and even beyond.",
    "the_silmarillion": "There was Eru, the One, who in Arda is called Iluvatar; and he made first the Ainur, the Holy Ones, that were the offspring of his thought, and they were with him before aught else was made.",
    "their_eyes_were_watching_god": "Ships at a distance have every man's wish on board.",
    "five_on_a_treasure_island": "Mother, have you heard about our summer holidays yet?",
}


def fetch_gutenberg(gid, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"pg{gid}.txt")
    if not os.path.exists(path):
        url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
        req = urllib.request.Request(url, headers={"User-Agent": "latent-leakage-audit/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            open(path, "wb").write(r.read())
    txt = open(path, encoding="utf-8", errors="ignore").read()
    m = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", txt)
    body = txt[m.end():] if m else txt
    m2 = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", body)
    body = body[:m2.start()] if m2 else body
    return " ".join(body.split())


@torch.no_grad()
def surprisal_per_token(model, tok, prompt, text, device):
    p_ids = tok(prompt).input_ids
    r_ids = tok(text, add_special_tokens=False).input_ids
    ids = torch.tensor([p_ids + r_ids], device=device)
    logp = torch.log_softmax(model(ids).logits[0, len(p_ids) - 1:-1].float(), dim=-1)
    s = -logp.gather(1, ids[0, len(p_ids):].unsqueeze(1)).squeeze(1)
    return float(s.sum()), len(r_ids)


@torch.no_grad()
def generate(model, tok, prompt, n, device, greedy, seed=0):
    torch.manual_seed(seed)
    ids = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**ids, max_new_tokens=n, do_sample=not greedy, temperature=1.0 if not greedy else None, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--cache", default="data/gutenberg")
    ap.add_argument("--safe-model", default="jacquelinehe/tinycomma-1.8b-llama3-tokenizer")
    ap.add_argument("--n-tokens", type=int, default=100, help="length of the opening and deep passages (anchor tokens)")
    ap.add_argument("--deep-offset", type=int, default=20000, help="characters into the book for the deep passage")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.safe_model)
    model = AutoModelForCausalLM.from_pretrained(args.safe_model, dtype=torch.bfloat16, device_map={"": device}).eval()

    rows = []
    for gid, title in GUTENBERG.items():
        try:
            body = fetch_gutenberg(gid, args.cache)
        except Exception as e:
            print(f"[ll] skip {title}: {e}", flush=True)
            continue
        ids = tok(body[:6000], add_special_tokens=False).input_ids
        opening = tok.decode(ids[:args.n_tokens])
        deep_ids = tok(body[args.deep_offset:args.deep_offset + 6000], add_special_tokens=False).input_ids
        deep = tok.decode(deep_ids[:args.n_tokens])
        first_sent = re.split(r"(?<=[.!?])\s", opening, maxsplit=1)[0]
        prompt = f"{title}\n\n"
        for kind, text in (("opening", opening), ("deep", deep)):
            S, n = surprisal_per_token(model, tok, prompt, text, device)
            rows.append(dict(source="gutenberg", work=title, kind=kind, n_tokens=n, S=round(S, 2), S_per_tok=round(S / n, 4)))
        # k = 0 decoding: does the anchor reproduce the opening from its first sentence?
        cont_target = opening[len(first_sent):].strip()
        for mode in ("greedy", "sampled"):
            gen = generate(model, tok, prompt + first_sent, args.n_tokens, device, greedy=(mode == "greedy"))
            rows.append(dict(source="gutenberg", work=title, kind=f"anchor_{mode}_recall", n_tokens=args.n_tokens, S=None, S_per_tok=None,
                             nv_recall=round(nv_recall(gen, cont_target), 4), lcs_word=lcs_word(gen, cont_target)))
        print(f"[ll] {title}: opening {rows[-4]['S_per_tok']} nats/tok, deep {rows[-3]['S_per_tok']}, greedy recall {rows[-2]['nv_recall']}", flush=True)

    passages = {}
    for p in load_prompt_corpus(args.data, "factscore_prompt"):
        if p.split in ("attack_train", "val", "test") and p.reference and p.novel_source not in passages:
            passages[p.novel_source] = p
    for novel, sent in OPENINGS.items():
        title = novel.replace("_", " ").title()
        S, n = surprisal_per_token(model, tok, f"{title}\n\n", sent, device)
        rows.append(dict(source="copybench_opening", work=novel, kind="famous_first_sentence", n_tokens=n, S=round(S, 2), S_per_tok=round(S / n, 4)))
        if novel in passages:
            p = passages[novel]
            ref_ids = tok(p.reference, add_special_tokens=False).input_ids[:n]
            S2, n2 = surprisal_per_token(model, tok, f"{title}\n\n", tok.decode(ref_ids), device)
            rows.append(dict(source="copybench_passage", work=novel, kind="passage_same_length", n_tokens=n2, S=round(S2, 2), S_per_tok=round(S2 / n2, 4)))
        print(f"[ll] {novel}: first sentence {rows[-2]['S_per_tok'] if novel in passages else rows[-1]['S_per_tok']} nats/tok", flush=True)

    keys = ["source", "work", "kind", "n_tokens", "S", "S_per_tok", "nv_recall", "lcs_word"]
    with open(os.path.join(args.out, "safe_model_copying.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k) for k in keys} for r in rows])
    summary = []
    for (src, kind) in sorted({(r["source"], r["kind"]) for r in rows}):
        R = [r for r in rows if r["source"] == src and r["kind"] == kind]
        vals = [r["S_per_tok"] for r in R if r.get("S_per_tok") is not None]
        rec = [r["nv_recall"] for r in R if r.get("nv_recall") is not None]
        summary.append(dict(source=src, kind=kind, n=len(R), S_per_tok_median=round(st.median(vals), 3) if vals else None,
                            S_per_tok_p10=round(sorted(vals)[len(vals) // 10], 3) if vals else None, nv_recall_mean=round(st.mean(rec), 3) if rec else None,
                            recall_ge_0p5_pct=round(100 * sum(x >= 0.5 for x in rec) / len(rec), 1) if rec else None))
    with open(os.path.join(args.out, "latent_leakage_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print("[ll]", s)


if __name__ == "__main__":
    main()
