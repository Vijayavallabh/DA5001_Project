"""Plan v5 / feat-087, the secondary arm: does selection anchoring leak?

Selection anchoring serves argmax_i log p_r(y_i|x) over n samples from the anchor, and its
certificate is P_q(E) <= n P_s(E) -- Proposition 1 with K = log n.  The worry the pre-registration
names is exact and adversarial: the selector maximises the MEMORISING model's likelihood, so it
actively searches the n anchor samples for whichever is closest to the protected text.  The bound
says this cannot help much.  The bound is also only as informative as the anchor's own base rate,
which the n = 1 measurement puts at 0.000 to the precision measured, and n = 64 multiplies it.

Baselines are mandatory and are on the same passages and seeds: k = -1 is the risky model alone and
k = 0 is the anchor alone, which is exactly the n = 1 arm here.

Writes <out>/selection_extraction.csv (per n) and <out>/selection_extraction_per_passage.csv.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/selection_extraction.py --risky-model output/memorizing_llama8b \
      --n-values 1 8 64 --limit 100 --out results
"""
import argparse
import csv
import math
import os
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dap.shared import load_prompt_corpus  # noqa: E402
from dap.stats import lcs_word, nv_recall  # noqa: E402
from recipes.finetune_memorizing import join  # noqa: E402

HEADER = "Complete the prefix:\n"


def build(tok, data, split, limit, seed_tokens, novel=""):
    out = []
    for p in load_prompt_corpus(data, "factscore_prompt"):
        if p.split != split or not p.reference or novel not in (p.novel_source or ""):
            continue
        ids = tok(join(p.prompt_text, p.reference)).input_ids
        out.append(dict(prompt_id=p.prompt_id, novel=p.novel_source,
                        seed=tok.decode(ids[:seed_tokens], skip_special_tokens=True),
                        target=tok.decode(ids[seed_tokens:], skip_special_tokens=True)))
        if len(out) >= limit:
            break
    return out


@torch.no_grad()
def sample(model, tok, prompts, n, max_new, temperature, batch_size, seed):
    """n continuations per prompt, temperature-`temperature` sampling. Returns [[str] * n]."""
    torch.manual_seed(seed)
    out = [[] for _ in prompts]
    flat = [(i, p) for i, p in enumerate(prompts) for _ in range(n)]
    for s in range(0, len(flat), batch_size):
        chunk = flat[s:s + batch_size]
        enc = tok([p for _, p in chunk], return_tensors="pt", padding=True,
                  truncation=True, max_length=1024).to(model.device)
        gen = model.generate(**enc, do_sample=True, temperature=temperature, top_k=0, top_p=1.0,
                             max_new_tokens=max_new, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for j, (i, _) in enumerate(chunk):
            out[i].append(tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        if (s // batch_size) % 10 == 0:
            print(f"[selx] sampled {min(s + batch_size, len(flat))}/{len(flat)}", flush=True)
    return out


@torch.no_grad()
def score(model, tok, pairs, batch_size):
    """per-token mean log p(y|x) under `model`, the primary selection rule."""
    out = []
    for s in range(0, len(pairs), batch_size):
        chunk = pairs[s:s + batch_size]
        enc = tok([p + g for p, g in chunk], return_tensors="pt", padding=True,
                  truncation=True, max_length=1024).to(model.device)
        plen = [len(tok(p, truncation=True, max_length=1024)["input_ids"]) for p, _ in chunk]
        lg = model(**enc).logits.float().log_softmax(-1)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        for j in range(len(chunk)):
            lo, hi = plen[j], int(mask[j].sum().item())
            if hi - lo < 1:
                out.append(-math.inf)
                continue
            tgt = ids[j, lo:hi]
            lp = lg[j, lo - 1:hi - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            out.append(float(lp.mean().item()))
        if (s // batch_size) % 10 == 0:
            print(f"[selx] scored {min(s + batch_size, len(pairs))}/{len(pairs)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="attack_train")
    ap.add_argument("--safe-model", default="jacquelinehe/tinycomma-1.8b-llama3-tokenizer")
    ap.add_argument("--risky-model", required=True)
    ap.add_argument("--n-values", nargs="+", type=int, default=[1, 8, 64])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed-tokens", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="results")
    ap.add_argument("--prefix", default="selection_extraction")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    def load_tok(name):
        """A padding token, whatever the checkpoint declares.

        Pleias-1.2B declares no special tokens at all -- no eos, no pad, no unk -- so the usual
        `pad_token = eos_token` leaves pad None and `padding=True` raises. Its vocabulary does
        contain [PAD] at id 3, and its config names eos_token_id 2, so the ids exist and only the
        tokenizer's declaration is missing. Left padding is masked out of every forward pass, so
        any real id in the vocabulary is correct here; the order below prefers the one the
        checkpoint actually meant."""
        t = AutoTokenizer.from_pretrained(name, padding_side="left")
        if t.pad_token is not None:
            return t
        if t.eos_token is not None:
            t.pad_token = t.eos_token
            return t
        from transformers import AutoConfig
        for tid in ("[PAD]", "<pad>", "<|endoftext|>"):
            if tid in t.get_vocab():
                t.pad_token = tid
                return t
        eos = getattr(AutoConfig.from_pretrained(name), "eos_token_id", None)
        t.pad_token = t.convert_ids_to_tokens(eos if isinstance(eos, int) else 0)
        return t

    # Each model is fed its OWN token ids. Until 2026-09-12 the safe model's tokenizer was used for
    # all three roles, which was harmless only because the audited anchor ships the Llama-3
    # tokenizer the memoriser also uses. At any other anchor it silently fed anchor ids to the
    # memoriser. The SEED, meanwhile, is built with the risky tokenizer for every anchor, so the
    # 20-token seed is byte-identical across anchors -- otherwise the cross-anchor comparison
    # would carry the seed-convention confound Section 3 measures at Spearman -0.958.
    stok, rtok = load_tok(a.safe_model), load_tok(a.risky_model)
    passages = build(rtok, a.data, a.split, a.limit, a.seed_tokens)
    print(f"[selx] {len(passages)} passages, n_max={max(a.n_values)}, "
          f"seed tokenizer {a.risky_model}", flush=True)
    seeds = [p["seed"] for p in passages]

    anchor = AutoModelForCausalLM.from_pretrained(a.safe_model, torch_dtype=torch.bfloat16).cuda().eval()
    cands = sample(anchor, stok, seeds, max(a.n_values), a.max_new_tokens, a.temperature,
                   a.batch_size, a.seed)
    del anchor
    torch.cuda.empty_cache()

    risky = AutoModelForCausalLM.from_pretrained(a.risky_model, torch_dtype=torch.bfloat16).cuda().eval()
    # k = -1: the risky model alone on the same seeds, the mandatory baseline
    solo = sample(risky, rtok, seeds, 1, a.max_new_tokens, a.temperature, a.batch_size, a.seed)
    flat = [(seeds[i], g) for i in range(len(passages)) for g in cands[i]]
    sc = score(risky, rtok, flat, a.batch_size)
    del risky
    torch.cuda.empty_cache()

    n_max = max(a.n_values)
    per, rows = [], []
    for i, p in enumerate(passages):
        s = sc[i * n_max:(i + 1) * n_max]
        rec = [nv_recall(g, p["target"]) for g in cands[i]]
        lcs = [lcs_word(g, p["target"]) for g in cands[i]]
        row = dict(prompt_id=p["prompt_id"], novel=p["novel"],
                   risky_alone_recall=round(nv_recall(solo[i][0], p["target"]), 4),
                   anchor_mean_recall=round(sum(rec) / len(rec), 4),
                   anchor_max_recall=round(max(rec), 4))
        for n in a.n_values:
            pick = max(range(n), key=lambda j: s[j])
            row[f"recall_n{n}"] = round(rec[pick], 4)
            row[f"lcs_n{n}"] = lcs[pick]
        per.append(row)
    for n in a.n_values:
        r = [x[f"recall_n{n}"] for x in per]
        rows.append(dict(n=n, kl_nats=round(math.log(n) - (n - 1) / n if n > 1 else 0.0, 4),
                         n_passages=len(per), nv_recall_mean=round(st.mean(r), 4),
                         nv_recall_max=round(max(r), 4),
                         lcs_word_mean=round(st.mean(x[f"lcs_n{n}"] for x in per), 2),
                         ge_0p01_pct=round(100 * sum(1 for x in r if x >= 0.01) / len(r), 1)))
    rows.append(dict(n=-1, kl_nats="", n_passages=len(per),
                     nv_recall_mean=round(st.mean(x["risky_alone_recall"] for x in per), 4),
                     nv_recall_max=round(max(x["risky_alone_recall"] for x in per), 4),
                     lcs_word_mean="", ge_0p01_pct=round(
                         100 * sum(1 for x in per if x["risky_alone_recall"] >= 0.01) / len(per), 1)))

    os.makedirs(a.out, exist_ok=True)
    for name, data in ((f"{a.prefix}.csv", rows), (f"{a.prefix}_per_passage.csv", per)):
        with open(os.path.join(a.out, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    print(f"\n{'arm':>10s}  KL nats  nv-recall   max    LCS words  >=0.01")
    for r in rows:
        lab = "risky alone" if r["n"] == -1 else f"n = {r['n']}"
        print(f"{lab:>10s}  {str(r['kl_nats']):>7s}  {r['nv_recall_mean']:.4f}  "
              f"{r['nv_recall_max']:.4f}  {str(r['lcs_word_mean']):>9s}  {r['ge_0p01_pct']:5.1f}%")
    print(f"wrote {a.out}/{a.prefix}.csv")


if __name__ == "__main__":
    main()
