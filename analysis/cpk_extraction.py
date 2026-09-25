"""feat-211, the leakage half: CP-k's once-per-output rejection rule, run against a memoriser.

For each memorised passage the memoriser draws M continuations of the 20-token seed. The first whose
realised log-ratio against the anchor,
    R(y) = log p_mem(y | seed) - log p_s(y | seed),
summed over the generated token ids through the first end-of-text (kept), is at most kappa is served;
if none passes, the anchor's own draw is served. Then q(y) <= M p_mem(y) 1[R(y) <= kappa] + p_s(y)
<= (M e^kappa + 1) p_s(y) for every y, so D_inf(q || p_s) <= C = log(1 + M e^kappa) pathwise, whatever
the acceptance rate (results/onset_prediction_cpk_baseline.md). kappa is set from C.

Passages, seeds and targets are built by analysis/selection_extraction.py's build(), which builds them
exactly as composition_attack.py does, and every draw decodes the leakage sweep's 296 tokens.
R is computed on the ids the model generated, never on re-tokenised text, so it is the log-ratio of
the event that is actually served.

Writes <out>/cpk_extraction.csv and <out>/cpk_extraction_per_passage.csv.

Usage (host B):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \\
    .venv/bin/python analysis/cpk_extraction.py --risky-model output/memorizing_llama8b --out results
"""
import argparse
import csv
import math
import os
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.cpk_baseline import certificate, kappa_for  # noqa: E402
from analysis.selection_extraction import build  # noqa: E402
from dap.stats import nv_recall, rouge_l_score  # noqa: E402

GRID = (4.1589, 33.271, 83.178, 159.83, 250.0, 400.0, 600.0, 800.0)


def load_tok(name):
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(name, padding_side="left")
    if t.pad_token is None:
        t.pad_token = t.eos_token
    return t


def stop_ids(model, tok):
    e = model.generation_config.eos_token_id
    ids = set(e if isinstance(e, (list, tuple)) else [e])
    ids.add(tok.eos_token_id)
    return {int(x) for x in ids if x is not None}


@torch.no_grad()
def draw_ids(model, tok, prompts, m, max_new, temperature, batch_size, seed, stops):
    """m continuations per prompt as generated token-id lists, cut just after the first stop id."""
    torch.manual_seed(seed)
    out = [[] for _ in prompts]
    flat = [(i, p) for i, p in enumerate(prompts) for _ in range(m)]
    for s in range(0, len(flat), batch_size):
        chunk = flat[s:s + batch_size]
        enc = tok([p for _, p in chunk], return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, do_sample=True, temperature=temperature, top_k=0, top_p=1.0,
                             max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
        width = enc["input_ids"].shape[1]        # left padding: every row's generation starts here
        for j, (i, _) in enumerate(chunk):
            ids = gen[j, width:].tolist()
            cut = next((t + 1 for t, x in enumerate(ids) if x in stops), len(ids))
            out[i].append(ids[:cut])
        if (s // batch_size) % 20 == 0:
            print(f"[cpkx] drew {min(s + batch_size, len(flat))}/{len(flat)}", flush=True)
    return out


@torch.no_grad()
def sum_logprob(model, pairs, batch_size, pad_id):
    """sum_t log p(g_t | p, g_<t) for each (prompt_ids, gen_ids), right-padded so positions are exact."""
    out = []
    for s in range(0, len(pairs), batch_size):
        chunk = pairs[s:s + batch_size]
        seqs = [p + g for p, g in chunk]
        width = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
        att = torch.zeros((len(seqs), width), dtype=torch.long)
        for j, x in enumerate(seqs):
            ids[j, :len(x)] = torch.tensor(x)
            att[j, :len(x)] = 1
        ids, att = ids.to(model.device), att.to(model.device)
        lg = model(input_ids=ids, attention_mask=att).logits.float().log_softmax(-1)
        for j, (p, g) in enumerate(chunk):
            lo, hi = len(p), len(p) + len(g)
            tgt = ids[j, lo:hi]
            out.append(float(lg[j, lo - 1:hi - 1].gather(-1, tgt.unsqueeze(-1)).sum()))
        if (s // batch_size) % 50 == 0:
            print(f"[cpkx] scored {min(s + batch_size, len(pairs))}/{len(pairs)}", flush=True)
    return out


def serve(rs, kappa):
    """Index of the first draw with R <= kappa, or None (the anchor's draw is served)."""
    return next((j for j, r in enumerate(rs) if r <= kappa), None)


def main():
    from transformers import AutoModelForCausalLM
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="attack_train")
    ap.add_argument("--safe-model", default="jacquelinehe/tinycomma-1.8b-llama3-tokenizer")
    ap.add_argument("--risky-model", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--grid", type=float, nargs="+", default=list(GRID))
    ap.add_argument("--seed-tokens", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=296)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--score-batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="results")
    ap.add_argument("--prefix", default="cpk_extraction")
    a = ap.parse_args()

    rtok, stok = load_tok(a.risky_model), load_tok(a.safe_model)
    # R compares the two models on ONE id sequence, so the vocabularies must agree id for id
    assert rtok.get_vocab() == stok.get_vocab(), "the memoriser and the anchor do not share a tokenizer"
    passages = build(rtok, a.data, a.split, a.limit, a.seed_tokens)
    print(f"[cpkx] {len(passages)} passages from {sorted({p['novel'] for p in passages})}, M={a.m}, "
          f"{a.max_new_tokens} new tokens, seed {a.seed_tokens} tokens", flush=True)
    seeds = [p["seed"] for p in passages]
    pids = [rtok(s).input_ids for s in seeds]

    risky = AutoModelForCausalLM.from_pretrained(a.risky_model, torch_dtype=torch.bfloat16).cuda().eval()
    stops = stop_ids(risky, rtok)
    draws = draw_ids(risky, rtok, seeds, a.m, a.max_new_tokens, a.temperature, a.batch_size, a.seed, stops)
    flat = [(pids[i], g) for i in range(len(passages)) for g in draws[i]]
    lp_r = sum_logprob(risky, flat, a.score_batch_size, rtok.pad_token_id)
    del risky
    torch.cuda.empty_cache()

    anchor = AutoModelForCausalLM.from_pretrained(a.safe_model, torch_dtype=torch.bfloat16).cuda().eval()
    lp_s = sum_logprob(anchor, flat, a.score_batch_size, stok.pad_token_id)
    fallback = draw_ids(anchor, stok, seeds, 1, a.max_new_tokens, a.temperature, a.batch_size, a.seed + 1,
                        stop_ids(anchor, stok))
    del anchor
    torch.cuda.empty_cache()

    R = [[lp_r[i * a.m + j] - lp_s[i * a.m + j] for j in range(a.m)] for i in range(len(passages))]
    text = [[rtok.decode(g, skip_special_tokens=True) for g in draws[i]] for i in range(len(passages))]
    anc = [stok.decode(fallback[i][0], skip_special_tokens=True) for i in range(len(passages))]
    rec = [[nv_recall(t, p["target"]) for t in text[i]] for i, p in enumerate(passages)]
    anc_rec = [nv_recall(anc[i], p["target"]) for i, p in enumerate(passages)]

    per, rows = [], []
    for i, p in enumerate(passages):
        row = dict(prompt_id=p["prompt_id"], novel=p["novel"], risky_alone_recall=round(rec[i][0], 4),
                   risky_alone_R=round(R[i][0], 2), min_R=round(min(R[i]), 2),
                   median_R=round(st.median(R[i]), 2), anchor_recall=round(anc_rec[i], 4))
        for c in a.grid:
            j = serve(R[i], kappa_for(c, a.m))
            row[f"accepted_C{c:g}"] = "" if j is None else j
            row[f"recall_C{c:g}"] = round(rec[i][j] if j is not None else anc_rec[i], 4)
        per.append(row)
    for c in a.grid:
        k = kappa_for(c, a.m)
        got = [r[f"recall_C{c:g}"] for r in per]
        acc = [r for r in per if r[f"accepted_C{c:g}"] != ""]
        # G1: every served memoriser draw passed the test it was served under
        assert all(R[per.index(r)][int(r[f"accepted_C{c:g}"])] <= k + 1e-9 for r in acc)
        rows.append(dict(certificate_nats=c, kappa=round(k, 4), check_certificate=round(certificate(k, a.m), 4),
                         n_passages=len(per), served_from_memoriser_pct=round(100 * len(acc) / len(per), 1),
                         nv_recall_mean=round(st.mean(got), 4), nv_recall_max=round(max(got), 4),
                         ge_0p01_pct=round(100 * sum(x >= 0.01 for x in got) / len(got), 1),
                         ge_0p8_pct=round(100 * sum(x >= 0.8 for x in got) / len(got), 1)))
    base = dict(certificate_nats="", kappa="", check_certificate="", n_passages=len(per))
    rows.append({**base, "served_from_memoriser_pct": 100.0, "nv_recall_mean": round(st.mean(r[0] for r in rec), 4),
                 "nv_recall_max": round(max(r[0] for r in rec), 4),
                 "ge_0p01_pct": round(100 * sum(r[0] >= 0.01 for r in rec) / len(rec), 1),
                 "ge_0p8_pct": round(100 * sum(r[0] >= 0.8 for r in rec) / len(rec), 1), "certificate_nats": "memoriser alone"})
    rows.append({**base, "served_from_memoriser_pct": 0.0, "nv_recall_mean": round(st.mean(anc_rec), 4),
                 "nv_recall_max": round(max(anc_rec), 4),
                 "ge_0p01_pct": round(100 * sum(x >= 0.01 for x in anc_rec) / len(anc_rec), 1),
                 "ge_0p8_pct": round(100 * sum(x >= 0.8 for x in anc_rec) / len(anc_rec), 1), "certificate_nats": "anchor alone"})
    allR = [x for r in R for x in r]
    rows.append({**base, "certificate_nats": "median R of all memoriser draws, nats",
                 "nv_recall_mean": round(st.median(allR), 2)})
    for name, rr in ((f"{a.prefix}.csv", rows), (f"{a.prefix}_per_passage.csv", per)):
        with open(os.path.join(a.out, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rr[0].keys()) if name.endswith("passage.csv")
                               else list(rows[0].keys()))
            w.writeheader()
            w.writerows(rr)
    for r in rows:
        print(r, flush=True)


if __name__ == "__main__":
    main()
