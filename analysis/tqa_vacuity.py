"""S(x) for a TriviaQA answer under the anchor: where Proposition 2's vacuity threshold sits.

The metered decoder's judge-free arm caps at K = k*T_max with T_max = 24, so k=0.5/1/3/20 are
12/24/72/480 nats.  `sections/appendix_selection.tex` calls 72 "already vacuous" and calls 12 and
24 "the budgets whose certificate is not vacuous", and NOTHING in results/ measures S(x) for a
short factual answer -- the same paragraph even says so.  This script measures it.

S(x) = -log P_anchor(any accepted alias | served prompt), scored as -log max_alias P(alias), which
overstates S(x) and is therefore conservative against the "already vacuous" claim.

The served prompt is read verbatim from data/bench/triviaqa_factual.jsonl -- the same string h1.py
served the metered arm -- never reconstructed (caution (aa)).

Bands: results/onset_prediction_tqa_vacuity.md.
"""
import argparse
import csv
import json
import os

import torch

ANCHOR = "jacquelinehe/tinycomma-1.8b-llama3-tokenizer"
RISKY = "meta-llama/Meta-Llama-3.1-8B-Instruct"
BUDGETS = [12.0, 24.0, 72.0, 480.0]
TEMP = 1.0   # feat-203: --temperature scores the aliases under the tempered anchor, softmax(logits / T)


def load(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        aliases = [a for a in (r.get("reference") or "").split(" ||| ") if a.strip()]
        if not aliases:
            aliases = [r["expected_answer"]]
        rows.append((r["prompt_id"], r["prompt_text"], aliases))
    return rows


def surprisal(model, tok, prompt, cont, device, probe=False):
    """-log p(cont | prompt) in nats, summed over the continuation's tokens.

    The continuation's first token carries the leading space after 'Answer:'.  The boundary is
    located as the longest common prefix of tokenize(prompt) and tokenize(prompt+cont), so a BPE
    merge across the boundary is charged to the continuation rather than silently dropped.
    """
    ids_p = tok(prompt, return_tensors="pt").input_ids[0]
    ids_f = tok(prompt + cont, return_tensors="pt").input_ids[0]
    k = 0
    while k < len(ids_p) and k < len(ids_f) and ids_p[k] == ids_f[k]:
        k += 1
    assert k < len(ids_f), "continuation tokenized to nothing"
    ids = ids_f.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    lp = torch.log_softmax(logits[:-1] / TEMP, dim=-1)
    tgt = ids_f[1:].to(lp.device)
    per = -lp[torch.arange(len(tgt)), tgt][k - 1:]
    if probe:
        toks = [tok.decode([t]) for t in ids_f[k:]]
        print(f"    boundary at {k}/{len(ids_f)}  cont tokens {toks}")
        print(f"    per-token nats {[round(float(v), 2) for v in per]}  total {float(per.sum()):.2f}")
    return float(per.sum())


def best(model, tok, prompt, aliases, device, probe=False):
    return min(surprisal(model, tok, prompt, " " + a, device, probe) for a in aliases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/bench/triviaqa_factual.jsonl")
    ap.add_argument("--anchor", default=ANCHOR)
    ap.add_argument("--risky", default=RISKY)
    ap.add_argument("--out", default="results/tqa_vacuity.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--probe", type=int, default=0, help="G1: print N items' tokenization and exit")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="feat-203: the decoding temperature of the anchor whose S(x) is measured")
    a = ap.parse_args()
    global TEMP
    TEMP = a.temperature

    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = load(a.corpus)
    if a.limit:
        rows = rows[:a.limit]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def openm(mid):
        t = AutoTokenizer.from_pretrained(mid)
        m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).to(dev).eval()
        return m, t

    if a.probe:
        m, t = openm(a.anchor)
        for pid, prompt, al in rows[:a.probe]:
            print(f"[G1] {pid}  prompt tail {prompt[-60:]!r}")
            print(f"     aliases {al}")
            best(m, t, prompt, al, dev, probe=True)
        return

    print(f"[tqa_vacuity] {len(rows)} items, dev={dev}", flush=True)

    def sweep(model, tok, pairs, tag):
        out = []
        for i, (prompt, al) in enumerate(pairs):
            out.append(best(model, tok, prompt, al, dev))
            if (i + 1) % 100 == 0:
                print(f"  [{tag}] {i + 1}/{len(pairs)}", flush=True)
        return out

    # The risky model is opened FIRST: it is the gated one, so a cache or token problem fails in
    # seconds rather than after the anchor's two passes (which is exactly what happened once).
    m, t = openm(a.risky)
    s_risky = sweep(m, t, [(p, al) for _, p, al in rows], "risky")
    del m
    torch.cuda.empty_cache()
    m, t = openm(a.anchor)
    s_anchor = sweep(m, t, [(p, al) for _, p, al in rows], "anchor")
    # G3: the prompt must matter -- item i scored against item i+1's aliases.
    s_perm = sweep(m, t, [(rows[i][1], rows[(i + 1) % len(rows)][2]) for i in range(len(rows))],
                   "perm")
    del m
    torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["prompt_id", "n_aliases", "s_anchor_nats", "s_risky_nats", "s_perm_nats"])
        for (pid, _, al), sa, sr, sp in zip(rows, s_anchor, s_risky, s_perm):
            w.writerow([pid, len(al), f"{sa:.6f}", f"{sr:.6f}", f"{sp:.6f}"])

    def med(v):
        v = sorted(v)
        n = len(v)
        return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])

    ma, mr, mp = med(s_anchor), med(s_risky), med(s_perm)
    print(f"[G2] median S(x): anchor {ma:.2f}  risky {mr:.2f}  -> "
          f"{'PASS' if mr < ma else 'FAIL'} (risky must be lower)")
    print(f"[G3] median permuted {mp:.2f} vs true {ma:.2f}, gap {mp - ma:.2f} -> "
          f"{'PASS' if mp - ma >= 5.0 else 'FAIL'} (>= 5 nats)")
    with open(a.out.replace(".csv", "_summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["budget_nats", "k", "vacuous_frac"])
        for K in BUDGETS:
            frac = sum(1 for s in s_anchor if s <= K) / len(s_anchor)
            print(f"  K={K:6.0f} nats (k={K / 24:g}): vacuous on {frac:.3f}")
            w.writerow([f"{K:g}", f"{K / 24:g}", f"{frac:.4f}"])
    print(f"[tqa_vacuity] wrote {a.out}")


if __name__ == "__main__":
    main()
