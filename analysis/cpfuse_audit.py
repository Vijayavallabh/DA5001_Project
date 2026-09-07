"""feat-030 (plan v3, C20): does the audit protocol transfer to a mechanism whose guarantee is not a KL budget?

Section IV claims the protocol is not specific to Anchored Decoding. This tests that on CP-Fuse
(Abad et al., ICLR 2025), the other published inference-time copyright defence for language models,
which is structurally different in every way that matters: no anchor, no divergence budget, no K to
publish. CP-Fuse fuses two models each fine-tuned on a DISJOINT split of the protected corpus, so
neither has seen the whole of it, and at each step it reweights them to keep the running
log-likelihood of the output balanced -- if the text is memorised by one model, that model's
contribution is pulled down.

This is a reimplementation from the paper's description, not the authors' code, so it is reported as
such and its absolute numbers are not offered as a reproduction of their headline results. What the
audit measures is the part that transfers: the C2 regime check (how far from a single model does the
fusion actually sit?) and the C5 composition attack (do windowed queries recover the work anyway?).
The second is the point. If composition defeats a mechanism with no budget to compose, then
"per-query guarantees do not compose" is a statement about the query interface rather than about
KL accounting, and Section VI's finding is not a fact about one decoder.

Arms: each component model alone (it memorised its own half), CP-Fuse over the two, and the fusion
with the balancing switched off (a fixed 50/50 geometric mixture) to separate the mechanism from the
mere averaging of two models.

Reads:  --model-a, --model-b (the two shard fine-tunes), data/copybench_attack_train.jsonl
Writes: <out>/cpfuse_audit.csv (per arm, mode and shard), <out>/cpfuse_audit_examples.csv
Usage:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
          .venv/bin/python analysis/cpfuse_audit.py --model-a output/phase3/cpfuse_m0 \
            --model-b output/phase3/cpfuse_m1 --out results --limit 60
"""
import argparse, csv, os, statistics as st, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.certificate_cap import SPLITS  # noqa: E402
from dap.shared import load_prompt_corpus  # noqa: E402
from dap.stats import lcs_word, nv_recall  # noqa: E402

ALPHAS = torch.linspace(0.0, 1.0, 21)


def fuse_logits(l1, l2, alpha):
    """Geometric mixture in log space: log p ∝ alpha*log p1 + (1-alpha)*log p2."""
    lp1, lp2 = torch.log_softmax(l1, -1), torch.log_softmax(l2, -1)
    return torch.log_softmax(alpha * lp1 + (1.0 - alpha) * lp2, -1)


def cpfuse_step(l1, l2, c1, c2, balance=True):
    """CP-Fuse's reweighting: pick alpha minimising the larger of the two running log-likelihoods
    after this step, so whichever model already explains the output too well is pulled down.
    With balance=False this degenerates to a fixed 50/50 mixture, which isolates the reweighting."""
    if not balance:
        return fuse_logits(l1, l2, 0.5), 0.5
    lp1, lp2 = torch.log_softmax(l1, -1), torch.log_softmax(l2, -1)
    a = ALPHAS.to(lp1.device).unsqueeze(1)                       # (A, 1)
    lp = torch.log_softmax(a * lp1 + (1.0 - a) * lp2, dim=-1)    # (A, V)
    p = lp.exp()
    worst = torch.maximum(c1 + (p * lp1).sum(-1), c2 + (p * lp2).sum(-1))
    i = int(worst.argmin())
    return lp[i], float(ALPHAS[i])


@torch.no_grad()
def generate(m1, m2, tok, prompt, n_new, device, arm, seed):
    """Sample n_new tokens under one arm, with a KV cache on both models (without it each step
    re-runs the whole prefix and the audit is quadratic). Returns (text, mean alpha, % extreme alpha)."""
    g = torch.Generator(device=device).manual_seed(seed)
    p_ids = tok(prompt).input_ids
    ids = torch.tensor([p_ids], device=device)
    o1 = m1(ids, use_cache=True)
    o2 = m2(ids, use_cache=True)
    k1, k2 = o1.past_key_values, o2.past_key_values
    l1, l2 = o1.logits[0, -1].float(), o2.logits[0, -1].float()
    c1 = c2 = 0.0
    out, alphas, extreme = [], [], 0
    for _ in range(n_new):
        if arm == "a":
            lp, a = torch.log_softmax(l1, -1), 1.0
        elif arm == "b":
            lp, a = torch.log_softmax(l2, -1), 0.0
        else:
            lp, a = cpfuse_step(l1, l2, c1, c2, balance=(arm == "cpfuse"))
        nxt = torch.multinomial(lp.exp(), 1, generator=g)
        c1 += float(torch.log_softmax(l1, -1)[nxt]); c2 += float(torch.log_softmax(l2, -1)[nxt])
        alphas.append(a)
        extreme += int(a < 0.05 or a > 0.95)
        out.append(int(nxt))
        if int(nxt) == tok.eos_token_id:
            break
        step = nxt.view(1, 1)
        o1 = m1(step, past_key_values=k1, use_cache=True)
        o2 = m2(step, past_key_values=k2, use_cache=True)
        k1, k2 = o1.past_key_values, o2.past_key_values
        l1, l2 = o1.logits[0, -1].float(), o2.logits[0, -1].float()
    return (tok.decode(out, skip_special_tokens=True),
            (st.mean(alphas) if alphas else 0.5), (100.0 * extreme / max(1, len(alphas))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-a", required=True)
    ap.add_argument("--model-b", required=True)
    ap.add_argument("--tokenizer", default="jacquelinehe/tinycomma-1.8b-llama3-tokenizer")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="attack_train")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--seed-tokens", type=int, default=20)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--arms", nargs="+", default=["a", "b", "cpfuse", "mixture"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print("[cf] loading the two shard models", flush=True)
    m1 = AutoModelForCausalLM.from_pretrained(args.model_a, dtype=torch.bfloat16, device_map={"": device}).eval()
    m2 = AutoModelForCausalLM.from_pretrained(args.model_b, dtype=torch.bfloat16, device_map={"": device}).eval()

    # shard membership: model A trained on every 2nd passage by prompt_id, B on the rest (recipes/--shard)
    prompts = sorted((p for p in load_prompt_corpus(args.data, "factscore_prompt")
                      if p.split == args.split and p.reference), key=lambda p: p.prompt_id)
    shard_of = {p.prompt_id: (0 if i % 2 == 0 else 1) for i, p in enumerate(prompts)}
    prompts = prompts[: args.limit]

    rows, examples = [], []
    for arm in args.arms:
        for mode in ("single", "oracle"):
            per = []
            for p in prompts:
                full = f"{p.prompt_text} {p.reference}".strip()
                toks = tok(full, add_special_tokens=False).input_ids
                seed_ids, target_ids = toks[: args.seed_tokens], toks[args.seed_tokens:]
                if len(target_ids) < args.window:
                    continue
                target = tok.decode(target_ids, skip_special_tokens=True)
                if mode == "single":
                    gen, a_mean, extreme = generate(m1, m2, tok, tok.decode(seed_ids), len(target_ids),
                                                    device, arm, args.seed)
                else:  # oracle windows: each window prompted with the true text up to its start
                    pieces = []
                    for w in range(0, len(target_ids) - args.window + 1, args.window):
                        prefix = tok.decode(seed_ids + target_ids[:w], skip_special_tokens=True)
                        g, a_mean, extreme = generate(m1, m2, tok, prefix, args.window, device, arm,
                                                      args.seed + w)
                        pieces.append(g)
                    gen = " ".join(pieces)
                per.append((nv_recall(gen, target), lcs_word(gen, target), a_mean, extreme,
                            shard_of[p.prompt_id]))
                if len(examples) < 40:
                    examples.append(dict(arm=arm, mode=mode, prompt_id=p.prompt_id,
                                         shard=shard_of[p.prompt_id], nv_recall=round(per[-1][0], 4),
                                         generation=gen[:400]))
            for shard in (0, 1, "all"):
                sel = [x for x in per if shard == "all" or x[4] == shard]
                if not sel:
                    continue
                rows.append(dict(arm=arm, mode=mode, shard=shard, n=len(sel),
                                 nv_recall_mean=round(st.mean(x[0] for x in sel), 4),
                                 nv_recall_ge_0p8_pct=round(100 * sum(x[0] >= 0.8 for x in sel) / len(sel), 1),
                                 lcs_word_mean=round(st.mean(x[1] for x in sel), 1),
                                 alpha_mean=round(st.mean(x[2] for x in sel), 3),
                                 alpha_extreme_pct=round(st.mean(x[3] for x in sel), 2)))
            print(f"[cf] {arm:<8} {mode:<7} recall {rows[-1]['nv_recall_mean']:.3f} "
                  f"alpha {rows[-1]['alpha_mean']:.3f}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "cpfuse_audit.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.out, "cpfuse_audit_examples.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(examples[0])); w.writeheader(); w.writerows(examples)
    print(f"\n{'arm':<9} {'mode':<7} {'shard':<5} {'recall':>7} {'>=0.8%':>7} {'alpha':>6}")
    for r in rows:
        print(f"{r['arm']:<9} {r['mode']:<7} {str(r['shard']):<5} {r['nv_recall_mean']:>7.3f} "
              f"{r['nv_recall_ge_0p8_pct']:>7.1f} {r['alpha_mean']:>6.3f}")
    print("\nwrote", os.path.join(args.out, "cpfuse_audit.csv"))


if __name__ == "__main__":
    main()
