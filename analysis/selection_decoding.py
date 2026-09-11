"""Plan v5 / feat-087: the O(1)-nat decoder the no-free-lunch section declines to claim.

Theorem 1 says a bounded utility costs the Cramer rate function of that utility under the safe
model, which is O(1) nats however long the sequence, and results/utility_price.csv measures the
audited decoder paying 165 nats at k=3 where 0.052 would do. sections/frontier.tex then says, in as
many words, that the gap licenses the negative claim and not the constructive one. This builds the
constructive one and measures it.

SELECTION ANCHORING.  Draw n completions from the anchor p_s, score each by the RISKY model's
per-token log-likelihood -- the same objective the audited decoder tilts toward, computed by the
same checkpoint -- and serve the argmax.  Two properties follow immediately and neither is new as
mathematics (beirami2025bestofn, gao2023scaling):

    D_KL(q || p_s) <= log n - (n-1)/n          (1.204 nats at n = 8)
    q(y) <= n p_s(y)  =>  P_q(E) <= n P_s(E)   (Proposition 1 with K = log n)

so the vacuity threshold K = S(x) sits at n = e^S(x), and S(x) is about 850 nats on these passages.
The structural point is the asymmetry: a per-token budget kT grows with what is generated, a
selection budget log n does not.

The arms are nested -- arm n uses the FIRST n candidates in seed order -- so every arm reuses one
judging pass over all n candidates and none is judged twice.  n=1 is the anchor-alone control and
must reproduce the u_safe on record, or the pipeline is wrong and nothing else counts.

The oracle arm selects by the judge's own verdict.  It is a ceiling on what any n=8 selection could
do on this metric and is never used to score a band: it separates "selection cannot reach it" from
"the risky model's likelihood is a poor selector".

Bands committed in results/onset_prediction_selection.md before anything was generated.

Writes <out>/selection_decoding.csv (per arm) and <out>/selection_candidates.csv (per candidate).

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/selection_decoding.py --gen-dir output/phase5/sel_anchor8 --out results
"""
import argparse
import csv
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.utility import CLASSES, judge_batch, served_prompt  # noqa: E402

VERDICT_U = {"win": 1.0, "tie": 0.5, "loss": 0.0}


def kl_best_of_n(n):
    """Beirami et al.: the best-of-n policy's KL from its base is at most log n - (n-1)/n."""
    return math.log(n) - (n - 1) / n if n > 1 else 0.0


def load_candidates(run_dir, k="0"):
    """prompt_id -> [(seed, class, prompt, generation)], sorted by seed so arms nest."""
    out = {}
    for cls in CLASSES:
        path = os.path.join(run_dir, f"trajectories_k{k}_{cls}.jsonl")
        if not os.path.exists(path):
            continue
        for line in open(path):
            r = json.loads(line)
            m, a = r["metadata"], r["aggregate"]
            out.setdefault(m["prompt_id"], []).append(
                (m["seed"], cls, served_prompt(a), a.get("generation") or ""))
    return {p: sorted(v) for p, v in out.items()}


def load_baseline(run_dir):
    """prompt_id -> the unconstrained risky completion at its lowest seed. One fixed opponent per
    prompt, so every candidate of that prompt is judged against the same text."""
    out = {}
    for cls in CLASSES:
        path = os.path.join(run_dir, f"trajectories_k-1_{cls}.jsonl")
        if not os.path.exists(path):
            continue
        for line in open(path):
            r = json.loads(line)
            m, a = r["metadata"], r["aggregate"]
            key = m["prompt_id"]
            cand = (m["seed"], a.get("generation") or "")
            if key not in out or cand < out[key]:
                out[key] = cand
    return {p: v[1] for p, v in out.items()}


def score_candidates(pairs, model_id, device, dtype, batch_size=8):
    """[(prompt, generation)] -> [(total log p_r(y|x), n_tokens)] under the risky model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    out = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        texts = [p + g for p, g in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(device)
        plen = [len(tok(p, truncation=True, max_length=1024)["input_ids"]) for p, _ in chunk]
        with torch.no_grad():
            logits = model(**enc).logits.float().log_softmax(-1)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        for j, (p0, _) in enumerate(chunk):
            # the continuation's tokens are everything after the prompt, up to the pad
            lo, hi = plen[j], int(mask[j].sum().item())
            if hi - lo < 1:
                out.append((0.0, 0))
                continue
            tgt = ids[j, lo:hi]
            lp = logits[j, lo - 1:hi - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            out.append((float(lp.sum().item()), int(hi - lo)))
        if (i // batch_size) % 20 == 0:
            print(f"[sel] scored {min(i + batch_size, len(pairs))}/{len(pairs)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def boot_mean(vals, rng, reps=2000):
    n = len(vals)
    xs = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return xs[int(0.025 * reps)], xs[int(0.975 * reps)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-dir", default="output/phase5/sel_anchor8")
    ap.add_argument("--baseline-dir", default="output/sweep_plain")
    ap.add_argument("--risky-model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--judge", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-values", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--limit-prompts", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="results")
    ap.add_argument("--prefix", default="selection")
    ap.add_argument("--candidates", default="", help="reuse a scored+judged candidate CSV, no GPU")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    cand_path = os.path.join(a.out, f"{a.prefix}_candidates.csv")
    if a.candidates:
        rows = list(csv.DictReader(open(a.candidates)))
    else:
        import torch
        cands, base = load_candidates(a.gen_dir), load_baseline(a.baseline_dir)
        pids = sorted(p for p in cands if p in base)
        if a.limit_prompts:
            pids = pids[:a.limit_prompts]
        n_max = max(a.n_values)
        flat = [(p, i, c) for p in pids for i, c in enumerate(cands[p][:n_max])]
        print(f"[sel] {len(pids)} prompts, {len(flat)} candidates, n_max={n_max}", flush=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = getattr(torch, a.dtype)

        scores = score_candidates([(c[2], c[3]) for _, _, c in flat], a.risky_model, device, dtype)

        from transformers import AutoModelForCausalLM, AutoTokenizer
        jtok = AutoTokenizer.from_pretrained(a.judge, padding_side="left")
        if jtok.pad_token is None:
            jtok.pad_token = jtok.eos_token
        jmodel = AutoModelForCausalLM.from_pretrained(a.judge, torch_dtype=dtype).to(device).eval()
        items, flips = [], []
        for p, _, c in flat:
            flip = rng.random() < 0.5      # randomise which side the candidate appears on
            flips.append(flip)
            items.append((c[2], base[p], c[3]) if flip else (c[2], c[3], base[p]))
        verdicts = []
        for i in range(0, len(items), 200):
            verdicts += judge_batch(jmodel, jtok, items[i:i + 200], device)
            print(f"[sel] judged {len(verdicts)}/{len(items)}", flush=True)
        del jmodel
        torch.cuda.empty_cache()

        rows = []
        for (p, i, c), (lp, ntok), flip, v in zip(flat, scores, flips, verdicts):
            won = (v == "B") if flip else (v == "A")
            outcome = "tie" if v == "Tie" else ("win" if won else "loss")
            rows.append(dict(prompt_id=p, rank=i, seed=c[0], prompt_class=c[1], n_tokens=ntok,
                             logp_total=round(lp, 4),
                             logp_per_token=round(lp / ntok, 6) if ntok else 0.0,
                             shown_second=int(flip), verdict=v, outcome=outcome))
        os.makedirs(a.out, exist_ok=True)
        with open(cand_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {cand_path}", flush=True)

    # ---- arms, all from the one judging pass ------------------------------------------------
    by_prompt = {}
    for r in rows:
        by_prompt.setdefault(r["prompt_id"], []).append(r)
    for v in by_prompt.values():
        v.sort(key=lambda r: int(r["rank"]))

    RULES = [("per-token mean (primary)", lambda r: float(r["logp_per_token"])),
             ("summed log-likelihood", lambda r: float(r["logp_total"])),
             ("oracle: the judge itself", lambda r: VERDICT_U[r["outcome"]])]
    out_rows = []
    for label, key in RULES:
        for n in a.n_values:
            us = []
            for v in by_prompt.values():
                pick = max(v[:n], key=key)
                us.append(VERDICT_U[pick["outcome"]])
            lo, hi = boot_mean(us, rng)
            out_rows.append(dict(rule=label, n=n, kl_nats=round(kl_best_of_n(n), 4),
                                 n_prompts=len(us), u=round(sum(us) / len(us), 4),
                                 u_lo95=round(lo, 4), u_hi95=round(hi, 4),
                                 win_pct=round(100 * sum(1 for x in us if x == 1.0) / len(us), 1),
                                 tie_pct=round(100 * sum(1 for x in us if x == 0.5) / len(us), 1)))

    w = max(len(r["rule"]) for r in out_rows)
    print(f"\n{'rule':{w}s}   n  KL nats       u        95% CI      win%")
    for r in out_rows:
        print(f"{r['rule']:{w}s} {r['n']:3d}   {r['kl_nats']:6.3f}  {r['u']:.4f}  "
              f"[{r['u_lo95']:.3f}, {r['u_hi95']:.3f}]  {r['win_pct']:5.1f}")
    path = os.path.join(a.out, f"{a.prefix}_decoding.csv")
    with open(path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        wtr.writeheader(); wtr.writerows(out_rows)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
