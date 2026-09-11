"""Plan v5 / feat-088: does selection anchoring scale with n, and is its gain judge-specific?

feat-087 measured selection anchoring at one value of n, with one scoring judge, under a selector
that consults the risky model's own completion. Three objections follow immediately, and this
closes all three. Bands committed in results/onset_prediction_selection_scaling.md before the run.

  O1  does the gain grow with n?            n in {1,2,4,8,16,32,64}, arms nested by seed order
  O2  is +0.081 a property of Phi?          two scoring judges, neither of which selects
  O3  must the selector see the risky model? a POINTWISE reward, no reference completion at all

The selector here is log p("Yes") - log p("No") from Qwen2.5-7B-Instruct on one fixed template --
one forward pass per candidate, no generation, and no access to p_r at any point. The served
distribution is then a function of the anchor and the reward model only, which is what
Proposition 3 needs and what a deployment could run. Nothing about the certificate changes:
q(y) <= n p_s(y) holds for ANY score, so K = log n whatever the reward model is.

Writes <out>/selection_scaling.csv (per arm and judge), <out>/selection_rewards64.csv (the reward
pass, cached: rerunning the judging does not re-score 32,000 candidates).

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/selection_scaling.py --gen-dir output/phase5/sel_anchor64 --out results
"""
import argparse
import csv
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_decoding import (boot_mean, kl_best_of_n,  # noqa: E402
                                         load_baseline, load_candidates)
from analysis.utility import judge_batch  # noqa: E402

# Fixed in the pre-registration; no variants are scored.
REWARD_TMPL = """You are rating a single candidate response to an instruction.

Instruction: {prompt}

Response: {completion}

Is this a helpful, coherent and well-written response to the instruction? Answer Yes or No.
Answer:"""

GRID = (1, 2, 4, 8, 16, 32, 64)


def yes_no_ids(tok):
    """Token ids for Yes/No under both the leading-space and bare spellings, so the score does not
    depend on which one the tokenizer happens to prefer after 'Answer:'."""
    out = {}
    for word, variants in (("yes", ("Yes", " Yes", "yes", " yes")),
                           ("no", ("No", " No", "no", " no"))):
        ids = set()
        for v in variants:
            enc = tok.encode(v, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        out[word] = sorted(ids)
    assert out["yes"] and out["no"], "tokenizer produced no Yes/No ids"
    return out


def score_rewards(model, tok, items, device, batch_size=8, log_every=40):
    """items: (prompt, completion). Returns log p(Yes) - log p(No) at the first answer position."""
    import torch
    ids = yes_no_ids(tok)
    out = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        texts = [tok.apply_chat_template(
            [{"role": "user", "content": REWARD_TMPL.format(prompt=p[:1200], completion=c[:1200])}],
            tokenize=False, add_generation_prompt=True) for p, c in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float()
        lp = torch.log_softmax(logits, dim=-1)
        y = torch.logsumexp(lp[:, ids["yes"]], dim=-1)
        n = torch.logsumexp(lp[:, ids["no"]], dim=-1)
        out += (y - n).tolist()
        if (i // batch_size) % log_every == 0:
            print(f"[reward] {len(out)}/{len(items)}", flush=True)
    return out


def load_rewards(path):
    by = {}
    for r in csv.DictReader(open(path)):
        by.setdefault(r["prompt_id"], []).append((int(r["rank"]), float(r["reward"])))
    return {p: [s for _, s in sorted(v)] for p, v in by.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-dir", default="output/phase5/sel_anchor64")
    ap.add_argument("--baseline-dir", default="output/sweep_plain")
    ap.add_argument("--reward-model", default="Qwen/Qwen2.5-7B-Instruct")
    # Judge C was Llama-3.2-3B-Instruct until a smoke test found it answers "Tie" on 23 of 24
    # probe comparisons under this template -- no resolution, so it cannot decide O2 either way.
    # Replaced by Meta-Llama-3.1-8B-Instruct under the IDENTICAL protocol; it is the checkpoint that
    # generated the opponent, so any self-preference runs against the hypothesis under test. The
    # substitution and its evidence are recorded in results/onset_prediction_selection_scaling.md,
    # written before this arm produced a number. Use the Meta- prefixed id: the other one has no
    # tokenizer in the local cache.
    ap.add_argument("--judges", nargs="+",
                    default=["microsoft/Phi-3.5-mini-instruct",
                             "meta-llama/Meta-Llama-3.1-8B-Instruct"])
    ap.add_argument("--max-n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=8801)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N prompts. For smoke tests only: the bands assume 500.")
    ap.add_argument("--reward-cache", default="results/selection_rewards64.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    grid = [n for n in GRID if n <= a.max_n]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cands = load_candidates(a.gen_dir)
    base = load_baseline(a.baseline_dir)
    pids = sorted(p for p in cands if p in base and len(cands[p]) >= a.max_n)
    assert pids, f"no prompt has {a.max_n} candidates in {a.gen_dir}"
    if a.limit:
        pids = pids[:a.limit]
        print(f"[sel] SMOKE: {len(pids)} prompts only, bands do not apply", flush=True)
    print(f"[sel] {len(pids)} prompts x {a.max_n} candidates", flush=True)

    # ---- phase 1: the pointwise reward, cached -----------------------------------------------
    if not os.path.exists(a.reward_cache):
        tok = AutoTokenizer.from_pretrained(a.reward_model, padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        rm = AutoModelForCausalLM.from_pretrained(
            a.reward_model, torch_dtype=getattr(torch, a.dtype)).cuda().eval()
        items, keys = [], []
        for p in pids:
            for j in range(a.max_n):
                _, cls, prompt, gen = cands[p][j]
                items.append((prompt, gen))
                keys.append((p, j, cls, len(gen.split())))
        scores = score_rewards(rm, tok, items, "cuda", batch_size=a.batch_size)
        os.makedirs(os.path.dirname(a.reward_cache) or ".", exist_ok=True)
        with open(a.reward_cache, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["prompt_id", "rank", "prompt_class", "n_words", "reward"])
            for (p, j, cls, nw), s in zip(keys, scores):
                w.writerow([p, j, cls, nw, round(s, 5)])
        print(f"wrote {a.reward_cache}", flush=True)
        del rm
        torch.cuda.empty_cache()
    rewards = load_rewards(a.reward_cache)

    # arm n serves the argmax over the FIRST n candidates in seed order, so the arms nest and a
    # candidate that wins at n also wins at every larger n unless a later one beats it.
    picks = {(p, n): max(range(n), key=lambda j: rewards[p][j]) for p in pids for n in grid}
    distinct = sorted({(p, picks[(p, n)]) for p in pids for n in grid})
    print(f"[sel] {len(distinct)} distinct served completions across {len(grid)} arms", flush=True)

    # ---- phase 2: judge each distinct served completion once, per judge ------------------------
    out, per_judge = [], {}
    for judge in a.judges:
        tok = AutoTokenizer.from_pretrained(judge, padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        # no trust_remote_code: Phi-3.5-mini's repo carries custom modelling code transformers
        # tries to FETCH even with the weights cached, and HF_HUB_OFFLINE then fails the load.
        jm = AutoModelForCausalLM.from_pretrained(
            judge, torch_dtype=getattr(torch, a.dtype)).cuda().eval()
        items, flips = [], []
        for p, j in distinct:
            _, _, prompt, gen = cands[p][j]
            flip = rng.random() < 0.5
            flips.append(flip)
            items.append((prompt, base[p], gen) if flip else (prompt, gen, base[p]))
        verdicts = []
        for i in range(0, len(items), 200):
            verdicts += judge_batch(jm, tok, items[i:i + 200], "cuda")
            print(f"[judge {judge.split('/')[-1]}] {len(verdicts)}/{len(items)}", flush=True)
        u_of = {}
        for (p, j), flip, v in zip(distinct, flips, verdicts):
            won = (v == "B") if flip else (v == "A")
            u_of[(p, j)] = 1.0 if won else 0.5 if v == "Tie" else 0.0
        per = {p: {n: u_of[(p, picks[(p, n)])] for n in grid} for p in pids}
        per_judge[judge] = per
        base_u = [per[p][1] for p in pids]
        for n in grid:
            us = [per[p][n] for p in pids]
            lo, hi = boot_mean(us, rng)
            diffs = [per[p][n] - per[p][1] for p in pids]
            g = sum(diffs) / len(diffs)
            g_lo, g_hi = boot_mean(diffs, rng) if n > 1 else (0.0, 0.0)
            out.append(dict(judge=judge.split("/")[-1], selector="pointwise reward (Qwen2.5-7B)",
                            n=n, kl_nats=round(kl_best_of_n(n), 4), n_prompts=len(us),
                            u=round(sum(us) / len(us), 4), u_lo95=round(lo, 4), u_hi95=round(hi, 4),
                            gain=round(g, 4), gain_lo95=round(g_lo, 4), gain_hi95=round(g_hi, 4),
                            mean_words=round(sum(len(cands[p][picks[(p, n)]][3].split())
                                                 for p in pids) / len(pids), 1)))
            print(f"  {judge.split('/')[-1]:24s} n={n:3d}  KL {kl_best_of_n(n):5.3f}  "
                  f"u={out[-1]['u']:.4f} [{lo:.3f}, {hi:.3f}]  gain {g:+.4f} "
                  f"[{g_lo:+.4f}, {g_hi:+.4f}]", flush=True)
        assert abs(sum(base_u) / len(base_u) - out[-len(grid)]["u"]) < 1e-9
        del jm
        torch.cuda.empty_cache()

    # Spearman of u against log n, per judge, over the seven arms -- the O1 secondary.
    from analysis.seed_effect import spearman
    for judge in a.judges:
        rows = [r for r in out if r["judge"] == judge.split("/")[-1]]
        rho = spearman([math.log(r["n"]) for r in rows], [r["u"] for r in rows])
        print(f"  {judge.split('/')[-1]:24s} Spearman(u, log n) = {rho:+.3f} over {len(rows)} arms")
        for r in rows:
            r["spearman_u_logn"] = round(rho, 4)

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "selection_scaling.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    ppath = os.path.join(a.out, "selection_scaling_per_prompt.csv")
    with open(ppath, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["judge", "prompt_id"] + [f"u_n{n}" for n in grid])
        for judge, per in per_judge.items():
            for p in pids:
                w.writerow([judge.split("/")[-1], p] + [per[p][n] for n in grid])
    print(f"wrote {path} and {ppath}")


if __name__ == "__main__":
    main()
