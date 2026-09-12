"""feat-101: the judge-free axis.  Does selection anchoring lift an OBJECTIVE metric?

Every utility number in the paper is a pairwise judged preference, so a reviewer can ask whether
the constructive claim is a property of the mechanism or of the judge.  GSM8K exact match is not a
judge: the answer is a number and the grader is `==`.

Both selection rules scored here return one of the `n` samples drawn from the anchor, so both
satisfy `q(y) <= n p_s(y)` for the served string and both carry the SAME `\\log n` pathwise
certificate Proposition~4 proves -- including **majority vote**, which is self-consistency, the most
widely deployed inference-time method there is.  The metered decoder's certificate on the same
workload is `k T_max`.

Baselines are mandatory and on the same problems and seeds: `k = 0` is the anchor alone, which is
the `n = 1` arm here, and `k = -1` is the unconstrained risky model alone.

Two stages, both driven by one command; generation is skipped when its JSONL already exists, so a
re-score costs no GPU.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/selection_verifiable.py --anchor common-pile/comma-v0.1-2t \
      --limit 500 --max-n 64 --out results --tag _comma7b
"""
import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_scaling import score_rewards  # noqa: E402

NUM = re.compile(r"-?\d[\d,]*\.?\d*")
N_GRID = [1, 2, 4, 8, 16, 32, 64]


def gold(answer):
    return answer.split("####")[-1].strip().replace(",", "")


def extract(text):
    """The 8-shot format ends an answer with '#### N'; a base model that runs on starts the next
    question.  Cut at the run-on, prefer the number after '####', else the last number."""
    body = text.split("Question:")[0]
    tail = body.split("####")[-1] if "####" in body else body
    m = NUM.findall(tail)
    if not m:
        return None
    v = m[-1].replace(",", "").rstrip(".")
    return v if v not in ("", "-") else None


def load_gsm8k(limit, n_shot):
    from datasets import load_dataset
    d = load_dataset("openai/gsm8k", "main")
    shots = "".join(f"Question: {r['question']}\nAnswer: {r['answer']}\n\n"
                    for r in d["train"].select(range(n_shot)))
    test = d["test"] if not limit else d["test"].select(range(limit))
    return shots, [dict(qid=f"gsm{i}", question=r["question"], gold=gold(r["answer"]))
                   for i, r in enumerate(test)]


def generate(model_id, shots, items, n, max_new, temperature, batch_size, seed, path, greedy=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                                 device_map={"": 0}).eval()
    torch.manual_seed(seed)
    flat = [(it["qid"], shots + f"Question: {it['question']}\nAnswer:")
            for it in items for _ in range(n)]
    with open(path, "w") as fh:
        for s in range(0, len(flat), batch_size):
            chunk = flat[s:s + batch_size]
            enc = tok([p for _, p in chunk], return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, do_sample=not greedy,
                                     temperature=None if greedy else temperature,
                                     top_k=None if greedy else 0,
                                     top_p=None if greedy else 1.0,
                                     max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
            for j, (qid, _) in enumerate(chunk):
                txt = tok.decode(out[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                fh.write(json.dumps(dict(qid=qid, text=txt)) + "\n")
            fh.flush()
            if (s // batch_size) % 20 == 0:
                print(f"[gen] {min(s + batch_size, len(flat))}/{len(flat)}", flush=True)
    del model
    torch.cuda.empty_cache()


def read_gen(path):
    by = {}
    for line in open(path):
        r = json.loads(line)
        by.setdefault(r["qid"], []).append(r["text"])
    return by


def majority(answers):
    """Self-consistency: the modal extracted answer, ties to the earliest sample.  Returns the
    INDEX of the served sample, so the served string is one of the n drawn and Prop 4 applies."""
    counts = Counter(a for a in answers if a is not None)
    if not counts:
        return 0
    best = max(counts.values())
    top = {a for a, c in counts.items() if c == best}
    return next(i for i, a in enumerate(answers) if a in top)


def boot(correct, reps, seed):
    """Mean accuracy with a percentile CI over problems."""
    rng = random.Random(seed)
    m = len(correct)
    mean = sum(correct) / m
    draws = sorted(sum(correct[rng.randrange(m)] for _ in range(m)) / m for _ in range(reps))
    return mean, draws[int(0.025 * reps)], draws[int(0.975 * reps)]


def boot_gain(correct, base, reps, seed):
    """Paired bootstrap of acc(arm) - acc(n=1) over the same resampled problems."""
    rng = random.Random(seed)
    m = len(correct)
    mean = (sum(correct) - sum(base)) / m
    draws = []
    for _ in range(reps):
        idx = [rng.randrange(m) for _ in range(m)]
        draws.append(sum(correct[i] - base[i] for i in idx) / m)
    draws.sort()
    return mean, draws[int(0.025 * reps)], draws[int(0.975 * reps)]


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="common-pile/comma-v0.1-2t")
    ap.add_argument("--risky", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--reward-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--gen-dir", default="output/phase5/verifiable")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--max-n", type=int, default=64)
    ap.add_argument("--n-shot", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--reward-batch-size", type=int, default=8)
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=8801)
    ap.add_argument("--skip-risky", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    os.makedirs(a.gen_dir, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    shots, items = load_gsm8k(a.limit, a.n_shot)
    print(f"[verif] {len(items)} problems, {a.n_shot}-shot, anchor {a.anchor}", flush=True)

    anchor_path = os.path.join(a.gen_dir, f"anchor{a.tag}_n{a.max_n}.jsonl")
    if not os.path.exists(anchor_path):
        generate(a.anchor, shots, items, a.max_n, a.max_new, a.temperature,
                 a.batch_size, a.seed, anchor_path)
    gens = read_gen(anchor_path)
    assert all(len(gens[it["qid"]]) == a.max_n for it in items), "ragged generation file"

    # k = -1: the unconstrained risky model alone, same problems, greedy and sampled.
    risky = {}
    for name, greedy in (("greedy", True), ("sampled", False)):
        p = os.path.join(a.gen_dir, f"risky_{name}{a.tag}.jsonl")
        if not a.skip_risky and not os.path.exists(p):
            generate(a.risky, shots, items, 1, a.max_new, a.temperature,
                     a.batch_size, a.seed, p, greedy=greedy)
        if os.path.exists(p):
            risky[name] = read_gen(p)

    ans = {q: [extract(t) for t in v] for q, v in gens.items()}
    empty = sum(1 for v in ans.values() for x in v if x is None) / (len(ans) * a.max_n)
    print(f"[verif] no answer extracted in {empty:.4f} of samples", flush=True)

    # The pointwise reward, cached: n_max scores per problem, computed once.
    rw_path = os.path.join(a.out, f"selection_verifiable_rewards{a.tag}.csv")
    if not os.path.exists(rw_path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        rt = AutoTokenizer.from_pretrained(a.reward_model)
        rm = AutoModelForCausalLM.from_pretrained(a.reward_model, dtype=torch.bfloat16,
                                                  device_map={"": 0}).eval()
        pairs, keys = [], []
        for it in items:
            for i, t in enumerate(gens[it["qid"]]):
                pairs.append((f"Question: {it['question']}\nAnswer:", t))
                keys.append((it["qid"], i))
        sc = score_rewards(rm, rt, pairs, rm.device, batch_size=a.reward_batch_size)
        with open(rw_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["prompt_id", "rank", "reward"])
            for (q, i), s in zip(keys, sc):
                w.writerow([q, i, f"{s:.6f}"])
        del rm
        torch.cuda.empty_cache()
    rewards = {}
    for r in csv.DictReader(open(rw_path)):
        rewards.setdefault(r["prompt_id"], []).append((int(r["rank"]), float(r["reward"])))
    rewards = {q: [s for _, s in sorted(v)] for q, v in rewards.items()}

    rows = []
    base = {}
    for rule in ("majority vote (self-consistency)", "pointwise reward (Qwen2.5-7B)"):
        for n in [x for x in N_GRID if x <= a.max_n]:
            correct = []
            for it in items:
                q = it["qid"]
                cand = ans[q][:n]
                if rule.startswith("majority"):
                    i = majority(cand)
                else:
                    sc = rewards[q][:n]
                    i = max(range(n), key=lambda j: sc[j])
                correct.append(1.0 if cand[i] == it["gold"] else 0.0)
            if n == 1:
                base[rule] = correct
            acc, lo, hi = boot(correct, a.reps, a.seed + n)
            g, glo, ghi = boot_gain(correct, base[rule], a.reps, a.seed + n)
            rows.append(dict(arm=rule, n=n, budget_nats=round(__import__("math").log(n), 4),
                             n_problems=len(items), acc=round(acc, 4),
                             acc_lo95=round(lo, 4), acc_hi95=round(hi, 4),
                             gain=round(g, 4), gain_lo95=round(glo, 4), gain_hi95=round(ghi, 4)))
    for rule in base:
        sub = [r for r in rows if r["arm"] == rule]
        rho = spearman([__import__("math").log(r["n"]) for r in sub], [r["acc"] for r in sub])
        for r in sub:
            r["spearman_acc_logn"] = round(rho, 4)

    for name, g in risky.items():
        correct = [1.0 if extract(g[it["qid"]][0]) == it["gold"] else 0.0 for it in items]
        acc, lo, hi = boot(correct, a.reps, a.seed)
        rows.append(dict(arm=f"risky model alone, k=-1 ({name})", n=1, budget_nats="",
                         n_problems=len(items), acc=round(acc, 4), acc_lo95=round(lo, 4),
                         acc_hi95=round(hi, 4), gain="", gain_lo95="", gain_hi95="",
                         spearman_acc_logn=""))

    path = os.path.join(a.out, f"selection_verifiable{a.tag}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[verif] wrote {path}")
    for r in rows:
        print(f"  {r['arm']:<34} n={r['n']:<3} acc={r['acc']:.4f} "
              f"[{r['acc_lo95']:.4f}, {r['acc_hi95']:.4f}] gain={r['gain']}")


if __name__ == "__main__":
    main()
