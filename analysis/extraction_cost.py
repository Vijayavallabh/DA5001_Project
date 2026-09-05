"""Extraction cost versus the Proposition 2 bound (feat-019, plan v2 C8).

From a retry run of analysis/composition_attack.py (--modes oracle --retries N, per-query log), this counts, per
window, the attempts until the window was reproduced exactly (censored at N), and compares the empirical
success probability per attempt with the bound exp(K_i - S_i) of Proposition 2, where S_i is the anchor's
surprisal of the window given its true prefix (computed here with TinyComma) and K_i = k L. The bound is a theorem
for the pathwise decoder and a heuristic for the KL decoder; both runs are reported.

Writes <out>/<prefix>.csv (per k: windows, attempts, success rate per attempt, bound, and the fraction of windows
whose realised attempts fall below the bound's expectation 1/p) and <out>/<prefix>_windows.csv (per window).
Usage: CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 .venv/bin/python analysis/extraction_cost.py --queries output/phase2/retries_pathwise/queries.jsonl --constraint pathwise --limit 50 --out results
"""
import argparse, csv, json, math, os, statistics as st, sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dap.shared import load_prompt_corpus  # noqa: E402
from recipes.finetune_memorizing import join  # noqa: E402


@torch.no_grad()
def window_surprisals(model, tok, ids, seed_tokens, L, device):
    x = torch.tensor([ids], device=device)
    logp = torch.log_softmax(model(x).logits[0, :-1].float(), dim=-1)
    s = -logp.gather(1, x[0, 1:].unsqueeze(1)).squeeze(1)[seed_tokens - 1:].tolist()  # surprisal of each target token
    return [sum(s[w * L:(w + 1) * L]) for w in range(math.ceil(len(s) / L))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--constraint", default="kl")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="attack_train")
    ap.add_argument("--novel", default="")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--seed-tokens", type=int, default=20)
    ap.add_argument("--safe-model", default="jacquelinehe/tinycomma-1.8b-llama3-tokenizer")
    ap.add_argument("--out", default="results")
    ap.add_argument("--prefix", default="extraction_cost")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.safe_model)
    model = AutoModelForCausalLM.from_pretrained(args.safe_model, dtype=torch.bfloat16, device_map={"": device}).eval()
    prompts = [p for p in load_prompt_corpus(args.data, "factscore_prompt") if p.split == args.split and p.reference and (args.novel in (p.novel_source or ""))][:args.limit]

    attempts = defaultdict(list)  # (k, L, pid, w) -> list of matched flags in attempt order
    for line in open(args.queries):
        q = json.loads(line)
        if q["mode"] != "oracle":
            continue
        attempts[(q["k"], q["L"], q["prompt_id"], q["window"])].append(bool(q["matched"]))
    Ls = sorted({key[1] for key in attempts})
    S_cache = {}
    for p in prompts:
        ids = tok(join(p.prompt_text, p.reference)).input_ids
        for L in Ls:
            S_cache[(p.prompt_id, L)] = window_surprisals(model, tok, ids, args.seed_tokens, L, device)

    wrows = []
    for (k, L, pid, w), flags in sorted(attempts.items()):
        if (pid, L) not in S_cache or w >= len(S_cache[(pid, L)]):
            continue
        S_i = S_cache[(pid, L)][w]
        n = len(flags)
        succ = flags.index(True) + 1 if True in flags else None
        bound_p = math.exp(min(0.0, k * L - S_i)) if k > 0 else 1.0
        wrows.append(dict(constraint=args.constraint, k=k, L=L, prompt_id=pid, window=w, S_i=round(S_i, 2), K_i=k * L, surplus=round(S_i - k * L, 2),
                          attempts=n, matched=int(succ is not None), attempts_to_match=succ, bound_success_prob=bound_p,
                          bound_expected_attempts=(1.0 / bound_p) if bound_p > 0 else float("inf")))
    with open(os.path.join(args.out, f"{args.prefix}_windows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(wrows[0]))
        w.writeheader()
        w.writerows(wrows)
    summary = []
    for (k, L), R in sorted(defaultdict(list, {(r["k"], r["L"]): [x for x in wrows if x["k"] == r["k"] and x["L"] == r["L"]] for r in wrows}).items()):
        n_att = sum(r["attempts"] for r in R)
        n_succ = sum(r["matched"] for r in R)
        emp_p = n_succ / n_att if n_att else 0.0
        bound_mean = st.mean(r["bound_success_prob"] for r in R)
        viol = sum(1 for r in R if r["matched"] and r["attempts_to_match"] < r["bound_expected_attempts"] / 4)  # realised far below the expected cost
        summary.append(dict(constraint=args.constraint, k=k, L=L, windows=len(R), attempts_total=n_att, windows_matched_pct=round(100 * n_succ / len(R), 1),
                            empirical_success_per_attempt=round(emp_p, 4), bound_success_prob_mean=round(bound_mean, 4),
                            surplus_median=round(st.median(r["surplus"] for r in R), 2), surplus_positive_pct=round(100 * sum(r["surplus"] > 0 for r in R) / len(R), 1),
                            attempts_to_match_median=st.median([r["attempts_to_match"] for r in R if r["matched"]]) if n_succ else None,
                            bound_expected_attempts_median=round(st.median(min(r["bound_expected_attempts"], 1e9) for r in R), 1),
                            windows_matched_with_positive_surplus=sum(1 for r in R if r["matched"] and r["surplus"] > 0)))
    with open(os.path.join(args.out, f"{args.prefix}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print(f"[cost] {s['constraint']} k={s['k']:g} L={s['L']}: {s['windows']} windows, matched {s['windows_matched_pct']}%, empirical success/attempt {s['empirical_success_per_attempt']} vs bound mean {s['bound_success_prob_mean']}; "
              f"surplus median {s['surplus_median']} (positive in {s['surplus_positive_pct']}%); matched despite positive surplus: {s['windows_matched_with_positive_surplus']}")
    print("wrote", os.path.join(args.out, f"{args.prefix}.csv"))


if __name__ == "__main__":
    main()
