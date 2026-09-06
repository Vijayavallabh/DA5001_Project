"""Per-user odometer / filter simulation (feat-021a, plan v2 C10) from the per-query log of the composition attack.

A privacy-filter analogue (Rogers et al. 2016; Whitehouse et al. 2023): the deployer tracks each user's cumulative
KL spend sum_j Z_j across queries (or the realised log-ratio sum_j R_j) and stops serving once it exceeds B_user.
For each passage, strategy (mode, L) and k, this replays the logged queries in order, cuts the sequence at the
first query that would exceed B_user, re-stitches the windows served before the cut, and scores near-verbatim
recall against the target. The anchor surprisal S(x) of the passage (from results/budget_path.csv, if present)
is the natural scale for B_user: under the pathwise decoder the reproduction probability is at most
exp(B_user - S(x)) (Proposition 2 with the total budget capped).

Reads: --queries (jsonl from analysis/composition_attack.py --queries-out), --data (to rebuild the targets).
Writes: <out>/<prefix>.csv (per k, mode, L, B_user: recall mean, fraction of passages cut, queries served mean).
Usage: .venv/bin/python analysis/odometer.py --queries output/phase2/comp8b_kl/queries.jsonl --out results
"""
import argparse, csv, json, os, statistics as st, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dap.shared import load_prompt_corpus  # noqa: E402
from dap.stats import nv_recall  # noqa: E402
from recipes.finetune_memorizing import join  # noqa: E402


def replay(queries, budget, field):
    """queries: list of dicts in served order (one per window; the matched or last attempt). Returns (served texts, n served, cut?)."""
    total, texts = 0.0, []
    for q in queries:
        total += q[field]
        if budget is not None and total > budget:
            return texts, len(texts), True
        texts.append(q["text"])
    return texts, len(texts), False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="attack_train")
    ap.add_argument("--novel", default="")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed-tokens", type=int, default=20)
    ap.add_argument("--safe-model", default="jacquelinehe/tinycomma-1.8b-llama3-tokenizer")
    ap.add_argument("--budgets", nargs="+", type=float, default=[50, 100, 200, 400, 800, 1600, 3200])
    ap.add_argument("--field", choices=["Z", "R"], default="Z", help="charge the KL spend (Z) or the realised log-ratio (R)")
    ap.add_argument("--surprisals", default="results/budget_path.csv", help="per-passage anchor surprisal S_total (optional)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--prefix", default="odometer")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.safe_model)
    prompts = [p for p in load_prompt_corpus(args.data, "factscore_prompt") if p.split == args.split and p.reference and (args.novel in (p.novel_source or ""))][:args.limit]
    targets = {}
    for p in prompts:
        ids = tok(join(p.prompt_text, p.reference)).input_ids
        targets[p.prompt_id] = tok.decode(ids[args.seed_tokens:], skip_special_tokens=True)
    S = {}
    if args.surprisals and os.path.exists(args.surprisals):
        for r in csv.DictReader(open(args.surprisals)):
            S[r["prompt_id"]] = float(r["S_total"])

    # served query per (k, mode, L, prompt, window): the matched attempt if any, else the last attempt; all attempts count toward the spend
    served = defaultdict(dict)
    spent = defaultdict(lambda: defaultdict(float))
    for line in open(args.queries):
        q = json.loads(line)
        key = (q["k"], q["mode"], q["L"], q["prompt_id"])
        w = q["window"]
        spent[key][w] += q[args.field]
        prev = served[key].get(w)
        if prev is None or q.get("matched") or not prev.get("matched"):
            served[key][w] = q
    rows = []
    for key in sorted(served, key=lambda x: (x[0], x[1], x[2], x[3])):
        k, mode, L, pid = key
        if pid not in targets:
            continue
        qs = [dict(served[key][w], **{args.field: spent[key][w]}) for w in sorted(served[key])]
        for B in [None] + args.budgets:
            texts, n, cut = replay(qs, B, args.field)
            rows.append(dict(k=k, mode=mode, L=L, prompt_id=pid, B_user=B if B is not None else float("inf"), queries_served=n, queries_total=len(qs), cut=int(cut),
                             spend_total=round(sum(q[args.field] for q in qs), 2), S_total=S.get(pid), recall=round(nv_recall("".join(texts), targets[pid]), 4)))
    summary = []
    groups = defaultdict(list)
    for r in rows:
        groups[(r["k"], r["mode"], r["L"], r["B_user"])].append(r)
    for (k, mode, L, B), R in sorted(groups.items()):
        summary.append(dict(k=k, mode=mode, L=L, B_user=B, field=args.field, n=len(R), recall_mean=round(st.mean(r["recall"] for r in R), 4),
                            recall_ge_0p8_pct=round(100 * sum(r["recall"] >= 0.8 for r in R) / len(R), 1), cut_pct=round(100 * st.mean(r["cut"] for r in R), 1),
                            queries_served_mean=round(st.mean(r["queries_served"] for r in R), 2), spend_total_mean=round(st.mean(r["spend_total"] for r in R), 1),
                            S_total_median=round(st.median([r["S_total"] for r in R if r["S_total"] is not None]), 1) if any(r["S_total"] is not None for r in R) else None))
    with open(os.path.join(args.out, f"{args.prefix}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    with open(os.path.join(args.out, f"{args.prefix}_per_passage.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for s in summary:
        if s["mode"] == "oracle" and s["L"] == 50:
            print(f"[odo] k={s['k']:g} oracle L=50 B_user={s['B_user']}: recall {s['recall_mean']} (>=0.8: {s['recall_ge_0p8_pct']}%), cut {s['cut_pct']}%, spend {s['spend_total_mean']}, S median {s['S_total_median']}")
    print("wrote", os.path.join(args.out, f"{args.prefix}.csv"))


if __name__ == "__main__":
    main()
