"""Budget-path feasibility (feat-023, plan v2 C12 / Proposition 4).

The banking rule of Anchored Decoding is a token bucket of rate k with an initial debt delta_init: after t+1 steps
the cumulative spend may not exceed (t+1) k - delta_init. Reproducing a target y* nearly verbatim needs a spend of
about s_t = -log p_s(y*_t | prefix) at every step (data processing to the binary event "the true token"), so
verbatim recall is impossible at rate k whenever the surprisal prefix sums cross the budget line:
    feasible(k)  <=>  max_t [ sum_{i<=t} s_i - (t+1) k ] <= -delta_init,
i.e. k >= k_crit := max_t (S_t + delta_init) / (t+1). The slackened version of Proposition 4 replaces s_i by
(1-eta) s_i and k by k + log 2. Both need only the anchor: one teacher-forced forward pass per passage.

Reads the per-passage single-query recall of a composition run (results/composition.csv) so that the anchor-only
prediction F(k) = fraction feasible can be plotted against the observed recall, and reports the dispersion of s_t
(the varentropy) that separates k_crit from the mean per-token surprisal.

Outputs: <out>/<prefix>.csv (per passage), <out>/<prefix>_summary.csv (per k), <figures>/<prefix>.{pdf,png}.
Usage: CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 .venv/bin/python analysis/budget_path.py --composition results/composition.csv --limit 100
"""
import argparse, csv, math, os, statistics as st, sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dap.shared import load_prompt_corpus  # noqa: E402
from dap.warp import warp_logits  # noqa: E402
from recipes.finetune_memorizing import join  # noqa: E402

LN2 = math.log(2)


@torch.no_grad()
def token_surprisals(model, tok, ids, seed_tokens, temperature, repetition_penalty, device):
    """s_t = -log p_s(ids[t] | ids[:t]) for t >= seed_tokens, under the warped anchor."""
    x = torch.tensor([ids], device=device)
    logits = model(x).logits[0, :-1].float()  # row j predicts ids[j+1] after seeing ids[:j+1]
    logits = warp_logits(logits, x[0, :-1], temperature=temperature, repetition_penalty=repetition_penalty, offset=1)
    logp = torch.log_softmax(logits, dim=-1)
    tgt = x[0, 1:]
    s = -logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
    return s[seed_tokens - 1:].tolist()  # predictions of ids[seed_tokens:], i.e. the target tokens


def simulate_bucket(s, delta, k):
    """Token-bucket service model: the bank starts at -delta, gains k per step, and pays s_t for token t if it can;
    a token it cannot pay for is served from the anchor mixture and the allowance of that step is spent (bank -> 0).
    Returns the list of paid (reproducible) tokens."""
    bank, paid = -delta, []
    for v in s:
        bank += k
        if bank >= v:
            bank -= v
            paid.append(True)
        else:  # anchor-forced (bank < 0: the debt is still being repaid) or under-paid (0 <= bank < s_t: the allowance is spent)
            bank = min(bank, 0.0)
            paid.append(False)
    return paid


def reproducible_fraction(s, delta, k, min_run=25):
    """Fraction of target tokens inside runs of at least min_run consecutively paid tokens (near-verbatim recall
    counts only long verbatim spans; 20 words is about 25 tokens)."""
    paid = simulate_bucket(s, delta, k)
    total, run = 0, 0
    for p in paid + [False]:
        if p:
            run += 1
        else:
            if run >= min_run:
                total += run
            run = 0
    return total / len(s)


def feasible_prefix(s, delta, k):
    """Number of leading target tokens t* such that the bank never overflows before t*: the longest prefix the
    budget can pay for at rate k (predicted verbatim-reproducible fraction = t*/T)."""
    cum = 0.0
    for t, v in enumerate(s):
        cum += v
        if cum - (t + 1) * k > -delta:
            return t
    return len(s)


def k_critical(s, delta, eta=0.0, slack=0.0):
    """Smallest rate k at which the (slackened) surplus never exceeds the bank: max_t ((1-eta) S_t + delta)/(t+1) - slack."""
    cum, best = 0.0, -float("inf")
    for t, v in enumerate(s):
        cum += (1 - eta) * v
        best = max(best, (cum + delta) / (t + 1))
    return best - slack


def plot(summary, figures, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5})
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ks = [r["k"] for r in summary]
    ax.plot(ks, [r["feasible_pct"] / 100 for r in summary], "o-", color="C3", label="feasible at rate k (anchor surprisal only)")
    ax.plot(ks, [r["feasible_eta_pct"] / 100 for r in summary], "o--", color="C3", alpha=0.5, label="feasible, slackened (η = 0.1, log 2 per step)")
    ax.plot(ks, [r["pred_frac_mean"] for r in summary], "d-", color="C1", label="predicted recall (token bucket, runs ≥ 25 tokens)")
    ax.plot(ks, [r["paid_frac_mean"] for r in summary], "d--", color="C1", alpha=0.5, label="tokens the bucket can pay for (fraction)")
    if any(r.get("recall_mean") is not None for r in summary):
        ax.plot(ks, [r["recall_mean"] for r in summary], "s-", color="C0", label="observed single-query recall (mean)")
        ax.plot(ks, [r["recall_ge_0p5_pct"] / 100 for r in summary], "^-", color="C2", label="observed recall ≥ 0.5 (fraction)")
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([f"{k:g}" for k in ks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("per-token budget k")
    ax.set_ylabel("fraction of passages")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="none")
    fig.tight_layout()
    fig.savefig(os.path.join(figures, f"{prefix}.pdf"))
    fig.savefig(os.path.join(figures, f"{prefix}.png"), dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="attack_train")
    ap.add_argument("--novel", default="")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed-tokens", type=int, default=20)
    ap.add_argument("--safe-model", default="jacquelinehe/tinycomma-1.8b-llama3-tokenizer")
    ap.add_argument("--composition", default="results/composition.csv", help="per-passage composition results; '' to skip the observed-recall join")
    ap.add_argument("--k-values", nargs="+", type=float, default=[0.15, 0.5, 1, 3, 5, 10, 20])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--min-run", type=int, default=25, help="tokens; near-verbatim recall counts spans of >= 20 words")
    ap.add_argument("--out", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--prefix", default="budget_path")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.safe_model)
    model = AutoModelForCausalLM.from_pretrained(args.safe_model, dtype=torch.bfloat16, device_map={"": device}).eval()
    prompts = [p for p in load_prompt_corpus(args.data, "factscore_prompt") if p.split == args.split and p.reference and (args.novel in (p.novel_source or ""))][:args.limit]

    comp = defaultdict(dict)  # prompt_id -> {k: recall}; delta from the single-mode rows
    delta_obs = {}
    if args.composition and os.path.exists(args.composition):
        for r in csv.DictReader(open(args.composition)):
            if r["mode"] != "single":
                continue
            k = float(r["k"])
            comp[r["prompt_id"]][k] = float(r["nv_recall"])
            if k > 0 and r.get("delta_init_mean"):
                delta_obs.setdefault(r["prompt_id"], []).append(float(r["delta_init_mean"]))

    rows = []
    for i, p in enumerate(prompts):
        ids = tok(join(p.prompt_text, p.reference)).input_ids
        s = token_surprisals(model, tok, ids, args.seed_tokens, args.temperature, args.repetition_penalty, device)
        delta = st.mean(delta_obs[p.prompt_id]) if p.prompt_id in delta_obs else 0.0
        row = dict(prompt_id=p.prompt_id, novel=p.novel_source, n_target=len(s), delta_init=round(delta, 3),
                   S_total=round(sum(s), 2), s_mean=round(st.mean(s), 4), s_std=round(st.pstdev(s), 4), s_max=round(max(s), 3),
                   s_p90=round(sorted(s)[int(0.9 * (len(s) - 1))], 3),
                   k_crit=round(k_critical(s, delta), 4), k_crit_eta=round(k_critical(s, delta, eta=args.eta, slack=LN2), 4))
        for k in args.k_values:
            row[f"feasible_k{k:g}"] = int(row["k_crit"] <= k)
            row[f"feasible_eta_k{k:g}"] = int(row["k_crit_eta"] <= k)
            row[f"pred_frac_k{k:g}"] = round(reproducible_fraction(s, delta, k, args.min_run), 4)
            paid = simulate_bucket(s, delta, k)
            row[f"paid_frac_k{k:g}"] = round(sum(paid) / len(s), 4)
            row[f"forced_open_k{k:g}"] = next((i for i, x in enumerate(paid) if x), len(paid))  # anchor-forced opening tokens
            rec = comp.get(p.prompt_id, {}).get(k)
            row[f"recall_k{k:g}"] = rec if rec is None else round(rec, 4)
        rows.append(row)
        if i % 25 == 0:
            print(f"[bp] {i}/{len(prompts)} S={row['S_total']} s_mean={row['s_mean']} k_crit={row['k_crit']} delta={delta:.2f}", flush=True)
    with open(os.path.join(args.out, f"{args.prefix}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = []
    for k in args.k_values:
        recs = [r[f"recall_k{k:g}"] for r in rows if r[f"recall_k{k:g}"] is not None]
        feas = [r[f"feasible_k{k:g}"] for r in rows]
        srow = dict(k=k, n=len(rows), feasible_pct=round(100 * st.mean(feas), 1), feasible_eta_pct=round(100 * st.mean(r[f"feasible_eta_k{k:g}"] for r in rows), 1),
                    pred_frac_mean=round(st.mean(r[f"pred_frac_k{k:g}"] for r in rows), 4), paid_frac_mean=round(st.mean(r[f"paid_frac_k{k:g}"] for r in rows), 4),
                    forced_open_mean=round(st.mean(r[f"forced_open_k{k:g}"] for r in rows), 2),
                    recall_mean=round(st.mean(recs), 4) if recs else None, recall_ge_0p5_pct=round(100 * sum(x >= 0.5 for x in recs) / len(recs), 1) if recs else None,
                    recall_ge_0p8_pct=round(100 * sum(x >= 0.8 for x in recs) / len(recs), 1) if recs else None)
        if recs:
            agree = [int(r[f"feasible_k{k:g}"] == int(r[f"recall_k{k:g}"] >= 0.5)) for r in rows if r[f"recall_k{k:g}"] is not None]
            infeasible_but_recalled = [r for r in rows if r[f"recall_k{k:g}"] is not None and not r[f"feasible_k{k:g}"] and r[f"recall_k{k:g}"] >= 0.5]
            srow["agreement_pct"] = round(100 * st.mean(agree), 1)
            srow["infeasible_but_recall_ge_0p5_n"] = len(infeasible_but_recalled)  # violations of the necessary condition (should be ~0)
            srow["recall_mean_feasible"] = round(st.mean([r[f"recall_k{k:g}"] for r in rows if r[f"feasible_k{k:g}"] and r[f"recall_k{k:g}"] is not None] or [0.0]), 4)
            srow["recall_mean_infeasible"] = round(st.mean([r[f"recall_k{k:g}"] for r in rows if not r[f"feasible_k{k:g}"] and r[f"recall_k{k:g}"] is not None] or [0.0]), 4)
            pf = [r[f"pred_frac_k{k:g}"] for r in rows if r[f"recall_k{k:g}"] is not None]
            ob = [r[f"recall_k{k:g}"] for r in rows if r[f"recall_k{k:g}"] is not None]
            srow["pred_minus_obs_mean"] = round(st.mean(a - b for a, b in zip(pf, ob)), 4)
            srow["pred_obs_pearson"] = round(st.correlation(pf, ob), 3) if len(set(pf)) > 1 and len(set(ob)) > 1 else None
            srow["pred_obs_spearman"] = round(st.correlation(pf, ob, method="ranked"), 3) if len(set(pf)) > 1 and len(set(ob)) > 1 else None
        summary.append(srow)
    kc = [r["k_crit"] for r in rows]
    sm = [r["s_mean"] for r in rows]
    print(f"[bp] k_crit median {st.median(kc):.2f} (p10 {sorted(kc)[len(kc)//10]:.2f}, p90 {sorted(kc)[9*len(kc)//10]:.2f}); mean per-token surprisal median {st.median(sm):.2f}; "
          f"median k_crit / s_mean = {st.median([a/b for a,b in zip(kc,sm)]):.2f}; within-passage s_std median {st.median(r['s_std'] for r in rows):.2f}")
    for srow in summary:
        print(f"[bp] k={srow['k']:g}: feasible {srow['feasible_pct']}% (slackened {srow['feasible_eta_pct']}%), bucket pays {srow['paid_frac_mean']} of tokens, forced opening {srow['forced_open_mean']} tokens, predicted recall {srow['pred_frac_mean']}, observed recall mean {srow['recall_mean']} , >=0.5: {srow['recall_ge_0p5_pct']}%, pearson {srow.get('pred_obs_pearson')} spearman {srow.get('pred_obs_spearman')}, infeasible-but-recalled {srow.get('infeasible_but_recall_ge_0p5_n')}")
    with open(os.path.join(args.out, f"{args.prefix}_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    plot(summary, args.figures, args.prefix)
    print("wrote", os.path.join(args.out, f"{args.prefix}.csv"), os.path.join(args.figures, f"{args.prefix}.pdf"))


if __name__ == "__main__":
    main()
