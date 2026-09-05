"""Certificate-strength audit (feat-006, contribution C3).

For every CopyBench reference passage y with prompt x, compute the safe-model surprisal
S(x) = -ln p_s(y | x) (TinyComma) and the risky-model surprisal (Llama-3.1-8B-Instruct), then the
tightest cap on the probability that a K-NAF decoder reproduces y that follows from
D_KL(p* || p_s) <= K by data processing:  max { p : d(p || e^{-S}) <= K }  (binary KL inversion).
The certificate is vacuous when S <= K (cap = 1). The closed-form approximation (K + ln 2) / S (plan Section 3, C3) is
reported for comparison; it differs from the exact inversion by at most 1/(e S).

Writes results/certificate_caps.csv (per passage), results/certificate_cap_summary.csv (per K),
figures/certificate_cap_curve.{pdf,png}.

Usage: CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 .venv/bin/python analysis/certificate_cap.py --data data --out results
"""
import argparse, csv, math, os, statistics as st, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dap.shared import load_prompt_corpus, wrap_chat  # noqa: E402

LN2 = math.log(2)
SPLITS = ("attack_train", "val", "test")


def binary_kl(p, q):
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def cap_exact(S, K):
    """Largest p in [e^-S, 1] with d(p || e^-S) <= K. Monotone in p on that interval, so bisect."""
    if S <= K:
        return 1.0
    q = math.exp(-S) if S < 700 else 0.0
    lo, hi = max(q, 1e-300), 1.0 - 1e-12
    for _ in range(100):
        mid = (lo + hi) / 2
        d = binary_kl(mid, q) if q > 0 else mid * S + (1 - mid) * math.log(1 - mid)
        lo, hi = (mid, hi) if d <= K else (lo, mid)
    return lo


def cap_simple(S, K):
    return min(1.0, (K + LN2) / S) if S > 0 else 1.0


@torch.no_grad()
def surprisal(model, tok, prompt, reference, device):
    p_ids = tok(prompt).input_ids
    r_ids = tok(reference, add_special_tokens=False).input_ids
    ids = torch.tensor([p_ids + r_ids], device=device)
    logp = torch.log_softmax(model(ids).logits[0, len(p_ids) - 1:-1].float(), dim=-1)
    tok_lp = logp.gather(1, ids[0, len(p_ids):].unsqueeze(1)).squeeze(1)
    return float(-tok_lp.sum()), len(r_ids), float(-tok_lp[0])


def plot(summary, figures, t_max=200):
    """Certificate-strength curve from the rows of certificate_cap_summary.csv (feat-013 rebuilds figures from CSV)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    allrows = sorted([r for r in summary if r["split"] == "all"], key=lambda r: r["k"])
    ks = [r["k"] for r in allrows]
    ax.plot(ks, [r["vacuous_pct"] for r in allrows], "o-", color="C3", label="passages with vacuous certificate (S ≤ K)")
    ax.plot(ks, [100 * r["cap_median"] for r in allrows], "s--", color="C0", label="median cap on P(reproduce reference)")
    for split, ls in zip(SPLITS, (":", "-.", (0, (1, 1)))):
        rows = sorted([r for r in summary if r["split"] == split], key=lambda r: r["k"])
        ax.plot([r["k"] for r in rows], [r["vacuous_pct"] for r in rows], linestyle=ls, color="C3", lw=0.8, alpha=0.6, label=f"vacuous, {split}")
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([f"{k:g}\nK={k * t_max:g}" for k in ks])
    ax.set_xlabel("per-token budget k (K = k·T_max, T_max = %d)" % t_max)
    ax.set_ylabel("%")
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.3)
    top = ax.secondary_xaxis("top", functions=(lambda k: k, lambda k: k))
    top.set_xticks(ks)
    top.set_xticklabels([f"1e{(k * t_max) / math.log(10):.0f}" for k in ks])
    top.set_xlabel("best-of-n selection permitted by the budget (n = e^K)")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(figures, "certificate_cap_curve.pdf"))
    fig.savefig(os.path.join(figures, "certificate_cap_curve.png"), dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--safe-model", default="jacquelinehe/tinycomma-1.8b-llama3-tokenizer")
    ap.add_argument("--risky-model", default="meta-llama/Llama-3.1-8B-Instruct", help="'' to skip")
    ap.add_argument("--k-values", nargs="+", type=float, default=[0.1, 0.15, 0.25, 0.5, 1.0, 3.0, 5.0])
    ap.add_argument("--t-max", type=int, default=200)
    ap.add_argument("--use-chat-template", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: passages per split")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.safe_model)
    prompts = [p for p in load_prompt_corpus(args.data, "factscore_prompt") if p.split in SPLITS and p.reference]
    if args.limit:
        prompts = [p for s in SPLITS for p in [q for q in prompts if q.split == s][:args.limit]]
    rows = [dict(prompt_id=p.prompt_id, split=p.split, novel_source=p.novel_source) for p in prompts]

    for tag, name in (("safe", args.safe_model), ("risky", args.risky_model)):
        if not name:
            continue
        print(f"[stage] loading {tag} model {name}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map={"": device}).eval()
        for i, p in enumerate(prompts):
            x = wrap_chat(p.prompt_text, tok) if args.use_chat_template else p.prompt_text
            S, n, first = surprisal(model, tok, x, p.reference, device)
            rows[i].update({f"S_{tag}": round(S, 3), "n_ref_tokens": n, f"S_{tag}_per_tok": round(S / n, 4), f"S_{tag}_first_tok": round(first, 3)})
            if i % 200 == 0:
                print(f"[stage] {tag} {i}/{len(prompts)} S={S:.1f} n={n}", flush=True)
        del model
        torch.cuda.empty_cache()

    Ks = [(k, k * args.t_max) for k in args.k_values]
    for r in rows:
        for k, K in Ks:
            r[f"cap_k{k:g}"] = round(cap_exact(r["S_safe"], K), 6)
    with open(os.path.join(args.out, "certificate_caps.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = []
    for k, K in Ks:
        for split in ("all",) + SPLITS:
            R = [r for r in rows if split == "all" or r["split"] == split]
            caps = [r[f"cap_k{k:g}"] for r in R]
            simple = [cap_simple(r["S_safe"], K) for r in R]
            row = dict(k=k, K=K, split=split, n=len(R),
                       vacuous_n=sum(r["S_safe"] <= K for r in R), vacuous_pct=round(100 * sum(r["S_safe"] <= K for r in R) / len(R), 2),
                       cap_median=round(st.median(caps), 4), cap_mean=round(st.mean(caps), 4), cap_min=round(min(caps), 6),
                       cap_simple_median=round(st.median(simple), 4),
                       bestofn_equiv_log10n=round(K / math.log(10), 1))
            if "S_risky" in R[0]:
                row["risky_prob_gt_cap_n"] = sum(math.exp(-r["S_risky"]) > r[f"cap_k{k:g}"] for r in R)
            summary.append(row)
    S_s = [r["S_safe"] for r in rows]
    print(f"S_safe: n={len(S_s)} median={st.median(S_s):.1f} p10={sorted(S_s)[len(S_s)//10]:.1f} p90={sorted(S_s)[9*len(S_s)//10]:.1f} per-token median={st.median([r['S_safe_per_tok'] for r in rows]):.3f}")
    if "S_risky" in rows[0]:
        S_r = [r["S_risky"] for r in rows]
        print(f"S_risky: median={st.median(S_r):.1f} per-token median={st.median([r['S_risky_per_tok'] for r in rows]):.3f}; median gap S_safe-S_risky={st.median([a-b for a,b in zip(S_s,S_r)]):.1f}")
    with open(os.path.join(args.out, "certificate_cap_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for row in summary:
        if row["split"] == "all":
            print(f"k={row['k']:g} K={row['K']:g}: vacuous {row['vacuous_pct']}% median cap {row['cap_median']} (best-of-n equiv. log10 n ~ {row['bestofn_equiv_log10n']})")

    plot(summary, args.figures, args.t_max)
    print("wrote", os.path.join(args.figures, "certificate_cap_curve.pdf"))


if __name__ == "__main__":
    main()
