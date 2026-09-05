"""Composition attack against per-prompt K-NAF (feat-009, contribution C5b; Cohen 2025 Thm 3.5 instantiated).

Every query stays within its own budget K = k * L, but an adversary who reconstructs a passage window by
window is not bounded by any single certificate. For each memorised excerpt (text = "Complete the prefix:\n"
+ prefix + reference of a CopyBench attack_train item) the first --seed-tokens tokens are the seed; the rest
is the target. Modes:
  single   one query, max_new_tokens = |target|                           (budget k*|target|, banking allowed)
  oracle   window i is queried with the TRUE text up to its start           (Cohen's oracle-prefix composition)
  chained  window i is queried with seed + everything extracted so far      (realistic, no oracle)
for window lengths L in --windows and k in --k-values (k=-1 risky only, k=0 safe only).
Per query we log Z, B, delta_init and check Z <= max(0, B) + 1e-3; per (passage, mode, L, k) we stitch the
outputs and score nv-recall, LCS (words), and the longest exactly reconstructed span against the target.

Outputs: results/composition.csv (per passage), results/composition_summary.csv (per mode/L/k),
figures/composition.{pdf,png}.
Usage: CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 .venv/bin/python analysis/composition_attack.py --risky-model output/memorizing_llama8b --limit 100
"""
import argparse, csv, json, math, os, statistics as st, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a_patch import AnchoredDecodingFactory  # noqa: E402
from dap.shared import load_prompt_corpus, true_gen_len  # noqa: E402
from dap.stats import lcs_word, nv_recall  # noqa: E402
from recipes.finetune_memorizing import join  # noqa: E402

EPS = 1e-3


def batches(xs, n):
    return [xs[i:i + n] for i in range(0, len(xs), n)]


class Attacker:
    def __init__(self, factory, tok, batch_size, temperature, seed):
        self.f, self.tok, self.bs, self.temp, self.seed = factory, tok, batch_size, temperature, seed

    def query(self, prompts, k, max_new):
        """Generate max_new tokens for each prompt at budget k; returns list of (text, Z, B, delta_init, n_tokens)."""
        from transformers import GenerationConfig
        out = []
        for chunk in batches(prompts, self.bs):
            cfg = GenerationConfig(do_sample=True, temperature=self.temp, max_new_tokens=max_new, num_return_sequences=1, num_beams=1,
                                   pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id)
            o = self.f.generate(text=chunk, generation_config=cfg, k_radius=k, seed=self.seed, parallelize=False, show_progress=False)
            stats = self.f.get_kl_stats_summary()
            Z, B = stats["final_cum_kl_spent_per_seq"], stats["final_budget_per_seq"]
            debt = stats["per_step"][0].get("prefix_debt") if stats["per_step"] else None
            enc = self.tok(chunk, return_tensors="pt", padding=True)
            plens = enc.attention_mask.sum(dim=1).tolist()
            seqs = o.sequences.detach().cpu()
            for j in range(len(chunk)):
                gen_ids = seqs[j].tolist()[int(plens[j]):]
                n = true_gen_len(gen_ids, [self.tok.pad_token_id, self.tok.eos_token_id])
                out.append((self.tok.decode(gen_ids[:n], skip_special_tokens=True), float(Z[j]), float(B[j]), float(debt[j]) if debt else None, n))
            torch.cuda.empty_cache()
        return out


def plot(summary, k_values, modes, windows, figures):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5})
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    pos = [k for k in k_values if k > 0]
    styles = {"single": ("o-", "C0"), "oracle": ("s--", "C3"), "chained": ("^-.", "C2")}

    def get(k, mode, L):
        return next((r["nv_recall_mean"] for r in summary if r["k"] == k and r["mode"] == mode and r["L"] == L), None)

    for mode in modes:
        for L in (windows if mode != "single" else [0]):
            fmt, col = styles.get(mode, ("x:", "C4"))
            alpha = 1.0 if L in (0, windows[0]) else 0.5
            ys = [get(k, mode, L) for k in pos]
            ax.plot(pos, ys, fmt, color=col, alpha=alpha, label=f"{mode}" + (f", L={L}" if L else ""))
            base = get(-1.0, mode, L)
            if base is not None:  # unconstrained risky model, same strategy
                ax.axhline(base, color=col, ls=":", lw=0.8, alpha=alpha)
    safe = get(0.0, "single", 0)
    if safe is not None:
        ax.axhline(safe, color="gray", ls="-", lw=0.8, label="anchor only (k=0)")
    ax.plot([], [], color="gray", ls=":", lw=0.8, label="dotted: same strategy, risky only (k=-1)")
    ax.set_xscale("log")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{k:g}" for k in pos])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("per-token budget k (every query within K = k·L)")
    ax.set_ylabel("nv-recall of the target (mean)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(figures, "composition.pdf"))
    fig.savefig(os.path.join(figures, "composition.png"), dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="attack_train")
    ap.add_argument("--safe-model", default="jacquelinehe/tinycomma-1.8b-llama3-tokenizer")
    ap.add_argument("--risky-model", required=True)
    ap.add_argument("--k-values", nargs="+", type=float, default=[-1, 0, 0.15, 0.5, 1, 3])
    ap.add_argument("--windows", nargs="+", type=int, default=[20, 50])
    ap.add_argument("--modes", nargs="+", default=["single", "oracle", "chained"])
    ap.add_argument("--seed-tokens", type=int, default=20)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--plot-only", action="store_true", help="re-plot from <out>/composition_summary.csv without running anything")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)
    if args.plot_only:
        summary = [{k: (float(v) if k not in ("mode",) else v) for k, v in r.items()} for r in csv.DictReader(open(os.path.join(args.out, "composition_summary.csv")))]
        for r in summary:
            r["L"] = int(r["L"])
        plot(summary, sorted({r["k"] for r in summary}), sorted({r["mode"] for r in summary}, key=["single", "oracle", "chained"].index), sorted({r["L"] for r in summary if r["L"]}), args.figures)
        return

    factory = AnchoredDecodingFactory.from_pretrained(safe_model_path=args.safe_model, risky_model_path=args.risky_model,
                                                      k_radius=max(0.0, args.k_values[0]), use_prefix_debt=True, prefix_n=5, log_kl_stats=True,
                                                      device="cuda", dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    tok = factory.tokenizer
    atk = Attacker(factory, tok, args.batch_size, args.temperature, args.seed)

    prompts = [p for p in load_prompt_corpus(args.data, "factscore_prompt") if p.split == args.split and p.reference][:args.limit]
    passages = []
    for p in prompts:
        ids = tok(join(p.prompt_text, p.reference)).input_ids
        seed_txt = tok.decode(ids[:args.seed_tokens], skip_special_tokens=True)
        passages.append(dict(prompt_id=p.prompt_id, novel=p.novel_source, ids=ids, seed=seed_txt,
                             target=tok.decode(ids[args.seed_tokens:], skip_special_tokens=True), n_target=len(ids) - args.seed_tokens))
    print(f"[ca] {len(passages)} passages; target length mean {st.mean(x['n_target'] for x in passages):.0f} tokens", flush=True)

    rows, t0 = [], time.time()
    for k in args.k_values:
        K_per_tok = k
        for mode in args.modes:
            for L in (args.windows if mode != "single" else [0]):
                queries = [[] for _ in passages]  # per passage: list of (Z, B, K, debt, n)
                stitched = [""] * len(passages)
                if mode == "single":
                    T = max(x["n_target"] for x in passages)
                    res = atk.query([x["seed"] for x in passages], k, T)
                    for i, (txt, Z, B, d, n) in enumerate(res):
                        stitched[i] = txt
                        queries[i].append((Z, B, K_per_tok * T, d, n))
                else:
                    n_win = max(math.ceil(x["n_target"] / L) for x in passages)
                    for w in range(n_win):
                        active = [i for i, x in enumerate(passages) if w * L < x["n_target"]]
                        if not active:
                            break
                        if mode == "oracle":
                            ps = [tok.decode(passages[i]["ids"][:args.seed_tokens + w * L], skip_special_tokens=True) for i in active]
                        else:
                            ps = [passages[i]["seed"] + stitched[i] for i in active]
                        res = atk.query(ps, k, L)
                        for i, (txt, Z, B, d, n) in zip(active, res):
                            stitched[i] += txt
                            queries[i].append((Z, B, K_per_tok * L, d, n))
                for i, x in enumerate(passages):
                    q = queries[i]
                    Zs = [z for z, _, _, _, _ in q]
                    Ks = [kk for _, _, kk, _, _ in q]
                    viol = sum(z > max(0.0, b) + EPS for z, b, _, _, _ in q)
                    rows.append(dict(k=k, mode=mode, L=L, prompt_id=x["prompt_id"], novel=x["novel"], n_target_tokens=x["n_target"], n_queries=len(q),
                                     spend_total=round(sum(Zs), 3), spend_max_query=round(max(Zs), 3),
                                     budget_K_per_query=round(Ks[0], 3) if Ks else None,
                                     max_query_Z_over_K=round(max((z / kk) for z, kk in zip(Zs, Ks)), 4) if k > 0 else None,
                                     invariant_violations=viol, delta_init_mean=round(st.mean(d for _, _, _, d, _ in q if d is not None), 3) if any(d is not None for _, _, _, d, _ in q) else None,
                                     tokens_generated=sum(n for _, _, _, _, n in q),
                                     nv_recall=round(nv_recall(stitched[i], x["target"]), 4), lcs_word=lcs_word(stitched[i], x["target"]),
                                     rouge_words=None, extracted=stitched[i][:2000]))
                done = [r for r in rows if r["k"] == k and r["mode"] == mode and r["L"] == L]
                print(f"[ca] k={k:g} {mode} L={L}: nv-recall mean {st.mean(r['nv_recall'] for r in done):.3f}, LCS words mean {st.mean(r['lcs_word'] for r in done):.1f}, "
                      f"violations {sum(r['invariant_violations'] for r in done)}, max Z/K {max((r['max_query_Z_over_K'] or 0) for r in done):.3f} ({time.time() - t0:.0f}s)", flush=True)

    with open(os.path.join(args.out, "composition.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = []
    for k in args.k_values:
        for mode in args.modes:
            for L in (args.windows if mode != "single" else [0]):
                R = [r for r in rows if r["k"] == k and r["mode"] == mode and r["L"] == L]
                nv = [r["nv_recall"] for r in R]
                summary.append(dict(k=k, mode=mode, L=L, n_passages=len(R), n_queries_mean=round(st.mean(r["n_queries"] for r in R), 1),
                                    nv_recall_mean=round(st.mean(nv), 4), nv_recall_median=round(st.median(nv), 4), nv_recall_ge_0p8_pct=round(100 * sum(x >= 0.8 for x in nv) / len(nv), 1),
                                    lcs_word_mean=round(st.mean(r["lcs_word"] for r in R), 2), lcs_word_max=max(r["lcs_word"] for r in R),
                                    spend_total_mean=round(st.mean(r["spend_total"] for r in R), 2), max_query_Z_over_K=max((r["max_query_Z_over_K"] or 0) for r in R),
                                    invariant_violations=sum(r["invariant_violations"] for r in R)))
    with open(os.path.join(args.out, "composition_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    plot(summary, args.k_values, args.modes, args.windows, args.figures)
    print("wrote", os.path.join(args.out, "composition.csv"), os.path.join(args.figures, "composition.pdf"))


if __name__ == "__main__":
    main()
