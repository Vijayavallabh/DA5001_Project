"""Composition attack against per-prompt K-NAF (feat-009, contribution C5b; Cohen 2025 Thm 3.5 instantiated),
extended for plan v2: --constraint pathwise (feat-019), --retries (extraction-cost measurement, Proposition 2),
per-query logging for the odometer analysis (feat-021), --no-prefix-debt (feat-025), --novel filter (feat-018).

Every query stays within its own budget K = k * L, but an adversary who reconstructs a passage window by
window is not bounded by any single certificate. For each memorised excerpt (text = "Complete the prefix:\n"
+ prefix + reference of a CopyBench item) the first --seed-tokens tokens are the seed; the rest is the target.
Modes:
  single   one query, max_new_tokens = |target|                           (budget k*|target|, banking allowed)
  oracle   window i is queried with the TRUE text up to its start           (Cohen's oracle-prefix composition)
  chained  window i is queried with seed + everything extracted so far      (realistic, no oracle)
for window lengths L in --windows and k in --k-values (k=-1 risky only, k=0 safe only).
With --retries N > 1, oracle windows are re-sampled (fresh seed per attempt) until the output matches the true
window exactly (whitespace-normalised) or N attempts are used; attempts per window are logged, which is the
empirical extraction cost that Proposition 2 lower-bounds by exp(S_i - k L_i) under the pathwise decoder.
Per query we log Z, B, R (realised log-ratio), delta_init and check Z <= max(0, B) + 1e-3 (and R <= max(0, B) + 1e-3
under --constraint pathwise); per (passage, mode, L, k) we stitch the outputs and score nv-recall, LCS (words),
and the longest exactly reconstructed span against the target.

Outputs: <out>/composition.csv (per passage), <out>/composition_summary.csv (per mode/L/k),
<figures>/composition.{pdf,png}, --queries-out (one JSON line per query: spend, budget, ratio, attempt, matched, text),
--text-out (stitched texts).
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


def normalise(s: str) -> str:
    return " ".join(s.split())


def window_matches(hypothesis: str, reference: str) -> bool:
    """Exact reproduction of a window up to whitespace; the event of Proposition 2."""
    h = normalise(hypothesis)
    return h != "" and h == normalise(reference)


class Attacker:
    def __init__(self, factory, tok, batch_size, temperature, seed, repetition_penalty=1.0, greedy=False):
        self.f, self.tok, self.bs, self.temp, self.seed, self.rp, self.greedy = factory, tok, batch_size, temperature, seed, repetition_penalty, greedy

    def query(self, prompts, k, max_new, seed_offset=0):
        """Generate max_new tokens for each prompt at budget k; returns list of (text, Z, B, delta_init, n_tokens, R)."""
        from transformers import GenerationConfig
        out = []
        self.last_activity = []  # per returned query: decode-step counts (forced to the anchor / free = risky unchanged / total)
        for chunk in batches(prompts, self.bs):
            cfg = GenerationConfig(do_sample=not self.greedy, temperature=self.temp, max_new_tokens=max_new, num_return_sequences=1, num_beams=1,
                                   repetition_penalty=self.rp, pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id)
            o = self.f.generate(text=chunk, generation_config=cfg, k_radius=k, seed=self.seed + seed_offset, parallelize=False, show_progress=False)
            stats = self.f.get_kl_stats_summary()
            Z, B = stats["final_cum_kl_spent_per_seq"], stats["final_budget_per_seq"]
            R = stats.get("final_realised_ratio_per_seq") or [0.0] * len(chunk)
            debt = stats["per_step"][0].get("prefix_debt") if stats["per_step"] else None
            bd_steps = [s.get("bd") for s in stats["per_step"] if s.get("bd") is not None]  # per step: list over the batch
            enc = self.tok(chunk, return_tensors="pt", padding=True)
            plens = enc.attention_mask.sum(dim=1).tolist()
            seqs = o.sequences.detach().cpu()
            for j in range(len(chunk)):
                gen_ids = seqs[j].tolist()[int(plens[j]):]
                n = true_gen_len(gen_ids, [self.tok.pad_token_id, self.tok.eos_token_id])
                bdj = [b[j] for b in bd_steps[:n]]
                act = dict(steps=len(bdj), forced=sum(x <= 1e-6 for x in bdj), free=sum(x >= 1 - 1e-6 for x in bdj))
                self.last_activity.append(act)
                out.append((self.tok.decode(gen_ids[:n], skip_special_tokens=True), float(Z[j]), float(B[j]), float(debt[j]) if debt else None, n, float(R[j])))
            torch.cuda.empty_cache()
        return out


def plot(summary, k_values, modes, windows, figures):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.2,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    pos = [k for k in k_values if k > 0]
    if not pos:  # baselines only: nothing to plot on a log axis
        return
    styles = {"single": ("o-", "C0"), "oracle": ("s--", "C3"), "chained": ("^-.", "C2")}

    def get(k, mode, L):
        return next((r["nv_recall_mean"] for r in summary if r["k"] == k and r["mode"] == mode and r["L"] == L), None)

    for mode in modes:
        for L in (windows if mode != "single" else [0]):
            fmt, col = styles.get(mode, ("x:", "C4"))
            alpha = 1.0 if L in (0, windows[0]) else 0.5
            ys = [get(k, mode, L) for k in pos]
            ax.plot(pos, ys, fmt, color=col, alpha=alpha, lw=1.3, ms=3.2, label=f"{mode}" + (f", $L={L}$" if L else ""))
            base = get(-1.0, mode, L)
            if base is not None:  # unconstrained risky model, same strategy
                ax.axhline(base, color=col, ls=":", lw=0.8, alpha=alpha)
    safe = get(0.0, "single", 0)
    if safe is not None:
        ax.axhline(safe, color="gray", ls="-", lw=0.8, label="anchor only (k=0)")
    ax.plot([], [], color="gray", ls=":", lw=0.8, label="dotted: same strategy, risky model alone")
    ax.set_xscale("log")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{k:g}" for k in pos])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("per-token budget $k$")
    ax.set_ylabel("near-verbatim recall")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(figures, "composition.pdf"))
    fig.savefig(os.path.join(figures, "composition.png"), dpi=150)


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="attack_train")
    ap.add_argument("--novel", default="", help="keep only passages whose novel name contains this substring (e.g. harry_potter)")
    ap.add_argument("--safe-model", default="jacquelinehe/tinycomma-1.8b-llama3-tokenizer")
    ap.add_argument("--risky-model", required=True)
    ap.add_argument("--k-values", nargs="+", type=float, default=[-1, 0, 0.15, 0.5, 1, 3])
    ap.add_argument("--windows", nargs="+", type=int, default=[20, 50])
    ap.add_argument("--modes", nargs="+", default=["single", "oracle", "chained"])
    ap.add_argument("--seed-tokens", type=int, default=20)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0, help="applied to both models before the solve, as in He et al. (their books setting: 0.7 / 1.1)")
    ap.add_argument("--constraint", choices=["kl", "pathwise"], default="kl", help="feat-019: KL budget (He et al.) or pathwise max-divergence budget")
    ap.add_argument("--bank-cap", type=float, default=None, help="feat-021: token-bucket depth in nats (unset = unbounded bank)")
    ap.add_argument("--no-prefix-debt", action="store_true", help="feat-025: delta_init = 0")
    ap.add_argument("--raw-prompt", action="store_true", help="feat-018: drop the 'Complete the prefix:' instruction header and seed with the raw passage text (base models)")
    ap.add_argument("--greedy", action="store_true", help="argmax decoding; only for the baselines k in {-1, 0}")
    ap.add_argument("--retries", type=int, default=1, help="oracle windows: re-sample up to N times until the window is reproduced exactly (extraction cost)")
    ap.add_argument("--max-memory", default="", help="per-device cap for large risky models, e.g. '0=72GiB,1=64GiB' (feat-017)")
    ap.add_argument("--risky-device-map", default="", help="'auto' to shard a large risky model across the visible GPUs; anchor stays on cuda:1 (feat-017)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--text-out", default="output/composition/composition_extracted.csv", help="where the stitched texts go (kept out of results/)")
    ap.add_argument("--queries-out", default="output/composition/queries.jsonl", help="one JSON line per query (feat-021 odometer input)")
    ap.add_argument("--plot-only", action="store_true", help="re-plot from <out>/composition_summary.csv without running anything")
    args = ap.parse_args()
    if args.greedy and any(k not in (-1.0, 0.0) for k in args.k_values):
        raise SystemExit("--greedy is only meaningful for the baselines k in {-1, 0}; anchored decoding requires sampling")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)
    if args.plot_only:
        summary = [{k: (_num(v) if k not in ("mode", "constraint") else v) for k, v in r.items()} for r in csv.DictReader(open(os.path.join(args.out, "composition_summary.csv")))]
        for r in summary:
            r["L"] = int(r["L"])
        plot(summary, sorted({r["k"] for r in summary}), sorted({r["mode"] for r in summary}, key=["single", "oracle", "chained"].index), sorted({r["L"] for r in summary if r["L"]}), args.figures)
        return

    max_memory = None
    if args.max_memory:
        max_memory = {int(a): b for a, b in (kv.split("=") for kv in args.max_memory.split(","))}
    factory = AnchoredDecodingFactory.from_pretrained(safe_model_path=args.safe_model, risky_model_path=args.risky_model,
                                                      k_radius=max(0.0, args.k_values[0]), use_prefix_debt=not args.no_prefix_debt, prefix_n=5, log_kl_stats=True,
                                                      constraint=args.constraint, bank_cap=args.bank_cap, device="cuda", dtype=torch.bfloat16, device_map="auto",
                                                      max_memory=max_memory, risky_device_map=(args.risky_device_map or None), trust_remote_code=True)
    tok = factory.tokenizer
    atk = Attacker(factory, tok, args.batch_size, args.temperature, args.seed, args.repetition_penalty, greedy=args.greedy)

    prompts = [p for p in load_prompt_corpus(args.data, "factscore_prompt") if p.split == args.split and p.reference and (args.novel in (p.novel_source or ""))][:args.limit]
    passages = []
    HEADER = "Complete the prefix:\n"
    for p in prompts:
        prefix_text = p.prompt_text[len(HEADER):] if (args.raw_prompt and p.prompt_text.startswith(HEADER)) else p.prompt_text
        ids = tok(join(prefix_text, p.reference)).input_ids
        seed_txt = tok.decode(ids[:args.seed_tokens], skip_special_tokens=True)
        passages.append(dict(prompt_id=p.prompt_id, novel=p.novel_source, ids=ids, seed=seed_txt,
                             target=tok.decode(ids[args.seed_tokens:], skip_special_tokens=True), n_target=len(ids) - args.seed_tokens))
    print(f"[ca] {len(passages)} passages; target length mean {st.mean(x['n_target'] for x in passages):.0f} tokens; seed {args.seed_tokens} tokens raw_prompt={args.raw_prompt} greedy={args.greedy}; constraint={args.constraint} "
          f"prefix_debt={not args.no_prefix_debt} temperature={args.temperature} rp={args.repetition_penalty} retries={args.retries}", flush=True)

    os.makedirs(os.path.dirname(args.queries_out) or ".", exist_ok=True)
    qf = open(args.queries_out, "w")

    def log_query(**kw):
        qf.write(json.dumps(kw) + "\n")

    rows, t0 = [], time.time()
    for k in args.k_values:
        K_per_tok = k
        for mode in args.modes:
            for L in (args.windows if mode != "single" else [0]):
                queries = [[] for _ in passages]  # per passage: list of (Z, B, K, debt, n, R)
                stitched = [""] * len(passages)
                attempts_total = [0] * len(passages)
                matched_windows = [0] * len(passages)
                n_windows = [0] * len(passages)
                if mode == "single":
                    T = max(x["n_target"] for x in passages)
                    res = atk.query([x["seed"] for x in passages], k, T)
                    for i, (txt, Z, B, d, n, R) in enumerate(res):
                        stitched[i] = txt
                        queries[i].append((Z, B, K_per_tok * T, d, n, R))
                        attempts_total[i] += 1
                        n_windows[i] += 1
                        log_query(k=k, mode=mode, L=L, prompt_id=passages[i]["prompt_id"], window=0, attempt=1, Z=Z, B=B, K=K_per_tok * T, delta_init=d, R=R, n_tokens=n, matched=None, text=txt, **atk.last_activity[i])
                else:
                    n_win = max(math.ceil(x["n_target"] / L) for x in passages)
                    for w in range(n_win):
                        active = [i for i, x in enumerate(passages) if w * L < x["n_target"]]
                        if not active:
                            break
                        if mode == "oracle":
                            ps = [tok.decode(passages[i]["ids"][:args.seed_tokens + w * L], skip_special_tokens=True) for i in active]
                            truths = [tok.decode(passages[i]["ids"][args.seed_tokens + w * L:args.seed_tokens + (w + 1) * L], skip_special_tokens=True) for i in active]
                        else:
                            ps = [passages[i]["seed"] + stitched[i] for i in active]
                            truths = [None] * len(active)
                        pending = list(range(len(active)))
                        last_txt = [""] * len(active)
                        for attempt in range(1, (args.retries if mode == "oracle" else 1) + 1):
                            if not pending:
                                break
                            res = atk.query([ps[j] for j in pending], k, L, seed_offset=(attempt - 1) * 1000)
                            acts = list(atk.last_activity)
                            still = []
                            for jj, (j, (txt, Z, B, d, n, R)) in enumerate(zip(pending, res)):
                                i = active[j]
                                attempts_total[i] += 1
                                queries[i].append((Z, B, K_per_tok * L, d, n, R))
                                matched = window_matches(txt, truths[j]) if truths[j] is not None else None
                                log_query(k=k, mode=mode, L=L, prompt_id=passages[i]["prompt_id"], window=w, attempt=attempt, Z=Z, B=B, K=K_per_tok * L, delta_init=d, R=R, n_tokens=n, matched=matched, text=txt, **acts[jj])
                                last_txt[j] = txt
                                if matched:
                                    matched_windows[i] += 1
                                    stitched[i] += txt
                                else:
                                    still.append(j)
                            pending = still
                        for j in pending:  # never matched (or no oracle): keep the last attempt
                            stitched[active[j]] += last_txt[j]
                        for i in active:
                            n_windows[i] += 1
                for i, x in enumerate(passages):
                    q = queries[i]
                    Zs = [z for z, _, _, _, _, _ in q]
                    Ks = [kk for _, _, kk, _, _, _ in q]
                    Rs = [r for _, _, _, _, _, r in q]
                    # the guarantee the decoder enforces: KL spend Z (He et al.) or realised log-ratio R (pathwise, Proposition 1);
                    # under the pathwise decoder Z is not bounded per trajectory (only its expectation is), so it is not checked
                    if args.constraint == "pathwise":
                        viol = sum(r > max(0.0, b) + EPS for _, b, _, _, _, r in q) if k > 0 else 0
                    else:
                        viol = sum(z > max(0.0, b) + EPS for z, b, _, _, _, _ in q)
                    rows.append(dict(k=k, mode=mode, L=L, constraint=args.constraint, prompt_id=x["prompt_id"], novel=x["novel"], n_target_tokens=x["n_target"], n_queries=len(q),
                                     spend_total=round(sum(Zs), 3), spend_max_query=round(max(Zs), 3),
                                     budget_K_per_query=round(Ks[0], 3) if Ks else None,
                                     max_query_Z_over_K=round(max((z / kk) for z, kk in zip(Zs, Ks)), 4) if k > 0 else None,
                                     R_total=round(sum(Rs), 3), max_query_R_over_K=round(max((r / kk) for r, kk in zip(Rs, Ks)), 4) if k > 0 else None,
                                     invariant_violations=viol, delta_init_mean=round(st.mean(d for _, _, _, d, _, _ in q if d is not None), 3) if any(d is not None for _, _, _, d, _, _ in q) else None,
                                     tokens_generated=sum(n for _, _, _, _, n, _ in q),
                                     attempts_total=attempts_total[i], windows_total=n_windows[i], windows_matched=matched_windows[i],
                                     nv_recall=round(nv_recall(stitched[i], x["target"]), 4), lcs_word=lcs_word(stitched[i], x["target"]),
                                     extracted=stitched[i][:2000]))
                done = [r for r in rows if r["k"] == k and r["mode"] == mode and r["L"] == L]
                print(f"[ca] k={k:g} {mode} L={L}: nv-recall mean {st.mean(r['nv_recall'] for r in done):.3f}, LCS words mean {st.mean(r['lcs_word'] for r in done):.1f}, "
                      f"violations {sum(r['invariant_violations'] for r in done)}, max Z/K {max((r['max_query_Z_over_K'] or 0) for r in done):.3f}, "
                      f"attempts/window {sum(r['attempts_total'] for r in done) / max(1, sum(r['windows_total'] for r in done)):.2f} ({time.time() - t0:.0f}s)", flush=True)
    qf.close()

    os.makedirs(os.path.dirname(args.text_out) or ".", exist_ok=True)
    with open(args.text_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "mode", "L", "prompt_id", "extracted"])
        w.writeheader()
        w.writerows([{c: r[c] for c in ["k", "mode", "L", "prompt_id", "extracted"]} for r in rows])
    metric_cols = [c for c in rows[0] if c != "extracted"]
    with open(os.path.join(args.out, "composition.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=metric_cols)
        w.writeheader()
        w.writerows([{c: r[c] for c in metric_cols} for r in rows])
    summary = []
    for k in args.k_values:
        for mode in args.modes:
            for L in (args.windows if mode != "single" else [0]):
                Rws = [r for r in rows if r["k"] == k and r["mode"] == mode and r["L"] == L]
                nv = [r["nv_recall"] for r in Rws]
                summary.append(dict(k=k, mode=mode, L=L, constraint=args.constraint, n_passages=len(Rws), n_queries_mean=round(st.mean(r["n_queries"] for r in Rws), 1),
                                    nv_recall_mean=round(st.mean(nv), 4), nv_recall_median=round(st.median(nv), 4), nv_recall_ge_0p8_pct=round(100 * sum(x >= 0.8 for x in nv) / len(nv), 1),
                                    lcs_word_mean=round(st.mean(r["lcs_word"] for r in Rws), 2), lcs_word_max=max(r["lcs_word"] for r in Rws),
                                    spend_total_mean=round(st.mean(r["spend_total"] for r in Rws), 2), max_query_Z_over_K=max((r["max_query_Z_over_K"] or 0) for r in Rws),
                                    R_total_mean=round(st.mean(r["R_total"] for r in Rws), 2), max_query_R_over_K=max((r["max_query_R_over_K"] or 0) for r in Rws),
                                    attempts_per_window=round(sum(r["attempts_total"] for r in Rws) / max(1, sum(r["windows_total"] for r in Rws)), 3),
                                    windows_matched_pct=round(100 * sum(r["windows_matched"] for r in Rws) / max(1, sum(r["windows_total"] for r in Rws)), 1),
                                    invariant_violations=sum(r["invariant_violations"] for r in Rws)))
    with open(os.path.join(args.out, "composition_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    plot(summary, args.k_values, args.modes, args.windows, args.figures)
    print("wrote", os.path.join(args.out, "composition.csv"), os.path.join(args.figures, "composition.pdf"))


if __name__ == "__main__":
    main()
