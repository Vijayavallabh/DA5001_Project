"""feat-033 (plan v3, C19): how the certificate's vacuity depends on the length of the protected work.

Section V-C measures the cap on reproduction for 64-token CopyBench references and finds it vacuous
(S(x) <= K) for every passage at k >= 3. The Limitations call that a lower bound, because the anchor's
surprisal of a work grows with its length while the published budget K = k*T_max does not. This script
replaces the caveat with a curve, and separates two questions the caveat runs together:

  whole-work    is S(x) <= K for a passage of n tokens, with K fixed by the deployment (k * T_max)?
                Vacuity must fall with n, and the curve says how fast.
  per-window    is S_w <= k*W for each W-token window of the same work, given the true prefix?
                This is the certificate the windowed adversary of Section VI-B actually faces, and it
                does not improve with the length of the work at all.

Both come from one forward pass per work: the per-token surprisal vector gives every prefix length and
every window sum without re-running the model.

Two sources, because neither alone is clean:
  gutenberg  50 public-domain books (data/gutenberg/, cached, re-fetchable). Real contiguous prose of any
             length, but the Common Pile includes Project Gutenberg, so the anchor has likely read them
             and their surprisal is low. Good for the shape of the curve.
  copybench  the 16 copyrighted novels, excerpts concatenated within a novel. Anchor-unfamiliar, but the
             excerpts are not contiguous, so each block restarts without its true context and its
             surprisal is if anything overstated. Good for the level.
Agreement between the two on the exponent is the check; neither is asked to carry the result alone.

Reads:  data/gutenberg/*.txt (via analysis.latent_leakage.fetch_gutenberg), data/copybench_*.jsonl
Writes: <out>/length_scaling.csv (per work and length), <out>/length_scaling_summary.csv (per source,
        length, k), <figures>/length_scaling.{pdf,png} if --figures is given.
Usage:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 \
          .venv/bin/python analysis/length_scaling.py --out results --figures figures
"""
import argparse, csv, os, statistics as st, sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.certificate_cap import SPLITS, cap_exact  # noqa: E402
from analysis.latent_leakage import GUTENBERG, fetch_gutenberg  # noqa: E402
from dap.shared import load_prompt_corpus  # noqa: E402

LENGTHS = [64, 128, 256, 512, 1024, 2048]
PROMPT_TOKENS = 100  # the raw seed of the Cooper et al. protocol used elsewhere in the audit


@torch.no_grad()
def per_token_surprisal(model, tok, ids, device, prompt_len):
    """-ln p_s(y_t | y_<t) for every token after the prompt, in one pass."""
    t = torch.tensor([ids], device=device)
    logits = model(t).logits[0, :-1].float()
    logp = torch.log_softmax(logits[prompt_len - 1:], dim=-1)
    tgt = t[0, prompt_len:].unsqueeze(1)
    return (-logp.gather(1, tgt).squeeze(1)).tolist()


def works(args, tok):
    """(source, name, token ids) for each work, prompt included, truncated to the context window."""
    out = []
    for gid, title in GUTENBERG.items():
        try:
            body = fetch_gutenberg(gid, args.cache)
        except Exception as e:                                  # cached copies only; never block on the network
            print(f"[skip] gutenberg {gid}: {type(e).__name__}", flush=True)
            continue
        body = body[len(body) // 4:]                            # a quarter in: past front matter and any preface
        ids = tok(body[:80_000], add_special_tokens=True).input_ids
        if len(ids) >= PROMPT_TOKENS + LENGTHS[0]:
            out.append(("gutenberg", title, ids))
    by_novel = defaultdict(list)
    for p in load_prompt_corpus(args.data, "factscore_prompt"):
        if p.split in SPLITS and p.novel_source and p.reference:   # book splits only; other classes carry no novel
            by_novel[p.novel_source].append((p.prompt_id, p.reference))
    for novel, items in sorted(by_novel.items()):
        text = " ".join(r for _, r in sorted(items))            # excerpts in id order; not contiguous, see docstring
        ids = tok(text, add_special_tokens=True).input_ids
        if len(ids) >= PROMPT_TOKENS + LENGTHS[0]:
            out.append(("copybench", novel, ids))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--cache", default="data/gutenberg")
    ap.add_argument("--out", default="results")
    ap.add_argument("--figures", default="")
    ap.add_argument("--safe-model", default="jacquelinehe/tinycomma-1.8b-llama3-tokenizer")
    ap.add_argument("--k-values", nargs="+", type=float, default=[1.0, 3.0, 5.0, 10.0, 20.0])
    ap.add_argument("--t-max", type=int, default=200, help="the deployment's T_max: K = k * T_max is fixed by it")
    ap.add_argument("--window", type=int, default=50, help="window length of the composition attack")
    ap.add_argument("--replot", action="store_true", help="rebuild the figure from the summary CSV; no model, no GPU")
    args = ap.parse_args()

    if args.replot:
        summary = [{k: (v if k == "source" else float(v)) for k, v in r.items()}
                   for r in csv.DictReader(open(os.path.join(args.out, "length_scaling_summary.csv")))]
        lengths = sorted({int(s["n_tokens"]) for s in summary})
        for s in summary:
            s["n_tokens"] = int(s["n_tokens"])
        plot(summary, lengths, sorted({s["k"] for s in summary}), args.figures or "figures")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.safe_model)
    model = AutoModelForCausalLM.from_pretrained(args.safe_model, dtype=torch.bfloat16,
                                                 device_map={"": device}).eval()
    ctx = getattr(model.config, "max_position_embeddings", 4096)
    lengths = [n for n in LENGTHS if PROMPT_TOKENS + n <= ctx]
    print(f"[stage] context {ctx} tokens -> lengths {lengths}", flush=True)

    rows = []
    items = works(args, tok)
    for i, (source, name, ids) in enumerate(items):
        ids = ids[:PROMPT_TOKENS + max(lengths)]
        s = per_token_surprisal(model, tok, ids, device, PROMPT_TOKENS)
        for n in lengths:
            if len(s) < n:
                continue
            pre = s[:n]
            wins = [sum(pre[j:j + args.window]) for j in range(0, n - args.window + 1, args.window)]
            rows.append(dict(source=source, work=name, n_tokens=n,
                             S=round(sum(pre), 2), S_per_tok=round(sum(pre) / n, 4),
                             n_windows=len(wins), S_window_median=round(st.median(wins), 2)))
        if i % 10 == 0:
            print(f"[stage] {i}/{len(items)} {source} {name[:40]}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "length_scaling.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    summary = []
    for source in ("gutenberg", "copybench"):
        for n in lengths:
            R = [r for r in rows if r["source"] == source and r["n_tokens"] == n]
            if not R:
                continue
            for k in args.k_values:
                K, Kw = k * args.t_max, k * args.window
                summary.append(dict(
                    source=source, n_tokens=n, k=k, n_works=len(R),
                    S_median=round(st.median([r["S"] for r in R]), 1),
                    whole_work_vacuous_pct=round(100 * sum(r["S"] <= K for r in R) / len(R), 1),
                    cap_median=round(st.median([cap_exact(r["S"], K) for r in R]), 4),
                    per_window_vacuous_pct=round(100 * sum(r["S_window_median"] <= Kw for r in R) / len(R), 1),
                ))
    with open(os.path.join(args.out, "length_scaling_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader(); w.writerows(summary)

    for source in ("gutenberg", "copybench"):
        print(f"\n{source}: whole-work vacuity (%) by length, and the per-window certificate at the same k")
        print(f"{'k':>5} " + " ".join(f"{n:>7}" for n in lengths) + " | per-window (50 tok)")
        for k in args.k_values:
            cells, win = [], "-"
            for n in lengths:
                m = [s for s in summary if s["source"] == source and s["n_tokens"] == n and s["k"] == k]
                cells.append(f"{m[0]['whole_work_vacuous_pct']:>7.1f}" if m else f"{'-':>7}")
                if m:
                    win = f"{m[0]['per_window_vacuous_pct']:.1f}"
            print(f"{k:>5g} " + " ".join(cells) + f" | {win}")
    if args.figures:
        plot(summary, lengths, args.k_values, args.figures)
    print("\nwrote", os.path.join(args.out, "length_scaling.csv"), "and length_scaling_summary.csv")


def plot(summary, lengths, ks, figures):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7})
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    cols = plt.cm.viridis([i / max(1, len(ks) - 1) * 0.85 for i in range(len(ks))])
    for k, c in zip(ks, cols):
        for source, ls, mk in (("copybench", "-", "o"), ("gutenberg", "--", "s")):
            pts = [(n, s["whole_work_vacuous_pct"]) for n in lengths
                   for s in summary if s["source"] == source and s["n_tokens"] == n and s["k"] == k]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, marker=mk, ms=3, lw=1.2, color=c,
                        label=(f"$k={k:g}$" if source == "copybench" else None))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("length of the protected work (tokens)")
    ax.set_ylabel("certificate vacuous (% of works)")
    ax.set_ylim(-4, 104)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=5,
              columnspacing=1.0, handlelength=1.4,
              title="solid: CopyBench novels    dashed: Gutenberg public domain",
              title_fontsize=6)
    fig.tight_layout(pad=0.45)
    os.makedirs(figures, exist_ok=True)
    fig.savefig(os.path.join(figures, "length_scaling.pdf"))
    fig.savefig(os.path.join(figures, "length_scaling.png"), dpi=150)
    print("wrote", os.path.join(figures, "length_scaling.pdf"))


if __name__ == "__main__":
    main()
