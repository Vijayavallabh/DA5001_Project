"""Plan v5 / feat-054: the law without a fitted normaliser.

Section 4 rescales the budget by a single number per pair, the median anchor surprisal rate s(x),
and the single-query curves collapse. The oracle-window curves do not (spread 0.057 against 0.005),
and the paper's explanation is that a per-window budget K_i = k L charges the window's surprisal and
not the work's. That explanation makes a sharper prediction than "use a different median", and this
script tests it.

Proposition 1 says a budget of K buys the atom {output = x} only once K reaches S(x). A budget rate
k therefore covers exactly those protected objects whose surprisal RATE is at most k -- passages in
the single-query attack, 50-token windows in the oracle attack, because that is the object each
query is charged for. So define, per pair,

    F(k) = fraction of the protected objects whose anchor surprisal rate is at most k,

and plot recall against F(k) instead of against k/s(x). F uses the whole distribution rather than
its median and has no free parameter and nothing fitted. If the law is really about surprisal being
the unit, both attacks should collapse in this coordinate, each against its own object.

Writes <out>/surprisal_cdf.csv (the distributions) and <out>/cdf_collapse.csv (the comparison).

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/surprisal_cdf.py --out results
"""
import argparse, bisect, csv, os, statistics as st, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.onset import curve  # noqa: E402
from analysis.regimes import token_nats  # noqa: E402
from dap.shared import load_prompt_corpus  # noqa: E402
from recipes.finetune_memorizing import join  # noqa: E402

SEED_TOKENS = 20        # matches analysis/budget_path.py
GRID = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def load_pairs(path):
    """name<TAB>composition_summary.csv<TAB>budget_path.csv<TAB>anchor model id[<TAB>...]."""
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 4 or not f[3]:
            print(f"[cdf] no anchor model for {f[0]!r}, skipping", file=sys.stderr)
            continue
        out.append((f[0], f[1], f[2], f[3]))
    return out


def rates(nats, L):
    """Mean surprisal rate of every consecutive length-L block; the whole sequence when L is 0.

    Blocks are consecutive and non-overlapping because that is how the oracle attack splits a
    passage: window i is queried with the true text up to its start.
    """
    if L <= 0:
        return [sum(nats) / len(nats)] if nats else []
    return [sum(nats[i:i + L]) / L for i in range(0, len(nats) - L + 1, L)]


def cdf_at(sorted_vals, k):
    return bisect.bisect_right(sorted_vals, k) / len(sorted_vals) if sorted_vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--data", default="data")
    ap.add_argument("--pairs-file", default="results/onset_pairs.tsv")
    ap.add_argument("--window", type=int, default=50, help="oracle window length, in tokens")
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = {p.prompt_id: p for p in load_prompt_corpus(a.data, "factscore_prompt") if p.reference}

    dist, rows = {}, []
    for name, comp, bp, anchor in load_pairs(a.pairs_file):
        ids = [r["prompt_id"] for r in csv.DictReader(open(bp))]
        tok = AutoTokenizer.from_pretrained(anchor)
        model = AutoModelForCausalLM.from_pretrained(anchor, dtype=getattr(torch, a.dtype)).to(dev).eval()
        passage, window = [], []
        for pid in ids:
            p = corpus.get(pid)
            if p is None:
                continue
            text = join(p.prompt_text, p.reference)
            ids_ = tok(text, add_special_tokens=True).input_ids
            prefix = tok.decode(ids_[:SEED_TOKENS], skip_special_tokens=True)
            target = tok.decode(ids_[SEED_TOKENS:], skip_special_tokens=True)
            nats, _ = token_nats(model, tok, prefix, target, dev)
            if not nats:
                continue
            passage += rates(nats, 0)
            window += rates(nats, a.window)
        del model
        torch.cuda.empty_cache()
        passage.sort(); window.sort()
        dist[name] = (passage, window)
        rows.append({"pair": name, "anchor": anchor, "n_passages": len(passage),
                     "n_windows": len(window),
                     "passage_q10": round(passage[len(passage) // 10], 4),
                     "passage_median": round(st.median(passage), 4),
                     "window_q10": round(window[len(window) // 10], 4),
                     "window_median": round(st.median(window), 4),
                     "window_q10_over_passage_median": round(window[len(window) // 10] / st.median(passage), 4)})
        print(f"[cdf] {name}: {len(passage)} passages, {len(window)} windows of {a.window}; "
              f"passage median {st.median(passage):.3f}, window q10 {window[len(window)//10]:.3f}",
              flush=True)

    # where does the MEASURED onset sit in each pair's own surprisal distribution? If the
    # constant 0.89 is really about the anchor, the same fraction of protected objects should be
    # covered at the onset in every pair -- a statement of the law with nothing fitted in it.
    onsets = {}
    op = os.path.join(a.out, "onset.csv")
    if os.path.exists(op):
        for r in csv.DictReader(open(op)):
            if r["onset_est"]:
                onsets[(r["pair"], r["mode"])] = float(r["onset_est"])
    for r in rows:
        p_, w_ = dist[r["pair"]]
        o_s, o_o = onsets.get((r["pair"], "single")), onsets.get((r["pair"], "oracle"))
        r["onset_single"] = round(o_s, 4) if o_s else ""
        r["F_passage_at_onset"] = round(cdf_at(p_, o_s), 4) if o_s else ""
        r["onset_oracle"] = round(o_o, 4) if o_o else ""
        r["F_window_at_oracle_onset"] = round(cdf_at(w_, o_o), 4) if o_o else ""

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "surprisal_cdf.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # recall as a function of F(k), interpolated onto a common F grid
    curves = {}
    for name, comp, bp, _ in load_pairs(a.pairs_file):
        if name not in dist:
            continue
        for mode, L, which in (("single", 0, 0), ("oracle", a.window, 1)):
            c = curve(comp, mode, L)
            xs = [cdf_at(dist[name][which], k) for k in sorted(c)]
            if xs:   # a pair swept in one mode only (Phi-3.5 has no oracle arm)
                curves[(name, mode)] = (xs, [c[k] for k in sorted(c)])

    out = []
    for mode in ("single", "oracle"):
        series = {n: v for (n, m), v in curves.items() if m == mode}
        if len(series) < 2:
            continue
        for x in GRID:
            vals = {}
            for n, (xs, ys) in series.items():
                if x < xs[0] or x > xs[-1]:
                    vals[n] = None
                    continue
                i = max(1, bisect.bisect_left(xs, x))
                t = 0.0 if xs[i] == xs[i - 1] else (x - xs[i - 1]) / (xs[i] - xs[i - 1])
                vals[n] = ys[i - 1] + t * (ys[i] - ys[i - 1])
            v = [y for y in vals.values() if y is not None]
            if len(v) != len(series):
                continue
            out.append({"mode": mode, "F": x, "n_pairs": len(v),
                        **{f"recall_{n}": round(y, 5) for n, y in vals.items()},
                        "spread": round(max(v) - min(v), 5),
                        "sd": round(st.pstdev(v), 5)})

    if out:
        keys = list({k for r in out for k in r})
        order = [k for k in out[0] if k in keys] + [k for k in keys if k not in out[0]]
        with open(os.path.join(a.out, "cdf_collapse.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=order, restval="")
            w.writeheader()
            w.writerows(out)
        print("\nrecall against F(k) = fraction of protected objects the budget rate covers:")
        for mode in ("single", "oracle"):
            d = [r["spread"] for r in out if r["mode"] == mode]
            if d:
                print(f"  {mode:7s} mean spread over F in "
                      f"[{min(r['F'] for r in out if r['mode']==mode):g}, "
                      f"{max(r['F'] for r in out if r['mode']==mode):g}]: {st.mean(d):.4f} "
                      f"({len(d)} grid points)")
    fp = [r["F_passage_at_onset"] for r in rows if r["F_passage_at_onset"] != ""]
    fw = [r["F_window_at_oracle_onset"] for r in rows if r["F_window_at_oracle_onset"] != ""]
    if fp:
        print(f"\nfraction of protected objects the budget covers AT the measured onset:")
        for r in rows:
            print(f"  {r['pair'][:40]:42s} single F={r['F_passage_at_onset']}  "
                  f"oracle F={r['F_window_at_oracle_onset']}")
        print(f"  single: {min(fp):.3f}-{max(fp):.3f} (sd {st.pstdev(fp):.4f})"
              + (f"   oracle: {min(fw):.3f}-{max(fw):.3f} (sd {st.pstdev(fw):.4f})" if fw else ""))
    print(f"\nwrote {a.out}/surprisal_cdf.csv" + (" and cdf_collapse.csv" if out else ""))


if __name__ == "__main__":
    main()
