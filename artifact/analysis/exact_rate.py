"""Was Eq. (req) the wrong quantity, or the right one badly estimated?

`r(x) = s_s(x) - s_r(x)` is the NEAR-DETERMINISTIC COLLAPSE of what the decoding geometry actually
names. On the geometric path psi_t(1) = 0, so at full tilt the per-token charge along a target is

    kbar(x) = (1/T) sum_t D_KL(p_r,t || p_s,t),

a full-distribution mean. `r(x)` evaluates that integrand at the single realised token x_t, which
agrees with it only where p_r is concentrated on x_t. Every miss of the refuted refinement is low,
which is what a systematically biased estimator looks like -- so the refutation may be of the
estimator rather than of the quantity.

This measures both in ONE teacher-forced pass per passage, with the anchor and the memoriser loaded
together, so the comparison is like-for-like rather than against a stored column.

The nine onsets are already on record: this is a DIAGNOSTIC, not a prediction. Bands were committed
in results/onset_prediction_exact_rate.md before it ran, and that file says the same thing first.

Reads:  results/exact_rate_pairs.tsv  (name / anchor model / memoriser dir / budget_path csv)
Writes: <out>/exact_rate.csv, <out>/exact_rate_per_work.csv

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/exact_rate.py --out results
"""
import argparse
import csv
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.onset_theory import works  # noqa: E402

MANIFEST = "results/exact_rate_pairs.tsv"


def rates(lp_s, lp_r, target_ids):
    """(exact per-token D_KL(p_r||p_s), plug-in log p_r(x_t) - log p_s(x_t)) as lists.

    Correctness-critical and the whole point of the script: the exact rate integrates over the
    vocabulary under p_r, the plug-in reads one column. tests/test_exact_rate.py pins both against
    a hand-computed two-token case, and pins that they coincide when p_r is a point mass.
    """
    import torch
    v = min(lp_s.shape[-1], lp_r.shape[-1])
    if lp_s.shape[-1] != lp_r.shape[-1]:
        # padded embedding rows (Comma-7B carries 64256 for a 64000-token vocabulary). Renormalise
        # both over the common support rather than comparing densities on different supports.
        lp_s = torch.log_softmax(lp_s[:, :v], dim=-1)
        lp_r = torch.log_softmax(lp_r[:, :v], dim=-1)
    kl = (lp_r.exp() * (lp_r - lp_s)).sum(-1)
    idx = target_ids.clamp(max=v - 1).unsqueeze(1)
    plug = (lp_r.gather(1, idx) - lp_s.gather(1, idx)).squeeze(1)
    return kl.tolist(), plug.tolist()


def logprobs(model, ids, n_prefix):
    import torch
    with torch.no_grad():
        logits = model(ids).logits[0, :-1].float()      # row j predicts ids[j+1]
    return torch.log_softmax(logits[n_prefix - 1:], dim=-1)


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def perm_p(xs, ys, rho, iters=200000, seed=0):
    """Exact for n <= 8, sampled above; nine pairs is 362,880 permutations, so sample."""
    import itertools
    import random
    n = len(xs)
    if n <= 8:
        perms = list(itertools.permutations(ys))
        hits = sum(1 for p in perms if abs(spearman(xs, list(p))) >= abs(rho) - 1e-12)
        return hits / len(perms), "exact"
    rng = random.Random(seed)
    y = list(ys)
    hits = 0
    for _ in range(iters):
        rng.shuffle(y)
        if abs(spearman(xs, y)) >= abs(rho) - 1e-12:
            hits += 1
    return (hits + 1) / (iters + 1), f"sampled n={iters}"


def load_manifest(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) != 5:
            raise SystemExit(f"[exact-rate] {path}: want 5 fields, got {len(f)}: {line}")
        out.append((f[0], f[1], f[2], f[3], float(f[4])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs-file", default=MANIFEST)
    ap.add_argument("--data", default="data")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=100,
                    help="passages per pair; 100 matches every other onset measurement")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    summary, per_work = [], []
    for label, anchor_id, risky_dir, bpf, onset in load_manifest(a.pairs_file):
        if not os.path.exists(bpf):
            print(f"[exact-rate] missing {bpf}, skipping {label}", file=sys.stderr)
            continue
        keep = {r["prompt_id"] for r in csv.DictReader(open(bpf))}
        W = dict(sorted(works(a.data, keep).items())[:a.limit])
        if not W:
            print(f"[exact-rate] no passages for {label}", file=sys.stderr)
            continue
        tok = AutoTokenizer.from_pretrained(risky_dir)
        anchor = AutoModelForCausalLM.from_pretrained(
            anchor_id, dtype=getattr(torch, a.dtype)).to(dev).eval()
        risky = AutoModelForCausalLM.from_pretrained(
            risky_dir, dtype=getattr(torch, a.dtype)).to(dev).eval()

        kb, pl = [], []
        for pid, (pre, tgt) in sorted(W.items()):
            p_ids = tok(pre).input_ids if pre else [tok.bos_token_id or tok.eos_token_id]
            t_ids = tok(tgt, add_special_tokens=False).input_ids
            if not t_ids:
                continue
            ids = torch.tensor([p_ids + t_ids], device=dev)
            kl, plug = rates(logprobs(anchor, ids, len(p_ids)),
                             logprobs(risky, ids, len(p_ids)),
                             ids[0, len(p_ids):])
            k_bar, r_x = sum(kl) / len(kl), sum(plug) / len(plug)
            kb.append(k_bar)
            pl.append(r_x)
            per_work.append(dict(pair=label, prompt_id=pid, n_tokens=len(t_ids),
                                 kbar=round(k_bar, 5), plug_in=round(r_x, 5),
                                 kbar_gt_plug=int(k_bar > r_x)))
        del anchor, risky
        torch.cuda.empty_cache()

        k_med, p_med = st.median(kb), st.median(pl)
        summary.append(dict(
            pair=label, n=len(kb), measured_onset=onset,
            kbar_median=round(k_med, 4), plug_in_median=round(p_med, 4),
            kbar_q25=round(sorted(kb)[len(kb) // 4], 4),
            plug_in_q25=round(sorted(pl)[len(pl) // 4], 4),
            kbar_over_onset=round(k_med / onset, 4), plug_in_over_onset=round(p_med / onset, 4),
            kbar_rel_err=round(abs(k_med / onset - 1), 4),
            plug_in_rel_err=round(abs(p_med / onset - 1), 4),
            frac_kbar_gt_plug=round(sum(1 for x, y in zip(kb, pl) if x > y) / len(kb), 4),
            onset_quantile_of_kbar=round(sum(1 for x in kb if x <= onset) / len(kb), 4)))
        print(f"  {label[:40]:42s} kbar={k_med:.3f}  plug={p_med:.3f}  onset={onset:.3f}  "
              f"kbar/onset={k_med / onset:.3f}  plug/onset={p_med / onset:.3f}", flush=True)

    if not summary:
        return 1
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "exact_rate.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    with open(os.path.join(a.out, "exact_rate_per_work.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_work[0])); w.writeheader(); w.writerows(per_work)

    # ---- scoring against the committed bands ----
    e1 = sum(r["kbar_gt_plug"] for r in per_work) / len(per_work)
    e1_read = "BIASED LOW" if e1 >= 0.90 else "MIXED" if e1 >= 0.60 else "NOT THE EXPLANATION"
    k_err = st.median(r["kbar_rel_err"] for r in summary)
    p_err = st.median(r["plug_in_rel_err"] for r in summary)
    k_hi = sum(1 for r in summary if r["kbar_over_onset"] > 1)
    p_hi = sum(1 for r in summary if r["plug_in_over_onset"] > 1)
    n = len(summary)
    biased = max(k_hi, n - k_hi) >= max(7, n - 2)
    e2_read = ("RESCUED" if k_err < p_err and not biased else
               "BETTER BUT STILL BIASED" if k_err < p_err else "NO BETTER")
    xs = [r["kbar_median"] for r in summary]
    ys = [r["measured_onset"] for r in summary]
    rho = spearman(xs, ys)
    p, how = perm_p(xs, ys, rho)
    print(f"\n  E1 fraction kbar > plug-in      {e1:.4f}   {e1_read}")
    print(f"  E2 median |rate/onset - 1|      kbar {k_err:.4f} vs plug-in {p_err:.4f}; "
          f"misses high {k_hi}/{n} (plug-in {p_hi}/{n})   {e2_read}")
    print(f"  E3 Spearman(kbar, onset)        {rho:+.4f}  p = {p:.5f} ({how})")
    print(f"  E4 onset sits at quantile       " +
          ", ".join(f"{r['onset_quantile_of_kbar']:.2f}" for r in summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
