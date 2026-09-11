"""Plan v5 / feat-087 follow-up: is the oracle arm's utility a capacity or a judge artefact?

analysis/selection_decoding.py's oracle selects by the judge's own verdict and is then scored by
that same judge, so its 0.807 is an upper bound by construction. This selects among the same eight
candidates by judge A's verdict -- already computed, ties broken deterministically by the per-token
likelihood -- and scores the selected completion with judge B, which did no choosing. The n = 1
control is scored by judge B on the same prompts, so the gain is measured entirely inside judge B.

Bands committed in results/onset_prediction_selection.md before this ran: a gain of at least 0.244
means the capacity is real; under 0.10 means the oracle was largely circular.

Writes <out>/selection_crossjudge.csv.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/selection_crossjudge.py --out results
"""
import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_decoding import (VERDICT_U, boot_mean, kl_best_of_n,  # noqa: E402
                                         load_baseline, load_candidates)
from analysis.utility import judge_batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default="results/selection_candidates.csv")
    ap.add_argument("--gen-dir", default="output/phase5/sel_anchor8")
    ap.add_argument("--baseline-dir", default="output/sweep_plain")
    ap.add_argument("--judge-b", default="microsoft/Phi-3.5-mini-instruct")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--decoder-summary", default="results/judge_separation_v6_judge2.csv")
    ap.add_argument("--decoder-spend", default="results/utility_price.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = list(csv.DictReader(open(a.candidates)))
    by = {}
    for r in rows:
        by.setdefault(r["prompt_id"], []).append(r)
    for v in by.values():
        v.sort(key=lambda r: int(r["rank"]))
    cands, base = load_candidates(a.gen_dir), load_baseline(a.baseline_dir)
    pids = sorted(p for p in by if p in cands and p in base)
    print(f"[xj] {len(pids)} prompts", flush=True)

    # arm 1: the anchor alone (rank 0). arm 8: judge A's pick among the first n, ties by likelihood.
    picks = {}
    for p in pids:
        v = by[p][:a.n]
        picks[(p, 1)] = 0
        picks[(p, a.n)] = max(range(len(v)),
                              key=lambda i: (VERDICT_U[v[i]["outcome"]],
                                             float(v[i]["logp_per_token"])))

    tok = AutoTokenizer.from_pretrained(a.judge_b, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # no trust_remote_code: Phi-3.5-mini's repo carries custom modelling code that transformers
    # tries to FETCH even when the weights are cached, and HF_HUB_OFFLINE then fails the load.
    # The installed transformers has a native Phi-3 implementation, which is what the judge arms
    # already on record used.
    model = AutoModelForCausalLM.from_pretrained(a.judge_b,
                                                 torch_dtype=getattr(torch, a.dtype)).cuda().eval()
    keys = [(p, n) for p in pids for n in (1, a.n)]
    items, flips = [], []
    for p, n in keys:
        _, _, prompt, gen = cands[p][picks[(p, n)]]
        flip = rng.random() < 0.5
        flips.append(flip)
        items.append((prompt, base[p], gen) if flip else (prompt, gen, base[p]))
    verdicts = []
    for i in range(0, len(items), 200):
        verdicts += judge_batch(model, tok, items[i:i + 200], "cuda")
        print(f"[xj] judged {len(verdicts)}/{len(items)}", flush=True)

    us = {1: [], a.n: []}
    per = {}
    for (p, n), flip, v in zip(keys, flips, verdicts):
        won = (v == "B") if flip else (v == "A")
        u = 1.0 if won else 0.5 if v == "Tie" else 0.0
        us[n].append(u)
        per.setdefault(p, {})[n] = u
    # The two arms run on the SAME prompts, so the informative interval on the gain is paired:
    # bootstrapping the arms separately would widen it by the between-prompt variance the pairing
    # removes.
    order = sorted(per)
    diffs = [per[p][a.n] - per[p][1] for p in order]
    d_lo, d_hi = boot_mean(diffs, rng)
    with open(os.path.join(a.out, "selection_crossjudge_per_prompt.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["prompt_id", "u_n1", f"u_n{a.n}", "diff"])
        for p in order:
            w.writerow([p, per[p][1], per[p][a.n], round(per[p][a.n] - per[p][1], 4)])

    # How far each mechanism sits from Theorem 1's frontier, both measured under judge B's own
    # law of U, so the judge's absolute scale cancels in the ratio of the two multiples.
    from analysis.utility_price import rate
    w = sum(1 for x in us[1] if x == 1.0) / len(us[1])
    t = sum(1 for x in us[1] if x == 0.5) / len(us[1])
    d_b = (w, t, 1 - w - t)

    out = []
    for n in (1, a.n):
        lo, hi = boot_mean(us[n], rng)
        u_n = sum(us[n]) / len(us[n])
        lam = rate(d_b, u_n)
        out.append(dict(judge=a.judge_b, selector="judge A (Qwen2.5-7B-Instruct)" if n > 1 else "none",
                        n=n, kl_nats=round(kl_best_of_n(n), 4), n_prompts=len(us[n]),
                        u=round(u_n, 4), u_lo95=round(lo, 4),
                        u_hi95=round(hi, 4),
                        win_pct=round(100 * sum(1 for x in us[n] if x == 1.0) / len(us[n]), 1),
                        lambda_star=round(lam, 5),
                        nats_over_frontier=(round(kl_best_of_n(n) / lam, 1) if lam > 1e-9 else "")))
    gain = out[1]["u"] - out[0]["u"]
    assert abs(gain - sum(diffs) / len(diffs)) < 1e-9
    verdict = ("REAL: the capacity survives an independent judge" if gain >= 0.244 else
               "PARTIAL: real but inflated by the selecting judge" if gain >= 0.10 else
               "ARTEFACT: the oracle was largely circular" if gain >= 0 else
               "REVERSED: judge B disagrees about what selection improved")
    for r in out:
        print(f"  judge B, n = {r['n']:2d}  KL {r['kl_nats']:.3f}  u = {r['u']:.4f} "
              f"[{r['u_lo95']:.3f}, {r['u_hi95']:.3f}]  win {r['win_pct']:.1f}%")
    print(f"  gain = {gain:+.4f}  paired 95% CI [{d_lo:+.4f}, {d_hi:+.4f}]  ->  {verdict}")
    # the metered decoder's best arm under the SAME judge, from the separation CSV already on
    # record, priced the same way. Not re-run; this is a read.
    dec = [r for r in csv.DictReader(open(a.decoder_summary))
           if r["decoder"] == "KL" and float(r["k"]) > 0]
    best = max(dec, key=lambda r: float(r["utility"]))
    spend = {r["k"]: float(r["mean_spend_nats"])
             for r in csv.DictReader(open(a.decoder_spend))}[f"{float(best['k']):.1f}"]
    lam_d = rate(d_b, float(best["utility"]))
    out.append(dict(judge=a.judge_b, selector=f"metered decoder at k = {float(best['k']):g}",
                    n="", kl_nats=round(spend, 2), n_prompts=int(best["n_judged"]),
                    u=round(float(best["utility"]), 4), u_lo95="", u_hi95="",
                    win_pct="", lambda_star=round(lam_d, 5),
                    nats_over_frontier=round(spend / lam_d, 1)))
    print(f"  metered decoder, k = {float(best['k']):g}: {spend:.1f} nats, u = {best['utility']}, "
          f"{spend / lam_d:.1f}x the frontier against selection's "
          f"{kl_best_of_n(a.n) / rate(d_b, out[1]['u']):.1f}x")

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "selection_crossjudge.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]) + ["gain", "gain_lo95", "gain_hi95",
                                                          "verdict"])
        w.writeheader()
        for r in out:
            w.writerow({**r, "gain": round(gain, 4), "gain_lo95": round(d_lo, 4),
                        "gain_hi95": round(d_hi, 4), "verdict": verdict})
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
