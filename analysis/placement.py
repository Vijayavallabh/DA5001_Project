"""Budget PLACEMENT at a fixed sequence budget: uniform per step, front-loaded, or on the draw.

Proposition 5 says an affordable causal policy must be the anchor at all but O(1) steps. That has
two horns and the paper only ever demonstrated one. The deployed bucket spreads its budget by
construction and lands on the vacuous horn (Proposition 3). But the proposition does not forbid a
causal policy from CONCENTRATING a bounded budget on the opening -- and until that arm exists,
"the budget has to leave the decode loop" is stronger than the evidence.

Three placements of the same log 8 = 2.0794 nats, on the same 500 ordinary prompts, judged by the
same judge B against the same unconstrained baseline:

  uniform      k = 2.0794 / 200 per token, no initial bank          (the deployed rule, scaled down)
  front-loaded the whole budget granted up front, negligible refill (the causal policy Prop 5 allows)
  draw         best of n = 8 safe-model samples                     (already on record, log n = 2.08)

Bands were committed in results/onset_prediction_placement.md before any arm was generated.

This script does no generation: it judges completions h1.py has already written. Every arm is
scored in one model load so the judge is identical across arms, and the anchor-alone control is
re-judged in the same pass rather than read from an earlier run, so every gain is paired.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/placement.py --out results
"""
import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_decoding import boot_mean, load_baseline, load_candidates  # noqa: E402
from analysis.utility import judge_batch  # noqa: E402


def load_arm(gen_dir):
    """prompt_id -> generation, for a single-trajectory h1.py run. Reuses the loader the selection
    arms use so the prompt-id convention cannot drift between mechanisms."""
    cands = load_candidates(gen_dir)
    return {p: v[0][3] for p, v in cands.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uniform-dir", default="output/phase5/place_unif_2p08")
    ap.add_argument("--front-dir", default="output/phase5/place_front_2p08")
    ap.add_argument("--front-big-dir", default="output/phase5/place_front_20")
    ap.add_argument("--anchor-dir", default="output/phase5/sel_anchor8",
                    help="rank 0 is the anchor-alone control, the same one the selection arms use")
    ap.add_argument("--baseline-dir", default="output/sweep_plain")
    ap.add_argument("--judge-b", default="microsoft/Phi-3.5-mini-instruct")
    ap.add_argument("--budget", type=float, default=2.0794, help="log 8, the matched budget")
    ap.add_argument("--big-budget", type=float, default=20.0)
    ap.add_argument("--selection-gain", type=float, default=0.081,
                    help="the draw placement's gain at the same budget, from selection_crossjudge")
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = load_baseline(a.baseline_dir)
    arms = {"anchor alone (control)": {p: v[0][3] for p, v in load_candidates(a.anchor_dir).items()}}
    for name, d, K in (("uniform, per step", a.uniform_dir, a.budget),
                       ("front-loaded", a.front_dir, a.budget),
                       ("front-loaded, large", a.front_big_dir, a.big_budget)):
        if os.path.isdir(d):
            arms[name] = load_arm(d)
        else:
            print(f"[place] missing {d}, skipping {name}", file=sys.stderr)
    pids = sorted(set.intersection(*[set(v) for v in arms.values()]) & set(base))
    print(f"[place] {len(pids)} prompts on {len(arms)} arms", flush=True)

    tok = AutoTokenizer.from_pretrained(a.judge_b, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.judge_b,
                                                 torch_dtype=getattr(torch, a.dtype)).cuda().eval()

    prompts = {p: load_candidates(a.anchor_dir)[p][0][2] for p in pids}
    u = {}
    for name, gens in arms.items():
        items, flips = [], []
        for p in pids:
            flip = rng.random() < 0.5
            flips.append(flip)
            items.append((prompts[p], base[p], gens[p]) if flip else (prompts[p], gens[p], base[p]))
        verdicts = []
        for i in range(0, len(items), 200):
            verdicts += judge_batch(model, tok, items[i:i + 200], "cuda")
            print(f"[place] {name}: judged {len(verdicts)}/{len(items)}", flush=True)
        u[name] = {}
        for p, flip, v in zip(pids, flips, verdicts):
            won = (v == "B") if flip else (v == "A")
            u[name][p] = 1.0 if won else 0.5 if v == "Tie" else 0.0

    ctrl = "anchor alone (control)"
    budgets = {ctrl: 0.0, "uniform, per step": a.budget, "front-loaded": a.budget,
               "front-loaded, large": a.big_budget}
    rows, gains = [], {}
    for name in arms:
        vals = [u[name][p] for p in pids]
        lo, hi = boot_mean(vals, rng)
        rec = dict(placement=name, budget_nats=round(budgets[name], 4), n_prompts=len(pids),
                   u=round(sum(vals) / len(vals), 4), u_lo95=round(lo, 4), u_hi95=round(hi, 4),
                   win_pct=round(100 * sum(1 for x in vals if x == 1.0) / len(vals), 1))
        if name != ctrl:
            d = [u[name][p] - u[ctrl][p] for p in pids]
            g_lo, g_hi = boot_mean(d, rng)
            gains[name] = (sum(d) / len(d), g_lo, g_hi)
            rec.update(gain=round(gains[name][0], 4), gain_lo95=round(g_lo, 4),
                       gain_hi95=round(g_hi, 4))
        else:
            rec.update(gain="", gain_lo95="", gain_hi95="")
        rows.append(rec)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "placement.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(a.out, "placement_per_prompt.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["prompt_id"] + [n for n in arms])
        for p in pids:
            w.writerow([p] + [u[n][p] for n in arms])

    # ---- scoring against the bands committed before any arm was generated ----
    def band(name):
        return gains.get(name)
    unif, front, big = band("uniform, per step"), band("front-loaded"), band("front-loaded, large")
    print()
    for r in rows:
        g = f"  gain {r['gain']:+.4f} [{r['gain_lo95']:+.4f}, {r['gain_hi95']:+.4f}]" \
            if r["gain"] != "" else ""
        print(f"  {r['placement']:24s} K={r['budget_nats']:>7.4f}  u={r['u']:.4f} "
              f"[{r['u_lo95']:.3f}, {r['u_hi95']:.3f}]{g}")
    if unif and front:
        overlap = not (front[2] < unif[1] or unif[2] < front[1])
        diff = abs(front[0] - unif[0])
        p1 = ("PLACEMENT MATTERS" if not overlap else
              "PLACEMENT IS SECOND ORDER" if diff > 0.03 else "PLACEMENT IS IRRELEVANT")
        print(f"\n  P1 front vs uniform at K={a.budget}: diff {diff:.4f}, "
              f"intervals {'overlap' if overlap else 'disjoint'}  ->  {p1}")
    if front:
        p2 = ("THE CAUSAL HORN IS EMPTY" if front[1] <= 0 <= front[2] else
              "CAUSAL CONCENTRATION WORKS" if abs(front[0] - a.selection_gain) <= 0.03 else
              "CAUSAL CONCENTRATION PARTLY WORKS" if front[0] < a.selection_gain - 0.03 else
              "CAUSAL CONCENTRATION BEATS SELECTION")
        print(f"  P2 front-loaded vs control: {front[0]:+.4f} "
              f"[{front[1]:+.4f}, {front[2]:+.4f}] against selection's "
              f"{a.selection_gain:+.4f}  ->  {p2}")
    if front and big:
        p3 = "SCALES" if big[1] > front[0] else "FLAT"
        print(f"  P3 K={a.big_budget} vs K={a.budget}: {big[0]:+.4f} "
              f"[{big[1]:+.4f}, {big[2]:+.4f}]  ->  {p3}")
    print("\n  P4 leakage is scored separately on the protected split by "
          "analysis/composition_attack.py; every arm reports it with its k=-1 and k=0 baselines.")


if __name__ == "__main__":
    main()
