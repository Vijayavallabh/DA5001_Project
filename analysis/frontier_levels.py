"""Order-averaged levels for Figure 1(b), every arm against ONE opponent on de-echoed text (feat-186).

Figure 1(b) used to plot the metered decoder's levels from one single-order pass
(judge_separation_v6_judge2.csv) beside selection's from another (selection_scaling.csv): two
passes on one axis, which caution (ap) forbids, on text that carried the prompt's tail (caution
(bc)). This judges every arm the panel plots in both presentation orders against the committed
opponent (output/sweep_plain k=-1, lowest seed), de-echoed, under judge B. A greedy judge shown both
orders is a deterministic function of the two texts, so an arm's level here does not depend on which
other arms are judged -- the check is that the four arms feat-184 Part A already judged reproduce
its per-prompt levels exactly. Descriptive; results/frontier_levels_note.md was committed first.

Usage (one card per call; --arms splits the work):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/frontier_levels.py --arms sel_n1,sel_n2,sel_n4 --tag a --out results
"""
import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.order_averaged_h2h import lowest_seed, paired_boot, true_prompts, u_of  # noqa: E402
from analysis.selection_decoding import kl_best_of_n, load_baseline, load_candidates  # noqa: E402
from analysis.selection_scaling import load_rewards  # noqa: E402
from analysis.utility import CLASSES, judge_batch, load_arm  # noqa: E402

METERED, ANCHOR = "output/phase2/conc_all", "output/sweep_plain"


def spend(k, pids_seeds=None):
    """Mean realised KL over the ordinary classes at budget k: over the judged trajectories when
    `pids_seeds` is given, else over every trajectory in the directory."""
    xs = []
    for c in CLASSES:
        for line in open(os.path.join(METERED, f"trajectories_k{k:g}_{c}.jsonl")):
            r = json.loads(line)
            m = r["metadata"]
            if pids_seeds is None or (m["prompt_id"], m["seed"]) in pids_seeds:
                xs.append(r["aggregate"]["total_spend"])
    return sum(xs) / len(xs), len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--judge", default="microsoft/Phi-3.5-mini-instruct")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=186)
    a = ap.parse_args()
    arms = a.arms.split(",")
    rng = random.Random(a.seed)

    opp = load_baseline(ANCHOR, deecho=True)
    cands = load_candidates("output/phase5/sel_anchor64", deecho=True)
    rewards = load_rewards("results/selection_rewards64.csv")
    prompts = true_prompts("data")
    texts, x = {}, {}
    for arm in arms:
        if arm.startswith("sel_n"):
            n = int(arm[5:])
            for p, r in rewards.items():
                r = r[:n]
                texts[(arm, p)] = cands[p][max(range(len(r)), key=lambda i: r[i])][3]
            x[arm] = ("kl_bound", kl_best_of_n(n), "")
        elif arm.startswith("met_k"):
            k = float(arm[5:])
            raw = load_arm(METERED, k, "kl", deecho=True)
            low = lowest_seed(raw)
            first = {}
            for (p, s) in raw:
                first[p] = min(s, first.get(p, s))
            for p, (_, g) in low.items():
                texts[(arm, p)] = g
            judged, _ = spend(k, set(first.items()))
            every, n_all = spend(k)
            x[arm] = ("mean_spend_judged", judged, f"{every:.4f} over all {n_all}")
        elif arm == "anchor_k0":
            for p, (_, g) in lowest_seed(load_arm(ANCHOR, 0.0, "kl", deecho=True)).items():
                texts[(arm, p)] = g
            x[arm] = ("anchor", 0.0, "")
        else:
            raise SystemExit(f"unknown arm {arm}")
    pids = sorted(set(opp) & set(prompts) & set.intersection(*({p for (m, p) in texts if m == arm}
                                                                for arm in arms)))
    assert len(pids) == 500, len(pids)
    print(f"[frontier] {len(pids)} prompts, arms {arms}", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.judge, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.judge, torch_dtype=torch.bfloat16).cuda().eval()

    U = {}
    for arm in arms:
        v = {}
        for tag, items in (("first", [(prompts[p], texts[(arm, p)], opp[p]) for p in pids]),
                           ("second", [(prompts[p], opp[p], texts[(arm, p)]) for p in pids])):
            got = []
            for i in range(0, len(items), 200):
                got += judge_batch(model, tok, items[i:i + 200], "cuda")
            v[tag] = got
            print(f"[frontier] {arm} arm-{tag} done", flush=True)
        for p, f, s in zip(pids, v["first"], v["second"]):
            U[(arm, p)] = 0.5 * (u_of(f, True) + u_of(s, False))

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"frontier_levels_per_prompt_{a.tag}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["prompt_id"] + [f"u_{arm}" for arm in arms])
        for p in pids:
            w.writerow([p] + [U[(arm, p)] for arm in arms])
    rows = []
    for arm in arms:
        us = [U[(arm, p)] for p in pids]
        lo, hi = paired_boot(us, rng)
        rows.append(dict(arm=arm, level=round(sum(us) / len(us), 4), lo95=round(lo, 4),
                         hi95=round(hi, 4), n=len(pids), x_kind=x[arm][0],
                         x_nats=round(x[arm][1], 4), note=x[arm][2]))
        print(f"[frontier] {arm:10s} level {rows[-1]['level']:.4f} [{lo:.4f}, {hi:.4f}]  "
              f"x={x[arm][1]:.4f} ({x[arm][0]})", flush=True)
    with open(os.path.join(a.out, f"frontier_levels_{a.tag}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
