"""feat-117: where does best-of-n stop turning over? The scorer-scale boundary.

feat-116 found, without registering it, that sweeping n with a 0.5B reward raises judged gain to a
peak at n=16 and then LOWERS it, while a 7.6B reward rises monotonically across the same grid. The
reading we offered is that the argmax of a weak score over a larger pool selects increasingly on the
score's noise, so log n is not a free knob: the certificate keeps improving in n while the utility
bought with it turns over. That rests on one scorer at one scale and was never given an interval.

This arm puts two scales between them -- Qwen2.5-1.5B-Instruct (1.5437B measured) and
Qwen2.5-3B-Instruct (3.0859B measured), same family, same chat template, same
log p("Yes") - log p("No") reward on the same fixed template -- and judges all four scorers in ONE
pass. One pass is not a convenience: this paper's own instrument checks forbid quoting judged levels
across passes, and the comparison here is between scorers, so they must share a control, an
opponent, a prompt set and a judging session.

The estimand is the terminal drop D_s = g_s(64) - g_s(16), paired per prompt. n=16 comes from
feat-116, a prior experiment, so for the two new scorers this is out-of-sample and not a cell picked
after looking.

Bands G0-G4 are results/onset_prediction_scorer_scale.md, committed and committed to git before this
ran. G0 is a replication gate: if it fails, nothing else may be quoted. This script computes the
bands; it does not choose them.

No generation. Phase 1 re-scores cached candidates, phase 2 judges text already on disk.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/scorer_scale.py --out results
"""
import argparse
import csv
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.compute_matched import cost_ratio, judge_both_orders, reward_cache  # noqa: E402
from analysis.order_averaged_h2h import lowest_seed, paired_boot, true_prompts  # noqa: E402
from analysis.scorer_agreement import spearman  # noqa: E402
from analysis.selection_decoding import load_baseline, load_candidates  # noqa: E402
from analysis.utility import load_arm  # noqa: E402

GRID = (2, 4, 8, 16, 32, 64)
DROP_FROM = 16              # fixed by feat-116, before this arm; see the pre-registration
FLOOR = 0.04                # the judge's own cross-pass floor, results/judge_consistency.csv

# tag, checkpoint, parameters in billions COUNTED OFF THE LOADED MODEL, reward-cache suffix.
# The 7B suffix is empty because results/selection_rewards64.csv is feat-088's cache and this arm
# reuses it rather than re-scoring 32,000 candidates with a model that already scored them.
SCORERS = [
    ("05b", "Qwen/Qwen2.5-0.5B-Instruct", 0.4940, "_qwen05b"),
    ("15b", "Qwen/Qwen2.5-1.5B-Instruct", 1.5437, "_qwen15b"),
    ("3b",  "Qwen/Qwen2.5-3B-Instruct",   3.0859, "_qwen3b"),
    ("7b",  "Qwen/Qwen2.5-7B-Instruct",   7.6156, ""),
    # feat-161: the missing rung. The judge-free ladder shows no saturation to 72B while this one
    # reads saturation at 7.6B; all five are judged in ONE pass so the comparison is within-pass.
    ("14b", "Qwen/Qwen2.5-14B-Instruct", 14.7701, "_qwen14b_judged"),
    ("72b", "Qwen/Qwen2.5-72B-Instruct", 72.7062, "_qwen72b_judged"),
]
# feat-116's numbers, which G0 requires this pass to reproduce before anything else is read
REF = {"sel7b_n64": 0.1075, "sel05b_n64": 0.0220, "metered_k10": 0.0400}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sel-dir", default="output/phase5/sel_anchor64")
    ap.add_argument("--metered-dir", default="output/phase2/conc_all")
    ap.add_argument("--anchor-dir", default="output/sweep_plain")
    ap.add_argument("--baseline-dir", default="output/sweep_plain")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--judge", default="microsoft/Phi-3.5-mini-instruct")
    ap.add_argument("--k", type=float, default=10.0)
    ap.add_argument("--max-n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=11703)
    ap.add_argument("--reward-max-memory", default="",
                    help="e.g. '0=75GiB,1=75GiB'; used for the 72B rung only. Empty keeps the "
                         "single-card path every committed arm was run under (caution (q)).")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cache-prefix", default="results/selection_rewards64",
                    help="reward caches are <prefix><suffix>.csv; point it elsewhere for a smoke "
                         "run so the committed caches are never written from a partial prompt set")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke tests only; the bands assume 500 and a --limit run must not write "
                         "a committed reward cache")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    opp = load_baseline(a.baseline_dir)
    cands = load_candidates(a.sel_dir)
    metered = lowest_seed(load_arm(a.metered_dir, a.k, "kl"))
    anchor = lowest_seed(load_arm(a.anchor_dir, 0.0, "kl"))
    prompts = true_prompts(a.data_dir)
    pids = sorted(p for p in cands if len(cands[p]) >= a.max_n and p in opp
                  and p in metered and p in anchor and p in prompts)
    assert pids, "no prompt is present in every arm"
    caches = {t_: f"{a.cache_prefix}{sfx}.csv" for t_, _, _, sfx in SCORERS}
    if a.limit:
        pids = pids[:a.limit]
        assert "results/" not in a.cache_prefix, \
            "a --limit run must point --cache-prefix away from results/"
        print(f"[ss] SMOKE: {len(pids)} prompts, no band applies", flush=True)
    print(f"[ss] {len(pids)} prompts x {a.max_n} candidates x {len(SCORERS)} scorers", flush=True)

    # ---- phase 1: one reward cache per scorer, scored once and kept --------------------------
    rewards = {}
    for tag, model, _, _ in SCORERS:
        # only the 72B rung needs sharding; every other scorer keeps the single-card path
        mm = a.reward_max_memory if tag == "72b" else None
        rewards[tag] = reward_cache(caches[tag], model, cands, pids, a.max_n, a.dtype,
                                    a.batch_size, max_memory=mm)
        assert all(len(rewards[tag][p]) >= a.max_n for p in pids), f"{tag} cache is short"

    picks = {(t, n, p): max(range(n), key=lambda j: rewards[t][p][j])
             for t, _, _, _ in SCORERS for n in GRID for p in pids}
    distinct = sorted({(p, 0) for p in pids}
                      | {(p, picks[(t, n, p)]) for t, _, _, _ in SCORERS for n in GRID for p in pids})
    print(f"[ss] {len(distinct)} distinct served completions across "
          f"{len(SCORERS) * len(GRID)} selection arms", flush=True)

    # ---- phase 2: one judging pass over everything -------------------------------------------
    tok = AutoTokenizer.from_pretrained(a.judge, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    jm = AutoModelForCausalLM.from_pretrained(
        a.judge, torch_dtype=getattr(torch, a.dtype)).cuda().eval()
    U = judge_both_orders(jm, tok, prompts, opp,
                          [((p, j), p, cands[p][j][3]) for p, j in distinct], "selection")
    U |= judge_both_orders(jm, tok, prompts, opp,
                           [(("metered", p), p, metered[p][1]) for p in pids], "metered_k10")
    U |= judge_both_orders(jm, tok, prompts, opp,
                           [(("anchor", p), p, anchor[p][1]) for p in pids], "anchor_k0")
    del jm
    torch.cuda.empty_cache()

    # ---- the arms ------------------------------------------------------------------------------
    g = {f"sel{t}_n{n}": [U[(p, picks[(t, n, p)])][0] - U[(p, 0)][0] for p in pids]
         for t, _, _, _ in SCORERS for n in GRID}
    g["metered_k10"] = [U[("metered", p)][0] - U[("anchor", p)][0] for p in pids]
    params = {t: pf for t, _, pf, _ in SCORERS}

    rows = []
    for name, vals in g.items():
        lo, hi = paired_boot(vals, rng)
        t = name[3:name.index("_n")] if name.startswith("sel") else ""
        n = int(name.split("_n")[1]) if name.startswith("sel") else ""
        rows.append(dict(arm=name, scorer_b=params.get(t, ""), n=n,
                         cost_vs_metered=round(cost_ratio(n, params[t]), 3) if t else 1.0,
                         nats_certified=round(math.log(n), 4) if n else round(a.k * 200, 1),
                         gain=round(sum(vals) / len(vals), 4), lo95=round(lo, 4),
                         hi95=round(hi, 4), n_prompts=len(pids), separates=not (lo <= 0 <= hi)))
    rows.sort(key=lambda r: (r["scorer_b"] if r["scorer_b"] != "" else 99, r["n"] or 0))
    by = {r["arm"]: r for r in rows}

    # ---- the bands -----------------------------------------------------------------------------
    bands = []
    miss = {k: round(by[k]["gain"] - v, 4) for k, v in REF.items()
            if abs(by[k]["gain"] - v) > FLOOR}
    g0 = "REPLICATES" if not miss else f"FAILED {miss}"
    bands.append(dict(band="G0 replication of feat-116 within +/-0.04",
                      quantity="; ".join(f"{k} {by[k]['gain']:+.4f} vs {v:+.4f}"
                                         for k, v in REF.items()),
                      value="", lo95="", hi95="", reading=g0))

    readings = {}
    for t, _, pf, _ in SCORERS:
        d = [x - y for x, y in zip(g[f"sel{t}_n64"], g[f"sel{t}_n{DROP_FROM}"])]
        m, (lo, hi) = sum(d) / len(d), paired_boot(d, rng)
        r = "FLAT" if lo <= 0 <= hi else "TURNS OVER" if m < 0 else "RISES"
        readings[t] = r
        bands.append(dict(band=f"G1 terminal drop, {pf}B scorer",
                          quantity=f"sel{t}_n64 - sel{t}_n{DROP_FROM}", value=round(m, 4),
                          lo95=round(lo, 4), hi95=round(hi, 4), reading=r))

    for t, _, pf, _ in SCORERS:
        vals = [by[f"sel{t}_n{n}"]["gain"] for n in GRID]
        rho = spearman([math.log(n) for n in GRID], vals)
        peak = max(range(len(GRID)), key=lambda i: vals[i])
        bands.append(dict(band=f"G2 shape, {pf}B scorer", value=round(rho, 4),
                          quantity=f"Spearman(gain, log n); peak at n={GRID[peak]}, "
                                   f"peak-relative drop {vals[-1] - vals[peak]:+.4f} (secondary)",
                          lo95="", hi95="",
                          reading="MONOTONE" if rho == 1.0 else "NOT MONOTONE"))

    safe = [(pf, t) for t, _, pf, _ in SCORERS if readings[t] != "TURNS OVER"]
    if not any(r == "TURNS OVER" for r in readings.values()):
        g3 = "NO TURNOVER FOUND"
        q = "no scorer reads TURNS OVER under its interval"
    elif safe:
        over = [f"{pf}B" for t, _, pf, _ in SCORERS if readings[t] == "TURNS OVER"]
        g3 = f"BOUNDARY AT {min(safe)[0]}B"
        q = (f"turns over at {', '.join(over)}; smallest scorer that does not: {min(safe)[0]}B")
    else:
        g3 = "BOUNDARY ABOVE 7.6B"
        q = "every scorer tested turns over"
    bands.append(dict(band="G3 the scorer-scale boundary", quantity=q, value="", lo95="", hi95="",
                      reading=g3))

    tags = [t for t, _, _, _ in SCORERS]
    g4 = "; ".join(f"{params[t]}B {by[f'sel{t}_n64']['gain']:+.4f}" for t in tags)
    for x, y in zip(tags, tags[1:]):
        d = [p - q_ for p, q_ in zip(g[f"sel{y}_n64"], g[f"sel{x}_n64"])]
        lo, hi = paired_boot(d, rng)
        bands.append(dict(band=f"G4 terminal gain, {params[y]}B over {params[x]}B",
                          quantity=f"sel{y}_n64 - sel{x}_n64", value=round(sum(d) / len(d), 4),
                          lo95=round(lo, 4), hi95=round(hi, 4),
                          reading="SEPARATES" if not (lo <= 0 <= hi) else "-"))
    bands.append(dict(band="G4 terminal gain by scale (descriptive)", quantity=g4, value="",
                      lo95="", hi95="", reading="four points carry no law and none is claimed"))

    # ---- write ---------------------------------------------------------------------------------
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "scorer_scale.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(a.out, "scorer_scale_bands.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(bands[0])); w.writeheader(); w.writerows(bands)
    with open(os.path.join(a.out, "scorer_scale_per_prompt.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        names = sorted(g)
        w.writerow(["prompt_id"] + [f"gain_{x}" for x in names])
        for i, p in enumerate(pids):
            w.writerow([p] + [round(g[x][i], 4) for x in names])

    print()
    for r in rows:
        print(f"  {r['arm']:14s} {str(r['scorer_b']):>7}B  cost {r['cost_vs_metered']:>7}x  "
              f"gain {r['gain']:+.4f} [{r['lo95']:+.4f}, {r['hi95']:+.4f}]"
              f"  {'SEP' if r['separates'] else '-'}")
    print()
    for b in bands:
        print(f"  {b['band']:38s} {str(b['value']):>8} [{b['lo95']}, {b['hi95']}]  {b['reading']}")
    print(f"wrote {os.path.join(a.out, 'scorer_scale.csv')}")


if __name__ == "__main__":
    main()
