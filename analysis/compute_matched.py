"""feat-116: does a 0.5B scorer buy the certificate at the metered decoder's compute?

The paper concedes that selection anchoring is expensive to serve: 57.5x the metered decoder's
forward-pass FLOPs at n = 64, 7.2x at n = 8 (results/serving_cost.csv). That CSV also carries the
arithmetic that would change the answer -- the 7B reward model, not the 1.8B anchor, is what costs
-- in a row marked `0.5B, NOT RUN`. This runs it.

The scorer is Qwen2.5-0.5B-Instruct, 0.494B measured: same family, same chat template, same
log p("Yes") - log p("No") on the same fixed template as the 7B reward, so scale is the only thing
that varies. The certificate is unaffected and is not on trial -- q(y) <= n p_s(y) holds for ANY
score, so a 0.5B scorer buys the same log n nats a 70B one would. What is on trial is whether a
scorer cheap enough to make the mechanism affordable can still find the good draw.

Protocol is feat-113's corrected one, not the one behind results/selection_scaling.csv:
the ONE TRUE PROMPT per item from dap.shared.load_prompt_corpus (never served_prompt(), caution
(aa)), BOTH presentation orders averaged per item, one fixed opponent, judge B only. Arms nest by
seed order, so each distinct served completion is judged once per order and the arms are assembled
from those utilities.

Bands are results/onset_prediction_compute_matched.md, committed before this ran. This script
computes them; it does not choose them. F5 is a replication gate: if sel7b_n64 or metered_k10 miss
feat-113 by more than the +/-0.04 cross-pass floor, nothing else in the run may be quoted.

No generation: phase 1 re-scores cached candidates, phase 2 judges text already on disk.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/compute_matched.py --out results
"""
import argparse
import csv
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.order_averaged_h2h import (lowest_seed, paired_boot,  # noqa: E402
                                         true_prompts, u_of)
from analysis.selection_decoding import load_baseline, load_candidates  # noqa: E402
from analysis.selection_scaling import load_rewards, score_rewards  # noqa: E402
from analysis.serving_cost import P_ANCHOR, P_RISKY, P_SCORER  # noqa: E402
from analysis.utility import judge_batch, load_arm  # noqa: E402

GRID = (2, 4, 8, 16, 32, 64)
P_SMALL = 0.494          # Qwen2.5-0.5B-Instruct, counted off the loaded model, not the label
SEQ_TOKENS = 217         # serving_cost.py's L: median prompt 17 + median generation 200
FLOOR = 0.04             # the judge's own cross-pass floor, results/judge_consistency.csv


def cost_ratio(n, p_scorer):
    """Serving FLOPs relative to the metered decoder, in the model serving_cost.py commits to."""
    return n * (P_ANCHOR + p_scorer) * SEQ_TOKENS / ((P_ANCHOR + P_RISKY) * SEQ_TOKENS)


def reward_cache(path, model, cands, pids, max_n, dtype, batch_size, max_memory=None):
    """Score every cached candidate once and keep it; re-judging must not re-score 32,000 draws.

    max_memory is for a reward model that does not fit on one card (e.g. a 72B in bf16, ~145 GB).
    It defaults to None, which keeps the single-card .cuda() path every committed arm was run
    under, so nothing already measured changes (caution (q)).
    """
    if os.path.exists(path):
        return load_rewards(path)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if max_memory:
        rm = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=getattr(torch, dtype), device_map="auto",
            max_memory={int(k): v for k, v in
                        (x.split("=") for x in max_memory.split(","))}).eval()
    else:
        rm = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=getattr(torch, dtype)).cuda().eval()
    items, keys = [], []
    for p in pids:
        for j in range(max_n):
            _, cls, prompt, gen = cands[p][j]
            items.append((prompt, gen))
            keys.append((p, j, cls, len(gen.split())))
    print(f"[reward] {model} over {len(items)} candidates", flush=True)
    scores = score_rewards(rm, tok, items, rm.device, batch_size=batch_size)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["prompt_id", "rank", "prompt_class", "n_words", "reward"])
        for (p, j, cls, nw), s in zip(keys, scores):
            w.writerow([p, j, cls, nw, round(s, 5)])
    print(f"wrote {path}", flush=True)
    del rm
    torch.cuda.empty_cache()
    return load_rewards(path)


def judge_both_orders(model, tok, prompts, opp, items, label, batch=200):
    """items: (key, prompt_id, text). Returns key -> (order-averaged u, single-order u, consistent).

    The key is carried separately from the prompt id because a selection cell is keyed by
    (prompt, candidate rank) and a decoder arm by (arm, prompt); the prompt shown to the judge is
    always the corpus one, looked up by the id."""
    keys = [k for k, _, _ in items]
    fwd = [(prompts[i], x, opp[i]) for _, i, x in items]      # arm shown first
    rev = [(prompts[i], opp[i], x) for _, i, x in items]      # arm shown second
    got = {}
    for tag, batchitems in (("first", fwd), ("second", rev)):
        v = []
        for i in range(0, len(batchitems), batch):
            v += judge_batch(model, tok, batchitems[i:i + batch], "cuda")
            print(f"[judge] {label} arm-{tag} {len(v)}/{len(batchitems)}", flush=True)
        got[tag] = v
    out = {}
    for k, x, y in zip(keys, got["first"], got["second"]):
        cons = (x == "A" and y == "B") or (x == "B" and y == "A") or (x == "Tie" and y == "Tie")
        out[k] = (0.5 * (u_of(x, True) + u_of(y, False)), u_of(x, True), cons)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sel-dir", default="output/phase5/sel_anchor64")
    ap.add_argument("--rewards-7b", default="results/selection_rewards64.csv")
    ap.add_argument("--rewards-small", default="results/selection_rewards64_qwen05b.csv")
    ap.add_argument("--small-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--metered-dir", default="output/phase2/conc_all")
    ap.add_argument("--anchor-dir", default="output/sweep_plain")
    ap.add_argument("--baseline-dir", default="output/sweep_plain")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--judge", default="microsoft/Phi-3.5-mini-instruct")
    ap.add_argument("--k", type=float, default=10.0)
    ap.add_argument("--max-n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=9163)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N prompts. Smoke tests only: the bands assume 500, "
                         "and a --limit run must never write over the committed reward cache.")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ---- the shared prompt set -----------------------------------------------------------------
    opp = load_baseline(a.baseline_dir)
    cands = load_candidates(a.sel_dir)
    metered = lowest_seed(load_arm(a.metered_dir, a.k, "kl"))
    anchor = lowest_seed(load_arm(a.anchor_dir, 0.0, "kl"))
    prompts = true_prompts(a.data_dir)
    r7 = load_rewards(a.rewards_7b)
    pids = sorted(p for p in cands
                  if len(cands[p]) >= a.max_n and p in opp and p in r7
                  and p in metered and p in anchor and p in prompts)
    assert pids, "no prompt is present in every arm"
    if a.limit:
        pids = pids[:a.limit]
        assert "smoke" in a.rewards_small, "a --limit run must not write the committed cache"
        print(f"[cm] SMOKE: {len(pids)} prompts only, no band below applies", flush=True)
    print(f"[cm] {len(pids)} prompts shared by every arm", flush=True)

    # ---- phase 1: the 0.5B reward, cached ------------------------------------------------------
    rs = reward_cache(a.rewards_small, a.small_model, cands, pids, a.max_n,
                      a.dtype, a.batch_size)

    # ---- the served completion of every (scorer, n) cell ---------------------------------------
    scorers = {"05b": (rs, P_SMALL), "7b": (r7, P_SCORER)}
    picks = {(s, n, p): max(range(n), key=lambda j: rw[p][j])
             for s, (rw, _) in scorers.items() for n in GRID for p in pids}
    distinct = sorted({(p, 0) for p in pids} | {(p, picks[(s, n, p)])
                                                for s in scorers for n in GRID for p in pids})
    print(f"[cm] {len(distinct)} distinct served completions "
          f"({2 * len(GRID)} selection arms + the n=1 control)", flush=True)

    # ---- phase 2: judge, both orders, one true prompt, one fixed opponent -----------------------
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

    # ---- assemble the arms: gains over each mechanism's own control -----------------------------
    g_met = [U[("metered", p)][0] - U[("anchor", p)][0] for p in pids]
    arms = {}
    for s, (_, pf) in scorers.items():
        for n in GRID:
            arms[f"sel{s}_n{n}"] = (
                [U[(p, picks[(s, n, p)])][0] - U[(p, 0)][0] for p in pids], cost_ratio(n, pf), n, s)
    arms["metered_k10"] = (g_met, 1.0, "", "--")

    rows = []
    for name, (g, ratio, n, s) in arms.items():
        lo, hi = paired_boot(g, rng)
        rows.append(dict(arm=name, scorer={"05b": "Qwen2.5-0.5B", "7b": "Qwen2.5-7B"}.get(s, "--"),
                         n=n, cost_vs_metered=round(ratio, 3),
                         nats_certified=round(math.log(n), 4) if n else round(a.k * 200, 1),
                         gain=round(sum(g) / len(g), 4), lo95=round(lo, 4), hi95=round(hi, 4),
                         n_prompts=len(pids), separates=not (lo <= 0 <= hi)))
    rows.sort(key=lambda r: (r["cost_vs_metered"], r["arm"]))

    # ---- the bands -----------------------------------------------------------------------------
    by = {r["arm"]: r for r in rows}
    g_of = {k: v[0] for k, v in arms.items()}
    bands = []

    f1 = by["sel05b_n64"]
    bands.append(("F1 0.5B scorer carries a gain at n=64", "sel05b_n64", f1["gain"],
                  f1["lo95"], f1["hi95"], "WORKS" if f1["separates"] else "FAILS"))

    d2 = [x - y for x, y in zip(g_of["sel05b_n64"], g_of["sel7b_n64"])]
    m2, (l2, h2) = sum(d2) / len(d2), paired_boot(d2, rng)
    bands.append(("F2 cost of shrinking the scorer 14x", "sel05b_n64 - sel7b_n64", round(m2, 4),
                  round(l2, 4), round(h2, 4),
                  "CHEAP" if l2 <= 0 <= h2 else "COSTLY" if m2 < 0 else "BETTER"))

    met = by["metered_k10"]["gain"]
    cross = [r for r in rows if r["scorer"] == "Qwen2.5-0.5B"
             and r["separates"] and r["gain"] >= met]
    if cross:
        c = min(cross, key=lambda r: r["cost_vs_metered"])
        ratio = c["cost_vs_metered"]
        read = ("BELOW THE METER" if ratio <= 1.0 else "WITHIN 4x" if ratio <= 4.0
                else "WITHIN 15x" if ratio <= 15.0 else "NO CROSSING")
        bands.append(("F3 crossover serving cost", c["arm"], ratio, "", "", read))
    else:
        bands.append(("F3 crossover serving cost", "none on the grid", "", "", "", "NO CROSSING"))

    d4 = [x - y for x, y in zip(g_of["sel05b_n4"], g_met)]
    m4, (l4, h4) = sum(d4) / len(d4), paired_boot(d4, rng)
    bands.append((f"F4 compute-matched head-to-head "
                   f"({next(r['cost_vs_metered'] for r in rows if r['arm'] == 'sel05b_n4'):.2f}x)",
                   "sel05b_n4 - metered_k10",
                  round(m4, 4), round(l4, 4), round(h4, 4),
                  "MATCHED-COMPUTE PARITY" if l4 <= 0 <= h4 else
                  "MATCHED-COMPUTE WIN" if m4 > 0 else "MATCHED-COMPUTE LOSS"))

    ref = {"sel7b_n64": 0.1045, "metered_k10": 0.0400}
    miss = {k: round(by[k]["gain"] - v, 4) for k, v in ref.items()
            if abs(by[k]["gain"] - v) > FLOOR}
    bands.append(("F5 replication of feat-113 within +/-0.04",
                  f"sel7b_n64 {by['sel7b_n64']['gain']:+.4f} vs {ref['sel7b_n64']:+.4f}; "
                  f"metered_k10 {by['metered_k10']['gain']:+.4f} vs {ref['metered_k10']:+.4f}",
                  "", "", "", "REPLICATES" if not miss else f"FAILED {miss}"))

    # ---- write ---------------------------------------------------------------------------------
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "compute_matched.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(a.out, "compute_matched_bands.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["band", "quantity", "value", "lo95", "hi95", "reading"])
        w.writerows(bands)
    with open(os.path.join(a.out, "compute_matched_per_prompt.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        names = sorted(arms)
        w.writerow(["prompt_id"] + [f"gain_{x}" for x in names])
        for i, p in enumerate(pids):
            w.writerow([p] + [round(arms[x][0][i], 4) for x in names])

    print()
    for r in rows:
        print(f"  {r['arm']:16s} {r['scorer']:13s} cost {r['cost_vs_metered']:6.2f}x  "
              f"gain {r['gain']:+.4f} [{r['lo95']:+.4f}, {r['hi95']:+.4f}]  "
              f"{'SEPARATES' if r['separates'] else '-'}")
    print()
    for b in bands:
        print(f"  {b[0]:42s} {b[2]} [{b[3]}, {b[4]}]  {b[5]}")
    cons = sum(U[k][2] for k in U) / len(U)
    print(f"\n  order consistency over every judged item: {cons:.4f}")
    print(f"wrote {os.path.join(a.out, 'compute_matched.csv')}")


if __name__ == "__main__":
    main()
