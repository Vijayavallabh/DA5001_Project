"""feat-210: many arms against ONE opponent in one judge pass, both orders, every verdict saved.

results/onset_prediction_windowed_meter.md registers the arms, controls and differences. This script
judges each named arm against the fixed opponent exactly as analysis/order_averaged_h2h.py judges its
four (the corpus prompt, de-echoed text, batches of 200 in prompt order, both presentation orders, the
same judge_batch), so an arm judged here on the machine that judged a committed pass reproduces that
pass's levels (the registration's gate G2 checks it). Verdicts are cached per arm, so arms can be added
to a pass as their generations finish; a cached arm is only reused when it covers the same prompts.

Arm spec, --arm NAME=KIND:ARGS
  sel:N[:REWARDS]        the argmax of the pool's cached reward over its first N draws (N=1: rank 0);
                         REWARDS, if given, is that arm's own reward cache (feat-212: a second scorer)
  arm:DIR:TOKEN:CONSTR   h1.py-style trajectories_k<TOKEN>_<class>.jsonl, lowest seed
  base:RANK              the opponent's own configuration at its RANK-th lowest seed

--control ARM=CTRL  gain(ARM) = level(ARM) - level(CTRL), per prompt
--diff A,B          gain(A) - gain(B), paired over prompts (both must have a control)

Usage (the registered passes are scripts/run_feat210_judge.sh):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \\
    .venv/bin/python analysis/matched_h2h.py --tag matched_B --arm ... --control ... --diff ...
"""
import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.order_averaged_h2h import lowest_seed, paired_boot, true_prompts, u_of  # noqa: E402
from analysis.selection_decoding import load_baseline, load_candidates  # noqa: E402
from analysis.selection_scaling import load_rewards  # noqa: E402
from analysis.utility import judge_batch, load_arm  # noqa: E402


def label(lo, hi):
    return "UNRESOLVED" if lo <= 0 <= hi else ("CONFIRMED" if lo > 0 else "REFUTED")


def load_texts(spec, a):
    """NAME=KIND:ARGS -> (name, {prompt_id: text})."""
    name, _, rest = spec.partition("=")
    kind, _, args = rest.partition(":")
    if kind == "sel":
        n_s, _, rpath = args.partition(":")
        n = int(n_s)
        cands, rewards = load_candidates(a.sel_dir, deecho=True), load_rewards(rpath or a.rewards)
        out = {}
        for p, c in cands.items():
            if p not in rewards:
                continue
            r = rewards[p][:n]
            out[p] = c[max(range(len(r)), key=lambda i: r[i])][3]
        return name, out
    if kind == "arm":
        d, tok, constr = args.split(":")
        got = lowest_seed(load_arm(d, tok, constr, deecho=True))
        assert got, f"{spec}: no trajectories"
        return name, {p: g for p, (_, g) in got.items()}
    if kind == "base":
        return name, load_baseline(a.baseline_dir, deecho=True, rank=int(args))
    raise SystemExit(f"unknown arm kind in {spec!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--control", action="append", default=[])
    ap.add_argument("--diff", action="append", default=[])
    ap.add_argument("--sel-dir", default="output/phase5/sel_anchor64")
    ap.add_argument("--rewards", default="results/selection_rewards64.csv")
    ap.add_argument("--baseline-dir", default="output/sweep_plain")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--judge", default="microsoft/Phi-3.5-mini-instruct")
    ap.add_argument("--device-map", default="")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--judge-max-chars", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7717)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    opp = load_baseline(a.baseline_dir, deecho=True, rank=0)
    prompts = true_prompts(a.data_dir)
    arms = [load_texts(s, a) for s in a.arm]
    names = [n for n, _ in arms]
    assert len(set(names)) == len(names), names
    pids = sorted(set(opp) & set(prompts).intersection(*[set(t) for _, t in arms]))
    for n, t in arms:  # an arm that misses a prompt would silently re-scope every other arm (caution (ap))
        assert set(pids) == set(opp) & set(prompts), f"{n} covers {len(set(t) & set(opp))} of {len(opp)} prompts"
    print(f"[mh2h] {len(pids)} prompts, {len(arms)} arms, judge {a.judge}", flush=True)

    vpath = os.path.join(a.out, f"matched_h2h_verdicts_{a.tag}.csv")
    cache = {}
    if os.path.exists(vpath):
        for r in csv.DictReader(open(vpath)):
            cache.setdefault(r["arm"], {})[r["prompt_id"]] = (r["first"], r["second"], r["text_sha"])
    model = tok = None
    V = {}
    import hashlib
    for n, t in arms:
        sha = {p: hashlib.sha1(t[p].encode()).hexdigest()[:12] for p in pids}
        c = cache.get(n, {})
        if set(c) == set(pids) and all(c[p][2] == sha[p] for p in pids):
            V[n] = {p: c[p][:2] for p in pids}
            print(f"[mh2h] {n}: cached", flush=True)
            continue
        if model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(a.judge, padding_side="left")
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            kw = dict(torch_dtype=getattr(torch, a.dtype))
            model = (AutoModelForCausalLM.from_pretrained(a.judge, device_map=a.device_map, **kw) if a.device_map
                     else AutoModelForCausalLM.from_pretrained(a.judge, **kw).cuda()).eval()
            jdev = model.device if a.device_map else "cuda"
        fwd = [(prompts[p], t[p], opp[p]) for p in pids]
        rev = [(prompts[p], opp[p], t[p]) for p in pids]
        got = {}
        for tag_, items in (("first", fwd), ("second", rev)):
            g = []
            for i in range(0, len(items), 200):
                g += judge_batch(model, tok, items[i:i + 200], jdev, max_chars=a.judge_max_chars)
            got[tag_] = g
            print(f"[mh2h] {n} arm-{tag_} {len(g)}/{len(items)}", flush=True)
        V[n] = {p: (x, y) for p, x, y in zip(pids, got["first"], got["second"])}
        cache[n] = {p: (*V[n][p], sha[p]) for p in pids}
        with open(vpath, "w", newline="") as fh:     # rewrite after every arm: a crash loses one arm
            w = csv.writer(fh)
            w.writerow(["arm", "prompt_id", "first", "second", "text_sha"])
            for arm_, d in cache.items():
                for p, (x, y, s) in sorted(d.items()):
                    w.writerow([arm_, p, x, y, s])

    texts = dict(arms)
    U = {(n, p): 0.5 * (u_of(V[n][p][0], True) + u_of(V[n][p][1], False)) for n in names for p in pids}
    CONS = {(n, p): V[n][p] in (("A", "B"), ("B", "A"), ("Tie", "Tie")) for n in names for p in pids}
    ctrl = dict(c.split("=") for c in a.control)
    rng = random.Random(a.seed)
    rows = []
    for n in names:                                   # levels, in --arm order
        xs = [U[(n, p)] for p in pids]
        lo, hi = paired_boot(xs, rng)
        empt = [p for p in pids if not texts[n][p].strip()]
        rows.append(dict(quantity="level", arm=n, value=round(sum(xs) / len(xs), 4), lo95=round(lo, 4),
                         hi95=round(hi, 4), n=len(pids), reading="",
                         consistency=round(sum(CONS[(n, p)] for p in pids) / len(pids), 4),
                         n_empty=len(empt),
                         level_empty=round(sum(U[(n, p)] for p in empt) / len(empt), 4) if empt else ""))
    gain = {}
    for n, c in ctrl.items():                         # gains, in --control order
        gain[n] = {p: U[(n, p)] - U[(c, p)] for p in pids}
        xs = list(gain[n].values())
        lo, hi = paired_boot(xs, rng)
        rows.append(dict(quantity="gain", arm=f"{n} - {c}", value=round(sum(xs) / len(xs), 4),
                         lo95=round(lo, 4), hi95=round(hi, 4), n=len(pids), reading=label(lo, hi)))
    for d in a.diff:                                  # differences of gains, in --diff order
        x, y = d.split(",")
        for scope in ("all", "consistent pair", "consistent four"):
            keep = [p for p in pids if scope == "all"
                    or (scope == "consistent pair" and CONS[(x, p)] and CONS[(y, p)])
                    or (scope == "consistent four" and all(CONS[(z, p)] for z in (x, y, ctrl[x], ctrl[y])))]
            if len(keep) < 20:
                continue
            xs = [gain[x][p] - gain[y][p] for p in keep]
            lo, hi = paired_boot(xs, rng)
            rows.append(dict(quantity="difference" if scope == "all" else f"difference, {scope}",
                             arm=f"{x} - {y}", value=round(sum(xs) / len(xs), 4), lo95=round(lo, 4),
                             hi95=round(hi, 4), n=len(keep), reading=label(lo, hi)))
    os.makedirs(a.out, exist_ok=True)
    cols = ["quantity", "arm", "value", "lo95", "hi95", "n", "reading", "consistency", "n_empty", "level_empty"]
    with open(os.path.join(a.out, f"matched_h2h_{a.tag}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, f"matched_h2h_per_prompt_{a.tag}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["prompt_id"] + [f"u_{n}" for n in names])
        for p in pids:
            w.writerow([p] + [U[(n, p)] for n in names])
    for r in rows:
        print(f"  {r['quantity']:28s} {r['arm']:34s} {r['value']:+.4f} [{r['lo95']:+.4f}, {r['hi95']:+.4f}] "
              f"n={r['n']} {r['reading']}", flush=True)


if __name__ == "__main__":
    main()
