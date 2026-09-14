"""The head-to-head at one (anchor, risky) pair: metered nats against selection nats.

The paper's central comparison -- the metered decoder gains +0.072 for 171.3 nats and selection
gains +0.142 for 3.175 -- is measured at one pair, and it cannot simply be repeated at the anchors
the breadth arm added: the factory fuses two distributions over ONE shared vocabulary, and among the
openly licensed safe models we hold only TinyComma-1.8B ships the Llama-3 tokenizer, so at every
other legitimate anchor the metered decoder does not run and only selection has a number. The one
further pair that exists is Llama-3.2-1B against Llama-3.1-8B-Instruct.

Llama-3.2-1B is NOT a legitimate safe model: it is trained on undisclosed data and may have read the
protected works. No certificate, leakage or s(x) number may be taken from this arm, and it generates
nothing on the protected split. What it does is hold the mechanism, the workload, the budget grid and
the judges fixed while the pair changes.

Both mechanisms are judged in one model load against the SAME opponent -- the k=-1 unconstrained
completions already in the metered directory -- so nothing but the arm differs. The gain is a paired
difference against a single shared anchor-alone control, the metered run's own k=0 arm, and
selection's within-pool gain against its own n=1 is reported beside it because the two answer
slightly different questions (see the scoring log).

Bands committed in results/onset_prediction_frontier_second_pair.md before the selection arm was
generated and before anything was judged.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/frontier_pair.py --sel-dir output/phase5/sel_llama321b_8 \
      --metered-dir output/phase5/imit_llama321b --tag _llama321b --out results
"""
import argparse
import csv
import glob
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_decoding import boot_mean, load_baseline, load_candidates  # noqa: E402
from analysis.selection_scaling import REWARD_TMPL, score_rewards, yes_no_ids  # noqa: E402
from analysis.utility import judge_batch  # noqa: E402

CLASSES = ("neutral", "creative", "factual")


def arm_k(gen_dir, k):
    """The k token h1.py wrote into the filenames for this budget. It is the CLI string, not a
    normalised float (AGENTS caution (o)), so it is read off the directory rather than formatted."""
    for cand in (k, f"{float(k):g}", f"{float(k)}"):
        if glob.glob(os.path.join(gen_dir, f"trajectories_k{cand}_neutral.jsonl")):
            return str(cand)
    raise SystemExit(f"[frontier] no k={k} arm in {gen_dir}")


def metered_arm(gen_dir, k):
    """prompt_id -> (generation, realised spend in nats) for one budget of an h1.py sweep."""
    out = {}
    for cls in CLASSES:
        p = os.path.join(gen_dir, f"trajectories_k{arm_k(gen_dir, k)}_{cls}.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            a, m = r["aggregate"], r["metadata"]
            key = m["prompt_id"]
            cand = (m["seed"], a.get("generation") or "", float(a.get("total_spend") or 0.0))
            if key not in out or cand < out[key]:
                out[key] = cand
    return {p: (v[1], v[2]) for p, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sel-dir", required=True)
    ap.add_argument("--metered-dir", required=True)
    ap.add_argument("--metered-ks", nargs="+", default=["0.5", "1", "3", "20"])
    ap.add_argument("--n-values", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--reward-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--judges", nargs="+",
                    default=["microsoft/Phi-3.5-mini-instruct",
                             "meta-llama/Meta-Llama-3.1-8B-Instruct"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=606)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = load_baseline(a.metered_dir)                     # the k=-1 opponent, one per prompt
    cands = load_candidates(a.sel_dir)                      # prompt -> [(seed, ?, prompt, gen), ...]
    ctrl = metered_arm(a.metered_dir, "0")                  # the shared anchor-alone control
    metered = {k: metered_arm(a.metered_dir, k) for k in a.metered_ks}

    pids = sorted(set(base) & set(cands) & set(ctrl) & set.intersection(
        *[set(v) for v in metered.values()]))
    if not pids:
        raise SystemExit("[frontier] empty prompt intersection")
    n_max = max(a.n_values)
    pids = [p for p in pids if len(cands[p]) >= n_max]
    print(f"[frontier] {len(pids)} prompts, {len(metered)} metered arms, n<={n_max}", flush=True)
    prompts = {p: cands[p][0][2] for p in pids}

    # ---- the pointwise reward, which never touches the risky model ----
    rtok = AutoTokenizer.from_pretrained(a.reward_model, padding_side="left")
    if rtok.pad_token is None:
        rtok.pad_token = rtok.eos_token
    rmodel = AutoModelForCausalLM.from_pretrained(
        a.reward_model, torch_dtype=getattr(torch, a.dtype)).cuda().eval()
    assert yes_no_ids(rtok)["yes"] and yes_no_ids(rtok)["no"], rtok.name_or_path
    items = [(prompts[p], cands[p][j][3]) for p in pids for j in range(n_max)]
    scores = score_rewards(rmodel, rtok, items, "cuda")
    del rmodel
    torch.cuda.empty_cache()
    assert "Answer" in REWARD_TMPL, "the reward template is the one feat-088 registered"

    picks = {}                                              # (prompt, n) -> chosen generation
    for i, p in enumerate(pids):
        s = scores[i * n_max:(i + 1) * n_max]
        for n in a.n_values:
            j = max(range(n), key=lambda t: s[t])           # nested: n=1 is rank 0
            picks[(p, n)] = cands[p][j][3]

    arms = {"anchor alone (control)": {p: ctrl[p][0] for p in pids}}
    spend = {"anchor alone (control)": 0.0}
    for k in a.metered_ks:
        arms[f"metered, k={k}"] = {p: metered[k][p][0] for p in pids}
        spend[f"metered, k={k}"] = sum(metered[k][p][1] for p in pids) / len(pids)
    for n in a.n_values:
        arms[f"selection, n={n}"] = {p: picks[(p, n)] for p in pids}
        spend[f"selection, n={n}"] = math.log(n) - (n - 1) / n if n > 1 else 0.0

    rows = []
    for judge in a.judges:
        tok = AutoTokenizer.from_pretrained(judge, padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            judge, torch_dtype=getattr(torch, a.dtype)).cuda().eval()
        u = {}
        for name, gens in arms.items():
            flips = [rng.random() < 0.5 for _ in pids]
            batch = [(prompts[p], base[p], gens[p]) if f else (prompts[p], gens[p], base[p])
                     for p, f in zip(pids, flips)]
            verdicts = []
            for i in range(0, len(batch), 200):
                verdicts += judge_batch(model, tok, batch[i:i + 200], "cuda")
                print(f"[frontier] {judge.split('/')[-1]} {name}: "
                      f"{len(verdicts)}/{len(batch)}", flush=True)
            u[name] = {p: (1.0 if (v == "B") == f else 0.0) if v in ("A", "B") else 0.5
                       for p, f, v in zip(pids, flips, verdicts)}
        del model
        torch.cuda.empty_cache()

        c = "anchor alone (control)"
        for name in arms:
            vals = [u[name][p] for p in pids]
            lo, hi = boot_mean(vals, rng)
            rec = dict(judge=judge.split("/")[-1], arm=name, n_prompts=len(pids),
                       spend_nats=round(spend[name], 4),
                       u=round(sum(vals) / len(vals), 4), u_lo95=round(lo, 4), u_hi95=round(hi, 4))
            d = [u[name][p] - u[c][p] for p in pids]
            g_lo, g_hi = boot_mean(d, rng)
            rec.update(gain=round(sum(d) / len(d), 4), gain_lo95=round(g_lo, 4),
                       gain_hi95=round(g_hi, 4))
            if name.startswith("selection"):
                d2 = [u[name][p] - u["selection, n=1"][p] for p in pids]
                l2, h2 = boot_mean(d2, rng)
                rec.update(gain_vs_own_n1=round(sum(d2) / len(d2), 4),
                           gain_vs_own_n1_lo95=round(l2, 4), gain_vs_own_n1_hi95=round(h2, 4))
            else:
                rec.update(gain_vs_own_n1="", gain_vs_own_n1_lo95="", gain_vs_own_n1_hi95="")
            rows.append(rec)

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"frontier_pair{a.tag}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {path}")

    # ---- F1, against the band committed before the selection arm was generated ----
    scorer = "Phi-3.5-mini-instruct"
    for judge in sorted({r["judge"] for r in rows}):
        sub = [r for r in rows if r["judge"] == judge]
        print(f"\n  {judge}")
        for r in sub:
            print(f"    {r['arm']:26s} spend {r['spend_nats']:>9.4f}  u={r['u']:.4f} "
                  f"[{r['u_lo95']:.3f}, {r['u_hi95']:.3f}]  gain {r['gain']:+.4f} "
                  f"[{r['gain_lo95']:+.4f}, {r['gain_hi95']:+.4f}]")
        met = [r for r in sub if r["arm"].startswith("metered")]
        sel8 = next((r for r in sub if r["arm"] == "selection, n=8"), None)
        if not (met and sel8):
            continue
        best = max(met, key=lambda r: r["gain"])
        note = "  (registered scorer)" if judge == scorer else "  (reported, not scored)"
        if sel8["gain_lo95"] > 0 and sel8["gain"] >= best["gain"] - 0.03 \
                and sel8["spend_nats"] < 0.1 * best["spend_nats"]:
            f1 = "REPLICATES"
        elif best["gain"] - sel8["gain"] > 0.03 and (sel8["gain_hi95"] < best["gain_lo95"]):
            f1 = "REVERSED"
        elif sel8["gain_lo95"] > 0:
            f1 = "WEAKER"
        else:
            f1 = "WEAKER (selection does not resolve)"
        ratio = best["spend_nats"] / sel8["spend_nats"] if sel8["spend_nats"] else float("inf")
        print(f"    F1 best metered is {best['arm']} at {best['spend_nats']:.1f} nats for "
              f"{best['gain']:+.4f}; selection n=8 spends {sel8['spend_nats']:.3f} "
              f"({ratio:.0f}x less) for {sel8['gain']:+.4f} "
              f"[{sel8['gain_lo95']:+.4f}, {sel8['gain_hi95']:+.4f}]  ->  {f1}{note}")
    print("\n  F3 no leakage, certificate or s(x) number comes from this pair: the anchor is not a "
          "safe model and the arm generates nothing on the protected split.")


if __name__ == "__main__":
    main()
