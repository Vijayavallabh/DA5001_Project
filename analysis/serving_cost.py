"""feat-114: what the two mechanisms cost to serve, in the currency a deployer actually pays.

Every efficiency comparison in this paper is denominated in nats, which is the certificate's
currency and not the deployer's. A reviewer of the v8 draft put it plainly: selection at n = 64
runs 64 generations plus 64 scoring passes, the metered decoder runs two models per token, and
"+0.142 for 3.175 nats against +0.072 for 171.3" systematically flatters the arm that is far more
expensive per served response. That is correct, and this script measures how much.

The model is the standard one: a transformer forward pass over L tokens costs about 2 * P * L
FLOPs, so with a KV cache a response of T generated tokens on a prompt of L_p tokens processes
L_p + T tokens. Costs are reported in billion-parameter-tokens, which is FLOPs / 2 and is
proportional to FLOPs for every arm here, so the ratios are the quantity of interest.

  metered decoder   both models at prefill and at every decode step: (P_anchor + P_risky)(L_p + T)
  selection, n      n independent anchor generations, then one scoring PREFILL per candidate:
                    n (P_anchor + P_scorer)(L_p + T)

The scoring pass is a prefill over the whole sequence, not a single token; costing it as one token
understates selection by about 4x and is the easy mistake here.

Writes <out>/serving_cost.csv. No GPU, no generation: token counts come from the logged
trajectories and the prompt corpus.

Usage:
  .venv/bin/python analysis/serving_cost.py --out results
"""
import argparse
import csv
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Parameter counts in billions, from the model cards of the checkpoints these arms actually ran.
P_ANCHOR = 1.8      # jacquelinehe/tinycomma-1.8b-llama3-tokenizer
P_RISKY = 8.0       # meta-llama/Llama-3.1-8B-Instruct
P_SCORER = 7.0      # Qwen/Qwen2.5-7B-Instruct, the pointwise reward
P_SMALL = 0.5       # an untested counterfactual scorer, reported as such and never as a result


def median_tokens(pattern, field="generation_length_tokens"):
    vals = []
    for f in glob.glob(pattern):
        for line in open(f):
            v = json.loads(line)["aggregate"].get(field)
            if v:
                vals.append(v)
    return st.median(vals) if vals else None, len(vals)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metered-glob", default="output/phase2/conc_all/trajectories_k10_*.jsonl")
    ap.add_argument("--sel-glob", default="output/phase5/sel_anchor64/trajectories_*.jsonl")
    ap.add_argument("--prompt-tokens", type=float, default=17.0,
                    help="median prompt length under the anchor tokenizer; see the docstring")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    t_met, n_met = median_tokens(a.metered_glob)
    t_sel, n_sel = median_tokens(a.sel_glob)
    assert t_met and t_sel, "no logged token counts found"
    # Both arms are capped at the same T_max, which is why one sequence length serves both.
    L = a.prompt_tokens + t_met
    print(f"[cost] metered {n_met} trajectories, median {t_met:.0f} generated tokens")
    print(f"[cost] selection {n_sel} candidates, median {t_sel:.0f} generated tokens")
    print(f"[cost] sequence {L:.0f} tokens (prompt {a.prompt_tokens:.0f} + generation {t_met:.0f})")

    metered = (P_ANCHOR + P_RISKY) * L
    rows = [dict(mechanism="metered decoder, k=10", n="", scorer="--",
                 seq_tokens=int(L), cost_bparam_tokens=round(metered, 1), ratio_vs_metered=1.0,
                 note="anchor and risky model at prefill and at every decode step")]
    for n in (1, 2, 4, 8, 16, 32, 64):
        c = n * (P_ANCHOR + P_SCORER) * L
        rows.append(dict(mechanism="selection anchoring", n=n, scorer="Qwen2.5-7B",
                         seq_tokens=int(L), cost_bparam_tokens=round(c, 1),
                         ratio_vs_metered=round(c / metered, 2),
                         note="n anchor generations + n scoring prefills"))
    for n in (8, 64):
        c = n * (P_ANCHOR + P_SMALL) * L
        rows.append(dict(mechanism="selection anchoring (counterfactual scorer)", n=n,
                         scorer="0.5B, NOT RUN", seq_tokens=int(L),
                         cost_bparam_tokens=round(c, 1), ratio_vs_metered=round(c / metered, 2),
                         note="arithmetic only; no arm was run with a 0.5B scorer"))

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "serving_cost.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"  {r['mechanism']:42s} n={str(r['n']):<3} {r['cost_bparam_tokens']:>9.0f}  "
              f"{r['ratio_vs_metered']:>6.2f}x")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
