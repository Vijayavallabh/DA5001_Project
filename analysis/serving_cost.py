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

FLOPs is the fair common currency and it is not the whole story in either direction: decode steps
are memory-bandwidth-bound and poorly utilised while a scoring prefill is compute-bound and runs
near peak, and selection's n draws are independent where a metered decode step is not. Both effects
push the same way -- a FLOP ratio OVERSTATES selection's wall-clock cost. We do not measure
wall-clock and do not claim it; the ratios below are what they say they are.

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
# Parameter counts in billions, COUNTED OFF THE LOADED CHECKPOINT and not read off its name.
# The three constants below were a model's label until 2026-09-15, and one of them was materially
# wrong: Qwen2.5-7B-Instruct holds 7.6156B parameters, not 7.0, which understated selection's
# serving cost by 8.8% everywhere it appeared -- the published 57.5x at n=64 is 61.3x. Found while
# measuring the 1.5B and 3B scorers for feat-117 and fixed in the direction that costs us. Reproduce
# every one of these with
#   sum(p.numel() for p in AutoModelForCausalLM.from_pretrained(<id>).parameters())
# and see tests/test_compute_matched.py, which pins them to a tolerance a mislabelled model fails.
P_ANCHOR = 1.7586   # jacquelinehe/tinycomma-1.8b-llama3-tokenizer  (label says 1.8)
P_RISKY = 8.0303    # meta-llama/Meta-Llama-3.1-8B-Instruct         (label says 8)
P_SCORER = 7.6156   # Qwen/Qwen2.5-7B-Instruct, the pointwise reward (label says 7)
P_SMALL = 0.4940    # Qwen/Qwen2.5-0.5B-Instruct                     (label says 0.5)
                    # Until feat-116 this was a round 0.5 and an untested counterfactual offered as
                    # a route out of the compute concession. The arm ran and REFUTED it: the small
                    # scorer's judged gain never reaches the metered decoder's, peaks at n=16 and
                    # falls thereafter (results/compute_matched.csv). The rows stay because they
                    # are now a measured negative and deleting one would hide it; what is gone is
                    # the claim they were written to support.


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
    for n in (1, 2, 4, 8, 16, 32, 64):
        c = n * (P_ANCHOR + P_SMALL) * L
        rows.append(dict(mechanism="selection anchoring (small scorer)", n=n,
                         scorer="Qwen2.5-0.5B", seq_tokens=int(L),
                         cost_bparam_tokens=round(c, 1), ratio_vs_metered=round(c / metered, 2),
                         note="feat-116 RAN and refuted the route: see compute_matched.csv, "
                              "this cost buys no gain the metered decoder does not already have"))

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
