"""Order consistency of every judge on the headline's four arms (Review 4, Q6). No GPU.

Each order-averaged pass judges every pair in both presentation orders and records, per arm, the share
of prompts whose two verdicts agree (A then B, B then A, or Tie twice). This collects those rows from the
six de-echoed passes over the headline's texts, one per judge, beside each pass's D3.

Usage: .venv/bin/python analysis/judge_order_consistency.py --out results
"""
import argparse
import csv
import os

PASSES = (("B", "Phi-3.5-mini-instruct", "order_averaged_h2h_deecho.csv"),
          ("C", "Llama-3.1-8B-Instruct (the opponent's checkpoint)", "order_averaged_h2h__opp_committed_judgeC_deecho.csv"),
          ("D", "Qwen2.5-72B-Instruct", "order_averaged_h2h__qwen72b_deecho.csv"),
          ("E", "Mixtral-8x7B-Instruct", "order_averaged_h2h__mixtral_deecho.csv"),
          ("F", "Qwen2.5-14B-Instruct", "order_averaged_h2h__qwen14b_deecho.csv"),
          ("G", "gemma-2-27b-it", "order_averaged_h2h__gemma27b_deecho.csv"))
ARMS = ("sel_n64", "sel_n1", "metered_k10", "anchor_k0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = []
    for judge, model, f in PASSES:
        rows = list(csv.DictReader(open(os.path.join(a.results, f))))
        cons = {r["arm"]: float(r["value"]) for r in rows if r["quantity"] == "order consistency"}
        d3 = next(r for r in rows if r["quantity"].startswith("D3"))
        assert set(ARMS) <= set(cons), (f, cons)
        out.append(dict(judge=judge, model=model, **{f"consistency_{x}": cons[x] for x in ARMS},
                        consistency_min=min(cons[x] for x in ARMS), consistency_max=max(cons[x] for x in ARMS),
                        d3=d3["value"], d3_lo95=d3["lo95"], d3_hi95=d3["hi95"], d3_reading=d3["reading"],
                        n=d3["n"], source=f))
    path = os.path.join(a.out, "judge_order_consistency.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(f"{r['judge']} {r['model']:50s} {r['consistency_min']:.3f}-{r['consistency_max']:.3f}  "
              f"D3 {r['d3']} [{r['d3_lo95']}, {r['d3_hi95']}] {r['d3_reading']}")
    print("wrote", path)


if __name__ == "__main__":
    main()
