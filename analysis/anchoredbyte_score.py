"""Score feat-187 (results/onset_prediction_anchoredbyte.md): the authors' AnchoredByte at Comma-7B + 70B.

Gates first, in the registered order, then the bands. No GPU.
  G0  every arm covers the 500 prompts, and every AnchoredByte record is prompt + generation
  G1  no trajectory's realised spend exceeds K + 1e-3
  G2  selection's per-prompt levels identical across the three passes
Writes results/anchoredbyte.csv: per k the binding and forced shares, spend against K, K / S_w, and
D1..D4 with the script's own readings.

Usage: .venv/bin/python analysis/anchoredbyte_score.py --out results
"""
import argparse
import csv
import glob
import json
import os
import statistics as st

KS = ("0.1", "0.5", "2")
D = "output/anchoredbyte/comma7b_70b"
# S_w for a 50-token window under Comma-7B: the audited anchor's 159.83 nats (window_vacuity.csv)
# rescaled by the two anchors' per-character surprisal on the same 1,124 passages (anchor_scaling.csv)
TINY, COMMA = "jacquelinehe/tinycomma-1.8b-llama3-tokenizer", "common-pile/comma-v0.1-2t"


def s_w_comma(res):
    w = next(r for r in csv.DictReader(open(os.path.join(res, "window_vacuity.csv")))
             if r["window_tokens"] == "50")
    rates = {}
    for r in csv.DictReader(open(os.path.join(res, "anchor_scaling.csv"))):
        rates.setdefault(r["model"], []).append(float(r["s_rate"]))
    return float(w["S_median_nats"]) * st.median(rates[COMMA]) / st.median(rates[TINY])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=D)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    sw = s_w_comma(a.out)
    rows, levels = [], {}
    for k in KS:
        recs = [json.loads(line) for f in sorted(glob.glob(os.path.join(a.dir, f"trajectories_k{k}_*.jsonl")))
                for line in open(f)]
        if not recs:
            print(f"[ab] k={k}: no trajectories yet")
            continue
        K = float(k) * 800
        g0 = len({r["metadata"]["prompt_id"] for r in recs}) == 500 and all(
            r["aggregate"]["full_text"] == r["prefix_analysis"]["prefix_text"] + r["aggregate"]["generation"]
            for r in recs)
        spend = [r["aggregate"]["total_spend"] for r in recs]
        g1 = max(spend) <= K + 1e-3
        nb = [r["aggregate"]["bytes_generated"] for r in recs]
        bind = sum(r["aggregate"]["steps_binding"] for r in recs) / max(1, sum(nb))
        forced = sum(r["aggregate"]["steps_forced_safe"] for r in recs) / max(1, sum(nb))
        row = dict(k=k, K=K, K_over_Sw=round(K / sw, 4), S_w_comma7b=round(sw, 2), prompts=len(recs),
                   G0="PASS" if g0 else "FAIL", G1="PASS" if g1 else "FAIL",
                   spend_median=round(st.median(spend), 2), spend_max=round(max(spend), 2),
                   binding_share=round(bind, 4), forced_share=round(forced, 4),
                   bytes_mean=round(st.mean(nb), 1), empty=sum(x == 0 for x in nb))
        h = os.path.join(a.out, f"order_averaged_h2h_ab70_k{k}.csv")
        if os.path.exists(h):
            for r in csv.DictReader(open(h, encoding="utf-8")):
                q = r["quantity"][:2]
                if q in ("D1", "D2", "D3", "D4", "D5"):
                    row[q] = f"{float(r['value']):+.4f} [{float(r['lo95']):+.4f}, {float(r['hi95']):+.4f}]"
                    row[f"{q}_reading"] = r["reading"]
            pp = os.path.join(a.out, f"order_averaged_h2h_per_prompt_ab70_k{k}.csv")
            levels[k] = {r["prompt_id"]: (r["u_sel_n64"], r["u_sel_n1"]) for r in csv.DictReader(open(pp))}
        rows.append(row)
    if len(levels) > 1:
        ref = next(iter(levels.values()))
        g2 = all(v == ref for v in levels.values())
        for r in rows:
            r["G2"] = "PASS" if g2 else "FAIL"
    keys = []
    for r in rows:
        keys += [x for x in r if x not in keys]
    path = os.path.join(a.out, "anchoredbyte.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print(f"S_w(Comma-7B, 50-token window) = {sw:.2f} nats; wrote {path}")


if __name__ == "__main__":
    main()
