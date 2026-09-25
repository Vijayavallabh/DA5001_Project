"""Score feat-208 (results/onset_prediction_blockwise_replication.md): feat-201's installment comparison
on a disjoint draw, under judges B and G, and judge G read post hoc on feat-201's own texts. No GPU.

Per-prompt order-averaged levels from different passes on one host are compared prompt by prompt (a
level is a deterministic function of the prompt, the two texts and the judge; caution (ap)). Writes
results/blockwise_replication.csv.

Usage: .venv/bin/python analysis/blockwise_replication.py --out results
"""
import argparse
import csv
import glob
import json
import os
import random

N_BOOT = 10000


def levels(res, tag):
    return {r["prompt_id"]: r for r in
            csv.DictReader(open(os.path.join(res, f"order_averaged_h2h_per_prompt_blockwise_{tag}.csv")))}


def gain(res, tag):
    return next(r for r in csv.DictReader(open(os.path.join(res, f"order_averaged_h2h_blockwise_{tag}.csv")))
                if r["quantity"].startswith("D4"))


def texts(d, arm):
    out = {}
    for f in glob.glob(os.path.join(d, arm, f"trajectories_k{arm}_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            out[r["metadata"]["prompt_id"]] = r["aggregate"]["generation"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--new", default="output/feat208")
    ap.add_argument("--old", default="output/feat201")
    ap.add_argument("--logs", default="output/logs")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    rng = random.Random(208)
    rows = []

    def add(band, quantity, value, lo="", hi="", reading="", n=""):
        rows.append(dict(band=band, quantity=quantity, value=value, lo95=lo, hi95=hi, reading=reading, n=n))

    def contrast(band, what, A, x, B, y, labels=("INSTALLMENTS WIN", "ONCE WINS")):
        d = [float(A[p][f"u_{x}"]) - float(B[p][f"u_{y}"]) for p in sorted(B)]
        n = len(d)
        xs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
        lo, hi = xs[int(0.025 * N_BOOT)], xs[int(0.975 * N_BOOT)]
        add(band, what, round(sum(d) / n, 4), round(lo, 4), round(hi, 4),
            labels[0] if lo > 0 else labels[1] if hi < 0 else "TIE", n)

    new = {L: f"blk{L}n64s2" for L in (10, 25, 200)}
    old = {L: f"blk{L}n64" for L in (10, 25, 200)}

    # G0: coverage, block logs, and the pure-sampling line of each new generation
    for L, arm in new.items():
        t = texts(a.new, arm)
        blog = list(csv.DictReader(open(os.path.join(a.new, arm, f"blocks_{arm}.csv"))))
        ok = len(t) == 500 and len({r["prompt_id"] for r in blog if r["block"] == "0"}) == 500
        add("G0", f"{arm}: 500 records, block log complete", float(ok), reading="PASS" if ok else "FAIL", n=len(t))
    lines = [ln for f in glob.glob(os.path.join(a.logs, "feat208_q*.log")) for ln in open(f, errors="replace")
             if "effective sampling" in ln]
    ok = len(lines) == 3 and all("temperature=1.0 top_k=0 top_p=1.0" in ln for ln in lines)
    add("G0", "three generation logs print pure sampling", len(lines), reading="PASS" if ok else "FAIL")
    # G1: the draws differ from feat-201's
    for L in new:
        tn, to = texts(a.new, new[L]), texts(a.old, old[L])
        both = [p for p in tn if tn[p].strip() and to.get(p, "").strip()]
        diff = sum(tn[p] != to[p] for p in both) / len(both)
        add("G1", f"L={L}: served text differs from feat-201's, share of prompts both non-empty", round(diff, 4),
            reading="PASS" if diff >= 0.90 else "FAIL", n=len(both))
    # G2: the judge-B path, against feat-198 on the same host
    ref = {r["prompt_id"]: r for r in
           csv.DictReader(open(os.path.join(a.results, "order_averaged_h2h_per_prompt_scorer_gemma27b.csv")))}
    for L, arm in new.items():
        U = levels(a.results, arm)
        same = sum(U[p]["u_metered_k10"] == ref[p]["u_metered_k10"] and U[p]["u_anchor_k0"] == ref[p]["u_anchor_k0"]
                   for p in ref)
        add("G2", f"{arm}: meter and anchor levels equal feat-198's", same, reading="PASS" if same == 500 else "FAIL",
            n=500)

    B = {L: levels(a.results, new[L]) for L in new}
    G = {L: levels(a.results, f"{new[L]}_judgeG") for L in new}
    contrast("P1", "judge B, new draw: L=10 minus once", B[10], new[10], B[200], new[200])
    contrast("P2", "judge B, new draw: L=25 minus once", B[25], new[25], B[200], new[200])
    contrast("P3", "judge G, new draw: L=10 minus once", G[10], new[10], G[200], new[200])
    contrast("desc", "judge G, new draw: L=25 minus once", G[25], new[25], G[200], new[200])
    for L in new:
        for j, tag in (("B", new[L]), ("G", f"{new[L]}_judgeG")):
            r = gain(a.results, tag)
            add("desc", f"judge {j}, new draw: {new[L]} gain over the anchor", float(r["value"]), float(r["lo95"]),
                float(r["hi95"]), r["reading"], r["n"])
        t = texts(a.new, new[L])
        w = sorted(len(x.split()) for x in t.values())
        add("desc", f"{new[L]}: median words / empty", f"{w[len(w) // 2]} / {sum(not x.strip() for x in t.values())}")
    # post hoc: judge G on feat-201's own texts
    PG = {L: levels(a.results, f"blk{L}n64_judgeG") for L in (10, 25, 50, 200)}
    for L in (10, 25, 50):
        contrast("posthoc", f"judge G, feat-201's draw: L={L} minus once", PG[L], f"blk{L}n64", PG[200], "blk200n64")
    for L in (10, 25, 50, 200):
        r = gain(a.results, f"blk{L}n64_judgeG")
        add("posthoc", f"judge G, feat-201's draw: blk{L}n64 gain over the anchor", float(r["value"]),
            float(r["lo95"]), float(r["hi95"]), r["reading"], r["n"])

    path = os.path.join(a.out, "blockwise_replication.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print("wrote", path)


if __name__ == "__main__":
    main()
