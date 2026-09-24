"""Score feat-188 (results/onset_prediction_headline_replication.md). No GPU.

G0  every new arm covers the 500 prompts, the pool has 64 draws on each, and no seed of a new arm
    appears in the arm it replicates
G1  the new pool's n=1 (lowest-seed) empty fraction against the committed pool's, de-echoed, by a
    two-proportion z per class and on the total, |z| < 2.58; failing it makes the arm INVALID
Readings of D3 (and D1) off order_averaged_h2h_<tag>.csv: REPLICATES (lo95 > 0), REVERSES (hi95 < 0),
DOES NOT REPLICATE otherwise. Writes results/headline_replication.csv.
Usage: .venv/bin/python analysis/replic_score.py --out results
"""
import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_decoding import CLASSES, load_candidates  # noqa: E402

R = "output/replic"
TAGS = (("P1", "R2", "replic_opp"), ("P2", "R1", "replic"), ("", "R1b", "replic_nonempty"),
        ("", "R2b", "replic_opp_nonempty"), ("", "R0", "seed52_deecho"))


def seeds(d, k):
    out = {}
    for c in CLASSES:
        p = os.path.join(d, f"trajectories_k{k}_{c}.jsonl")
        if os.path.exists(p):
            for line in open(p):
                m = json.loads(line)["metadata"]
                out.setdefault(m["prompt_id"], set()).add(m["seed"])
    return out


def z2(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if 0 < p < 1 else 0
    return (k1 / n1 - k2 / n2) / se if se else 0.0


def reading(lo, hi):
    return "REPLICATES" if lo > 0 else "REVERSES" if hi < 0 else "DOES NOT REPLICATE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    rows = []
    # G0
    pairs = (("pool", f"{R}/sel_anchor64", "output/phase5/sel_anchor64", "0", 64),
             ("metered k=10", f"{R}/conc_k10", "output/phase2/conc_all", "10", 1),
             ("anchor k=0", f"{R}/anchor_k0", "output/sweep_plain", "0", 1),
             ("opponent k=-1", f"{R}/opp", "output/sweep_plain", "-1", 1))
    g0 = True
    for name, new, old, k, n in pairs:
        s_new, s_old = seeds(new, k), seeds(old, k)
        cover = sum(1 for p in s_old if p in s_new and len(s_new[p]) >= n)
        clash = sum(1 for p in s_new if p in s_old and s_new[p] & s_old[p])
        ok = len(s_new) >= 500 and cover >= 500 and clash == 0
        g0 &= ok
        rows.append(dict(gate="G0", quantity=f"{name}: prompts covered with {n} draws, seeds shared",
                         value=f"{len(s_new)} prompts, {clash} with a shared seed", reading="PASS" if ok else "FAIL"))
    # G1
    newc = load_candidates(f"{R}/sel_anchor64", deecho=True)
    oldc = load_candidates("output/phase5/sel_anchor64", deecho=True)
    g1 = True
    tot = [0, 0, 0, 0]
    for c in CLASSES:
        kn = [not v[0][3].strip() for v in newc.values() if v[0][1] == c]
        ko = [not v[0][3].strip() for v in oldc.values() if v[0][1] == c]
        if not ko:
            continue
        z = z2(sum(kn), len(kn), sum(ko), len(ko))
        tot = [tot[0] + sum(kn), tot[1] + len(kn), tot[2] + sum(ko), tot[3] + len(ko)]
        g1 &= abs(z) < 2.58
        rows.append(dict(gate="G1", quantity=f"n=1 empty, {c}: new {sum(kn)}/{len(kn)} vs record {sum(ko)}/{len(ko)}",
                         value=round(z, 3), reading="PASS" if abs(z) < 2.58 else "FAIL"))
    z = z2(*tot)
    g1 &= abs(z) < 2.58
    rows.append(dict(gate="G1", quantity=f"n=1 empty, total: new {tot[0]}/{tot[1]} vs record {tot[2]}/{tot[3]}",
                     value=round(z, 3), reading="PASS" if abs(z) < 2.58 else "FAIL"))
    # readings
    for band, run, tag in TAGS:
        path = os.path.join(a.out, f"order_averaged_h2h_{tag}.csv")
        if not os.path.exists(path):
            rows.append(dict(gate=band or run, quantity=f"{run}: not yet judged", value="", reading=""))
            continue
        for r in csv.DictReader(open(path, encoding="utf-8")):
            q = r["quantity"][:2]
            if q not in ("D1", "D3"):
                continue
            lo, hi = float(r["lo95"]), float(r["hi95"])
            b = band if q == "D3" else ("P3" if run in ("R1", "R2") else "")
            rows.append(dict(gate=b, quantity=f"{run} {r['quantity']}",
                             value=f"{float(r['value']):+.4f} [{lo:+.4f}, {hi:+.4f}]",
                             reading=(f"descriptive ({reading(lo, hi)})" if run == "R0" else
                                      reading(lo, hi) if (g0 and g1) else "INVALID (gate)")))
    path = os.path.join(a.out, "headline_replication.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["gate", "quantity", "value", "reading"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print(f"G0 {'PASS' if g0 else 'FAIL'}  G1 {'PASS' if g1 else 'FAIL'}; wrote {path}")


if __name__ == "__main__":
    main()
