"""Score feat-211 (results/onset_prediction_cpk_baseline.md) and feat-212
(results/onset_prediction_scorer_judge_factorial.md) against their registered predictions.

Reads the judge passes written by scripts/run_feat211.sh (results/matched_h2h_<tag>.csv), CP-k's arm summary
(results/cpk_baseline.csv) and its leakage (results/cpk_extraction.csv). Writes results/cpk_baseline_scoring.csv
and results/scorer_judge_factorial.csv. Every verdict is computed from the CSVs; nothing is typed in.

Usage: .venv/bin/python analysis/score_feat211.py [--results results]
"""
import argparse
import csv
import os

JUDGES_212 = (("B", "cpk_B_hostb"), ("C", "fact_C"), ("D", "fact_D"), ("E", "fact_E"), ("F", "fact_F"),
              ("G", "cpk_G"))
GRID = (4.1589, 33.271, 83.178, 159.83)


def read_pass(res, tag):
    path = os.path.join(res, f"matched_h2h_{tag}.csv")
    return {(r["quantity"], r["arm"]): r for r in csv.DictReader(open(path))}


def row(P, quantity, arm):
    r = P[(quantity, arm)]
    return float(r["value"]), float(r["lo95"]), float(r["hi95"]), r["reading"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    R = a.results
    out = []

    def add(fid, pid, quantity, v, lo, hi, reading, predicted, right):
        out.append(dict(feature=fid, prediction=pid, quantity=quantity, value=v, lo95=lo, hi95=hi, reading=reading,
                        predicted=predicted, verdict="" if right is None else ("RIGHT" if right else "WRONG")))

    # ---- feat-211 -------------------------------------------------------------------------------------------
    cpk = {r["certificate_nats"]: r for r in csv.DictReader(open(os.path.join(R, "cpk_baseline.csv")))}
    for c in GRID:                                       # G1: the certificate each arm was built at
        r = cpk[f"{c:g}"] if f"{c:g}" in cpk else cpk[str(c)]
        assert abs(float(r["check_certificate"]) - c) < 1e-4, (c, r["check_certificate"])
    B, G = read_pass(R, "cpk_B_hostb"), read_pass(R, "cpk_G")
    for pid, diff in (("C1", "blk10n64 - cpk_83.18"), ("C2", "blk25n64 - cpk_33.27"), ("C3", "sel_n64 - cpk_4.16")):
        v, lo, hi, rd = row(B, "difference", diff)
        add("feat-211", pid, f"B: {diff}", v, lo, hi, rd, "CONFIRMED", rd == "CONFIRMED")
    for pid, diff in (("C4", "blk10n64 - cpk_83.18"), ("C4", "blk25n64 - cpk_33.27"), ("C4", "sel_n64 - cpk_4.16")):
        v, lo, hi, rd = row(G, "difference", diff)
        vb = row(B, "difference", diff)[0]
        add("feat-211", pid, f"G: {diff}", v, lo, hi, rd, "same sign as B", (v > 0) == (vb > 0))
    r83 = cpk["83.178"]
    acc, allm = float(r83["accepted_median_tokens"] or "nan"), float(r83["all_draws_median_tokens"])
    add("feat-211", "C5", "accepted median tokens at C=83.178 vs all draws", acc, allm, "", "",
        "below", acc < allm)
    leak = {r["certificate_nats"]: r for r in csv.DictReader(open(os.path.join(R, "cpk_extraction.csv")))}
    worst = max(float(leak[f"{c:g}"]["nv_recall_mean"]) for c in GRID)
    add("feat-211", "C6", "max near-verbatim recall over C <= 159.83", worst, "", "", "", "< 0.01", worst < 0.01)
    # descriptive
    for c in GRID:
        r = cpk[f"{c:g}"]
        add("feat-211", "desc", f"C={c:g}: served from the risky model, %", float(r["served_risky_pct"]), "", "", "",
            "", None)
        for P, j in ((B, "B"), (G, "G")):
            v, lo, hi, rd = row(P, "gain", f"cpk_{c:.2f} - anchor_k0")
            add("feat-211", "desc", f"{j}: gain cpk_{c:.2f}", v, lo, hi, rd, "", None)
    for diff in ("cpk_83.18 - pw_83.18", "cpk_33.27 - pw_33.27", "cpk_159.83 - metered_k10"):
        for P, j in ((B, "B"), (G, "G")):
            v, lo, hi, rd = row(P, "difference", diff)
            add("feat-211", "desc", f"{j}: {diff}", v, lo, hi, rd, "", None)
    grid = sorted((k for k in leak if k.replace(".", "").isdigit()), key=float)
    onset = [k for k in grid if float(leak[k]["nv_recall_mean"]) >= 0.01]
    add("feat-211", "desc", "leakage onset (smallest C with recall >= 0.01), nats", onset[0] if onset else "none",
        "", "", "", "", None)
    for k in grid:
        add("feat-211", "desc", f"leakage at C={k}: recall mean / served from memoriser %",
            float(leak[k]["nv_recall_mean"]), float(leak[k]["served_from_memoriser_pct"]), "", "", "", None)
    add("feat-211", "desc", "memoriser alone, near-verbatim recall",
        float(leak["memoriser alone"]["nv_recall_mean"]), "", "", "", "", None)

    # ---- feat-212 -------------------------------------------------------------------------------------------
    fact = []
    n_conf = 0
    for j, tag in JUDGES_212:
        P = read_pass(R, tag)
        s = row(P, "difference", "sel_n64 - sel_g64")
        dq = row(P, "difference", "sel_n64 - metered_k10")
        dg = row(P, "difference", "sel_g64 - metered_k10")
        n_conf += s[3] == "CONFIRMED"
        fact.append(dict(judge=j, tag=tag, S=s[0], S_lo=s[1], S_hi=s[2], S_reading=s[3], D3_qwen=dq[0],
                         D3_qwen_lo=dq[1], D3_qwen_hi=dq[2], D3_qwen_reading=dq[3], D3_gemma=dg[0],
                         D3_gemma_lo=dg[1], D3_gemma_hi=dg[2], D3_gemma_reading=dg[3]))
    add("feat-212", "F1", "judges under which S = sel_n64 - sel_g64 is CONFIRMED", n_conf, "", "", "", ">= 5 of 6",
        n_conf >= 5)
    for P, j in ((B, "B"), (G, "G")):
        for diff in ("sel_g64 - win_4.16", "sel_g64 - frontpw_4.16"):
            v, lo, hi, rd = row(P, "difference", diff)
            add("feat-212", "M1", f"{j}: {diff}", v, lo, hi, rd, "CONFIRMED", rd == "CONFIRMED")

    with open(os.path.join(R, "cpk_baseline_scoring.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    with open(os.path.join(R, "scorer_judge_factorial.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fact[0]))
        w.writeheader()
        w.writerows(fact)
    for r in out:
        print(r)
    for r in fact:
        print(r)


if __name__ == "__main__":
    main()
