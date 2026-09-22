"""feat-176: score the Gutenberg completion workload. Gates in the registered order, then bands.

The gates are code and not a reading, and the script REFUSES to print a band if one fails ---
caution (v): "when a gate fails, do not compute the band just to see". Bands:
results/onset_prediction_fifth_workload.md, committed before any generation.

Usage (where output/gutenberg lives):
  .venv/bin/python analysis/score_fifth_workload.py --out results
"""
import argparse
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_decoding import load_baseline, load_candidates  # noqa: E402
from analysis.workload_degeneracy import activity  # noqa: E402

CORPUS = "data/bench/gutenberg"
ROOT = "output/gutenberg"
TARGET = 0.08008           # feat-168's chosen arm's measured rate, derived not typed
BIND_K = 0.9               # the refined grid's argmin, registered before it ran
HEADER = "Complete the prefix:"
PROMPT_WORDS = 169.4       # printed by build_gutenberg_bench.py and recorded in the registration


def d3(sfx):
    p = f"results/order_averaged_h2h__gutenberg_{sfx}.csv"
    assert os.path.exists(p), p
    r = [x for x in csv.DictReader(open(p, encoding="utf-8")) if x["quantity"].startswith("D3")]
    assert len(r) == 1
    return float(r[0]["value"]), float(r[0]["lo95"]), float(r[0]["hi95"]), r[0]["reading"].strip()


def band(g, lo, hi):
    """B1/B2 as registered: WITH OURS if > 0 clear of zero, WITH ALPACAEVAL if < 0 clear of it."""
    if lo > 0:
        return "WITH OURS"
    if hi < 0:
        return "WITH ALPACAEVAL"
    return "UNRESOLVED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    gates, fails = [], []

    def gate(name, ok, detail):
        gates.append((name, "PASS" if ok else "FAIL", detail))
        if not ok:
            fails.append(name)

    # ---- G-cal: the refined grid's argmin must satisfy G0's 2x -------------------------------
    ref = list(csv.DictReader(open("results/gutenberg_kcal_refined.csv", encoding="utf-8")))
    chosen = min(ref, key=lambda r: abs(float(r["activity"]) - TARGET))
    ratio = float(chosen["activity"]) / TARGET
    gate("G-cal (refined grid brackets and its argmin clears 2x)",
         sum(1 for r in ref if float(r["activity"]) > TARGET) >= 1
         and sum(1 for r in ref if float(r["activity"]) < TARGET) >= 1
         and 0.5 <= ratio <= 2.0,
         f"argmin k={chosen['k']} at {float(chosen['activity']):.5f} = {ratio:.2f}x the target")

    # ---- G0: the binding cell binds and is not the opponent ----------------------------------
    pp = {r["prompt_id"] for r in csv.DictReader(
        open("results/order_averaged_h2h_per_prompt__gutenberg_conc_bind.csv", encoding="utf-8"))}
    act, tot, _n = activity(f"{ROOT}/conc_bind", BIND_K, pp)
    from analysis.order_averaged_h2h import lowest_seed
    from analysis.utility import load_arm
    met = lowest_seed(load_arm(f"{ROOT}/conc_bind", BIND_K, "kl"))
    opp = load_baseline(f"{ROOT}/baseline")
    shared = sorted(set(met) & set(opp) & pp)
    same = sum(1 for p in shared if met[p][1] == opp[p])
    gate("G0a (activity within 2x of the target)", 0.5 <= (act / tot) / TARGET <= 2.0,
         f"{act}/{tot} = {act / tot:.5%}, {(act / tot) / TARGET:.2f}x")
    gate("G0b (<10% byte-identical to the opponent)", same / len(shared) < 0.10,
         f"{same}/{len(shared)} = {same / len(shared):.1%}")

    # ---- G1: the corpus is what the registration names ---------------------------------------
    corpus = [json.loads(l) for l in open(f"{CORPUS}/factscore.jsonl", encoding="utf-8")]
    books = {r["source_novel"] for r in corpus}
    words = statistics.fmean(len(r["prompt_text"].split()) for r in corpus)
    served = [json.loads(l)["aggregate"]["full_text"]
              for l in open(f"{ROOT}/conc_bind/trajectories_k{BIND_K:g}_factual.jsonl",
                            encoding="utf-8")]
    hdr = sum(1 for t in served if t.startswith(HEADER))
    gate("G1 (500 prompts, factual slot, no header, >=40 books, length within 10%)",
         len(corpus) == 500 and hdr == 0 and len(books) >= 40
         and abs(words - PROMPT_WORDS) / PROMPT_WORDS < 0.10,
         f"{len(corpus)} prompts, {len(books)} books, {words:.1f} words, {hdr} carry the header")

    # ---- G2: the anchor is not degenerate here ------------------------------------------------
    cands = load_candidates(f"{ROOT}/sel_anchor64")
    n1 = [c[0][3] for c in cands.values() if c]
    empty = sum(1 for t in n1 if not t.strip()) / len(n1)
    gate("G2 (<10% empty completions at n=1)", empty < 0.10,
         f"{empty:.1%} empty over {len(n1)} prompts")

    print(f"{'gate':<62} {'reading':<6} detail")
    for name, rd, detail in gates:
        print(f"{name:<62} {rd:<6} {detail}")

    if fails:
        print(f"\n*** {len(fails)} GATE(S) FAILED: {fails}")
        print("*** NOT SCORED. The bands are not computed (caution (v)).")
        return

    print("\nAll gates pass; reading the bands.")
    rows = []
    for label, sfx, k in (("B1 binding", "conc_bind", BIND_K), ("B2 vacuous", "conc_k10", 10.0)):
        g, lo, hi, rd = d3(sfx)
        rows.append(dict(band=label, k=k, d3=g, lo95=lo, hi95=hi,
                         verdict=band(g, lo, hi), h2h_reading=rd))
        print(f"  {label:<12} k={k:<5} {g:+.4f} [{lo:+.4f}, {hi:+.4f}]  **{band(g, lo, hi)}**")

    out = os.path.join(a.out, "fifth_workload.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
