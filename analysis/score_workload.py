"""Score one workload arm: gates in the registered order, then bands.

Generalised from feat-176's `score_fifth_workload.py` (git mv, same file) once feat-174 and
feat-177 needed the same protocol. The gates are code and not a reading, and the script REFUSES to
print a band if one fails --- caution (v): "when a gate fails, do not compute the band just to
see". Every band is committed before generation in `results/onset_prediction_*_workload.md`.

Each workload's SPEC below is its registration transcribed: only the checks that document names
are run, because a gate nobody registered can fail a valid arm as easily as it can catch a bad one,
and two workloads' G1s are genuinely different (Gutenberg counts books, CoTaEval-QA names one).
Outputs are per workload (caution (ax): two passes must never share a file).

Usage:
  .venv/bin/python analysis/score_workload.py --workload gutenberg
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

# The rate feat-168's CHOSEN AlpacaEval arm realised (11,499 of 143,598 = 8.008%), not the
# committed metered arm's own 0.008% -- see results/onset_prediction_mixtral_power_k.md and the
# correction in adfb517. Every workload since is matched to that arm's binding rate.
TARGET = 0.08008
HEADER = "Complete the prefix:"

SPECS = {
    # feat-174, results/onset_prediction_fourth_workload.md
    "cotaeval_qa": dict(
        root="output/cotaeval_qa", corpus="data/bench/cotaeval_qa", bind_k=1.4,
        refined="results/cotaeval_qa_kcal_refined.csv", out="fourth_workload.csv",
        n_prompts=500, novel_is="newsqa"),
    # feat-176, results/onset_prediction_fifth_workload.md
    "gutenberg": dict(
        root="output/gutenberg", corpus="data/bench/gutenberg", bind_k=0.9,
        refined="results/gutenberg_kcal_refined.csv", out="fifth_workload.csv",
        n_prompts=500, min_books=40, prompt_words=169.4, no_header=True, max_empty=0.10),
    # feat-177, results/onset_prediction_sixth_workload.md
    "unseenbooks": dict(
        root="output/unseenbooks", corpus="data/bench/unseenbooks", bind_k=None,
        refined="results/unseenbooks_kcal_refined.csv", out="sixth_workload.csv",
        n_prompts=500, min_books=20, prompt_words=163.9, no_header=True, max_empty=0.10,
        leak_csv="results/g3_unseenbooks.csv"),
}


def d3(root_tag, sfx):
    p = f"results/order_averaged_h2h__{root_tag}_{sfx}.csv"
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
    ap.add_argument("--workload", required=True, choices=sorted(SPECS))
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    s = SPECS[a.workload]
    root, tag = s["root"], os.path.basename(s["root"])
    gates, fails = [], []

    def gate(name, ok, detail):
        gates.append((name, "PASS" if ok else "FAIL", detail))
        if not ok:
            fails.append(name)

    # ---- G-cal: the refined grid's argmin must satisfy G0's 2x -------------------------------
    ref = list(csv.DictReader(open(s["refined"], encoding="utf-8")))
    chosen = min(ref, key=lambda r: abs(float(r["activity"]) - TARGET))
    bind_k = s["bind_k"] if s["bind_k"] is not None else float(chosen["k"])
    ratio = float(chosen["activity"]) / TARGET
    gate("G-cal (refined grid brackets and its argmin clears 2x)",
         sum(1 for r in ref if float(r["activity"]) > TARGET) >= 1
         and sum(1 for r in ref if float(r["activity"]) < TARGET) >= 1
         and 0.5 <= ratio <= 2.0 and abs(float(chosen["k"]) - bind_k) < 1e-9,
         f"argmin k={chosen['k']} at {float(chosen['activity']):.5f} = {ratio:.2f}x the target")

    # ---- G0: the binding cell binds and is not the opponent ----------------------------------
    pp = {r["prompt_id"] for r in csv.DictReader(
        open(f"results/order_averaged_h2h_per_prompt__{tag}_conc_bind.csv", encoding="utf-8"))}
    act, tot, _n = activity(f"{root}/conc_bind", bind_k, pp)
    from analysis.order_averaged_h2h import lowest_seed
    from analysis.utility import load_arm
    met = lowest_seed(load_arm(f"{root}/conc_bind", bind_k, "kl"))
    opp = load_baseline(f"{root}/baseline")
    shared = sorted(set(met) & set(opp) & pp)
    same = sum(1 for p in shared if met[p][1] == opp[p])
    gate("G0a (activity within 2x of the target)", 0.5 <= (act / tot) / TARGET <= 2.0,
         f"{act}/{tot} = {act / tot:.5%}, {(act / tot) / TARGET:.2f}x")
    gate("G0b (<10% byte-identical to the opponent)", same / len(shared) < 0.10,
         f"{same}/{len(shared)} = {same / len(shared):.1%}")

    # ---- G1: the corpus is what the registration names ---------------------------------------
    corpus = [json.loads(l) for l in open(f"{s['corpus']}/factscore.jsonl", encoding="utf-8")]
    books = {r["source_novel"] for r in corpus}
    words = statistics.fmean(len(r["prompt_text"].split()) for r in corpus)
    served = [json.loads(l)["aggregate"]["full_text"]
              for l in open(f"{root}/conc_bind/trajectories_k{bind_k:g}_factual.jsonl",
                            encoding="utf-8")]
    hdr = sum(1 for t in served if t.startswith(HEADER))
    ok = len(corpus) == s["n_prompts"] and len(served) == s["n_prompts"]
    what = [f"{len(corpus)} prompts, {len(served)} served"]
    if s.get("novel_is"):
        ok = ok and corpus[0]["source_novel"] == s["novel_is"]
        what.append(f"first source_novel={corpus[0]['source_novel']!r}")
    if s.get("min_books"):
        ok = ok and len(books) >= s["min_books"]
        what.append(f"{len(books)} books")
    if s.get("prompt_words"):
        ok = ok and abs(words - s["prompt_words"]) / s["prompt_words"] < 0.10
        what.append(f"{words:.1f} words")
    if s.get("no_header"):
        ok = ok and hdr == 0
        what.append(f"{hdr} carry the header")
    gate("G1 (the corpus is what the registration names)", ok, ", ".join(what))

    # ---- G2: the anchor is not degenerate here ------------------------------------------------
    if s.get("max_empty"):
        cands = load_candidates(f"{root}/sel_anchor64")
        n1 = [c[0][3] for c in cands.values() if c]
        empty = sum(1 for t in n1 if not t.strip()) / len(n1)
        gate(f"G2 (<{s['max_empty']:.0%} empty completions at n=1)", empty < s["max_empty"],
             f"{empty:.1%} empty over {len(n1)} prompts")

    # ---- G3: the corpus really is unfamiliar to the anchor, MEASURED --------------------------
    if s.get("leak_csv"):
        r = [x for x in csv.DictReader(open(s["leak_csv"], encoding="utf-8"))
             if x["n"] == "1"]
        assert len(r) == 1, s["leak_csv"]
        assert int(r[0]["n_passages"]) == s["n_prompts"], "the probe must cover the whole corpus"
        nv = float(r[0]["nv_recall_mean"])
        gate("G3 (the anchor reads 0.000 near-verbatim recall here)", nv == 0.0,
             f"nv-recall {nv:.4f} over the workload's own {s['n_prompts']} passages")

    print(f"{'gate':<62} {'reading':<6} detail")
    for name, rd, detail in gates:
        print(f"{name:<62} {rd:<6} {detail}")

    if fails:
        print(f"\n*** {len(fails)} GATE(S) FAILED: {fails}")
        print("*** NOT SCORED. The bands are not computed (caution (v)).")
        return

    print("\nAll gates pass; reading the bands.")
    rows = []
    for label, sfx, k in (("B1 binding", "conc_bind", bind_k), ("B2 vacuous", "conc_k10", 10.0)):
        g, lo, hi, rd = d3(tag, sfx)
        rows.append(dict(band=label, k=k, d3=g, lo95=lo, hi95=hi,
                         verdict=band(g, lo, hi), h2h_reading=rd))
        print(f"  {label:<12} k={k:<5} {g:+.4f} [{lo:+.4f}, {hi:+.4f}]  **{band(g, lo, hi)}**")

    out = os.path.join(a.out, s["out"])
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
