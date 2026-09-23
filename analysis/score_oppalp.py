"""feat-178 L: score the opponent ladder ON ALPACAEVAL. Gates first, then bands.

Bands and gates: results/onset_prediction_opponent_by_workload.md, committed before any of these
generations existed. The script REFUSES to print a band if a gate fails (caution (v)).

The question is whether feat-175's dose-response -- selection's advantage over a rate-matched
metered decoder falls as the fixed opponent gets stronger -- is a property of opponents or a
property of OUR workload. Only --baseline-dir differs from scripts/run_mixpow_judge_k.sh, which
produced the committed AlpacaEval reading, and unlike the committed ladder every rung here comes
from one generator, so the slope is measured inside one pipeline (caution (at)).

Usage:
  .venv/bin/python analysis/score_oppalp.py --out results
"""
import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.opponent_by_workload import exact_p, spearman  # noqa: E402
from analysis.opponent_strength import d3_row, per_prompt, pipeline, shape  # noqa: E402

ARMS = [
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", "llama8b"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "qwen05b"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "qwen15b"),
    ("Qwen/Qwen2.5-3B-Instruct", "qwen3b"),
    ("Qwen/Qwen2.5-14B-Instruct", "qwen14b"),
]
N_PROMPTS = 805          # G1, the committed AlpacaEval pass's own prompt count
MAX_EMPTY = 0.10         # G2, registered
LEN_FACTOR = 2.0         # G2, registered -- TIGHTER than feat-175's 3.0, and deliberately so:
                         # these five rungs share one generator and one prompt set, so a spread
                         # that a cross-pipeline ladder had to tolerate would be a defect here.
MIN_SPAN = 0.10          # G3, registered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    gates, fails, rows = [], [], []

    def gate(name, ok, detail):
        gates.append((name, "PASS" if ok else "FAIL", detail))
        if not ok:
            fails.append(name)

    pids_by_arm = {}
    for model, tag in ARMS:
        sfx, run = f"__oppalp_{tag}", f"output/oppalp_{tag}"
        pp, d3 = per_prompt(sfx), d3_row(sfx)
        if pp is None or d3 is None:
            print(f"[skip] {tag}: not judged yet")
            continue
        pids = {r["prompt_id"] for r in pp}
        pids_by_arm[tag] = pids
        strength = 1 - statistics.fmean(float(r["u_anchor_k0"]) for r in pp)
        n, empty, words = shape(run, pids)
        rows.append(dict(opponent=model, tag=tag, generator=pipeline(run), n_judged=len(pp),
                         n_served=n, empty_frac=round(empty, 6), mean_words=round(words, 4),
                         strength=round(strength, 6), d3=float(d3["value"]),
                         d3_lo95=float(d3["lo95"]), d3_hi95=float(d3["hi95"]),
                         verdict=d3["reading"].strip()))

    if len(rows) < len(ARMS):
        print(f"\n{len(rows)}/{len(ARMS)} rungs judged; NOT SCORED until the ladder is complete.")
        return

    # ---- G0: one pipeline -----------------------------------------------------------------
    gens = {r["generator"] for r in rows}
    gate("G0 (every rung from one generator)", gens == {"blocklist_decode"},
         f"generators: {sorted(gens)}")

    # ---- G1: one prompt set ---------------------------------------------------------------
    sets = list(pids_by_arm.values())
    same = all(s == sets[0] for s in sets)
    gate("G1 (one prompt set, the committed pass's)",
         same and len(sets[0]) == N_PROMPTS,
         f"{len(sets[0])} prompts, identical across rungs: {same}")

    # ---- G2: no degenerate opponent -------------------------------------------------------
    med = statistics.median(r["mean_words"] for r in rows)
    bad = [r["tag"] for r in rows
           if r["empty_frac"] >= MAX_EMPTY
           or not (med / LEN_FACTOR <= r["mean_words"] <= med * LEN_FACTOR)]
    gate("G2 (no degenerate opponent)", not bad,
         f"median {med:.1f} words; offenders: {bad or 'none'}")

    # ---- G3: the instrument has range HERE -------------------------------------------------
    span = max(r["strength"] for r in rows) - min(r["strength"] for r in rows)
    gate(f"G3 (strengths span >= {MIN_SPAN})", span >= MIN_SPAN,
         f"{min(r['strength'] for r in rows):.4f} to "
         f"{max(r['strength'] for r in rows):.4f}, span {span:.4f}")

    print(f"{'gate':<46} {'reading':<6} detail")
    for name, rd, detail in gates:
        print(f"{name:<46} {rd:<6} {detail}")

    rows.sort(key=lambda r: r["strength"])
    print(f"\n{'opponent':<40} {'strength':>9} {'D3':>9} {'lo95':>9} {'hi95':>9}  verdict")
    for r in rows:
        print(f"{r['opponent']:<40} {r['strength']:>9.4f} {r['d3']:>+9.4f} "
              f"{r['d3_lo95']:>+9.4f} {r['d3_hi95']:>+9.4f}  {r['verdict']}")

    if fails:
        # G3 is the one gate whose failure has its own registered READING rather than a refusal:
        # an instrument with no range did not test the question (feat-175's H1 outcome).
        if fails == [g for g in fails if g.startswith("G3")]:
            print(f"\nL2 **NOT TESTED** -- the ladder has no range on this workload ({fails}).")
        else:
            print(f"\n*** {len(fails)} GATE(S) FAILED: {fails}")
            print("*** NOT SCORED. The bands are not computed (caution (v)).")
        return

    xs = [r["strength"] for r in rows]
    rho, p, n = exact_p(xs, [r["d3"] for r in rows])
    l1 = "CONSISTENT" if rho <= -0.7 else ("REFUTED" if rho >= 0.3 else "UNRESOLVED")
    crossing = [r for r in rows if r["d3"] > 0 and r["d3_lo95"] > 0]
    l2 = "OPPONENT EXPLAINS THE SPLIT" if crossing else "TASK TYPE SURVIVES"
    print(f"\nL1  rho = {rho:+.3f}, exact p = {p:.4f} over {n} permutations   **{l1}**")
    print(f"L2  {len(crossing)} rung(s) cross zero from below"
          f"{': ' + ', '.join(r['tag'] for r in crossing) if crossing else ''}   **{l2}**")
    # L4, registered beside L1: D3 is a difference of win rates against one opponent, so it has
    # less room as the anchor's own win rate u = 1 - strength falls. Re-run L1 on D3 / min(u, 1 - u),
    # the same normalisation analysis/opponent_by_workload.py applies to P1. Added 2026-09-23, after
    # the ladder landed, because the scorer had left out a reading the registration asks for.
    head = [min(x, 1 - x) for x in xs]
    for r, h in zip(rows, head):
        r["d3_normalised"] = round(r["d3"] / h, 6)
    rho_n, p_n, _ = exact_p(xs, [r["d3_normalised"] for r in rows])
    l4 = "CONSISTENT" if rho_n <= -0.7 else ("REFUTED" if rho_n >= 0.3 else "UNRESOLVED")
    print(f"L4  D3/headroom rho = {rho_n:+.3f}, exact p = {p_n:.4f}   **{l4}** (normalised L1)")
    print(f"\nNo significance is claimed: at n={len(rows)} the smallest two-sided p is {2 / n:.4f}.")

    out = os.path.join(a.out, "oppalp_ladder.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["rho", "exact_p", "l1", "l2",
                                                                 "rho_normalised",
                                                                 "exact_p_normalised", "l4"])
        w.writeheader()
        for r in rows:
            w.writerow(dict(r, rho=round(rho, 6), exact_p=round(p, 6), l1=l1, l2=l2,
                            rho_normalised=round(rho_n, 6), exact_p_normalised=round(p_n, 6),
                            l4=l4))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
