"""feat-178 P1/L3: is the workload split the OPPONENT's strength?

Bands: results/onset_prediction_opponent_by_workload.md, committed before this ran.

Opponent strength is `1 - mean(u_anchor_k0)` -- how often the fixed opponent beats the anchor's own
n=1 completion on that workload's prompts. It is a judged LEVEL, and caution (ap) forbids setting
single-order judged levels from different passes against each other. Every number here is
ORDER-AVERAGED, which draws nothing from the presentation-order RNG and so is a deterministic
function of the text under a greedy judge; results/n128_order_averaged_note.md measured a
single-order gain moving +0.142 -> +0.076 on byte-identical text while the order-averaged reading
reproduced to four decimals.

Every arm is read at its RATE-MATCHED BINDING budget, which is the cell the workload comparisons
are primary on. Mixing a binding cell with a vacuous one would compare two different mechanisms.

L4 is the check that could explain the result away: D3 is a difference of win rates against the
same opponent, so it has less room when u_anchor_k0 is near 0 or 1. If the ordering does not
survive dividing by the available headroom, the unification claim is withdrawn.

Usage:
  .venv/bin/python analysis/opponent_by_workload.py --out results
"""
import argparse
import csv
import itertools
import os
import statistics

# (workload, task type, per-prompt tag at the RATE-MATCHED BINDING budget, opponent dir)
ARMS = [
    ("ours",        "completion",  "wscope_c",            "output/wscope/a_baseline"),
    ("AlpacaEval",  "instruction", "mixpowk_judgeB",      "output/mixpow/baseline"),
    ("MT-Bench",    "instruction", "mtb_conc_bind",       "output/mtb/baseline"),
    ("Gutenberg",   "completion",  "gutenberg_conc_bind", "output/gutenberg/baseline"),
    ("CoTaEval-QA", "comprehension", "cotaeval_qa_conc_bind", "output/cotaeval_qa/baseline"),
    ("unseenbooks", "completion",  "unseenbooks_conc_bind", "output/unseenbooks/baseline"),
]


def opponent(run_dir):
    """WHO the opponent is, read off the records rather than assumed from a directory name.

    The whole axis is that ONE opponent is strong on one workload and weak on another. If an arm
    were judged against a different model, or against the same model through a different
    generator, its "strength" would be a statement about the opponent instead of the workload and
    the correlation would mean nothing. So the identity is written into the CSV and asserted
    uniform -- caution (at), two arms compared must have come from the same pipeline, and the runs
    record enough to check it.

    It returns blanks off-host, where output/ does not exist; main() then refuses to overwrite a
    populated CSV with them, so the committed columns cannot be silently emptied (caution (ax)).
    """
    import glob
    import json
    # A workload's opponent cell caps every class but `factual` at 0, so most of these files exist
    # and are EMPTY. Taking the first glob match reads `attack_train` and dies on an empty line.
    for f in sorted(glob.glob(os.path.join(run_dir, "trajectories_k-1_*.jsonl"))):
        line = next((ln for ln in open(f, encoding="utf-8") if ln.strip()), "")
        if not line:
            continue
        m = json.loads(line)["metadata"]
        return (m.get("target_model") or "", m.get("anchor_model") or "",
                "blocklist_decode" if "blocklist_ngram" in m else "h1.py")
    return "", "", ""


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):                      # average ties, as any rank test must
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def exact_p(xs, ys):
    """Two-sided exact permutation p. At n=5 the smallest attainable is 2/120 = 0.0167, which is
    why the registration forbids any significance claim from this arm whatever rho reads."""
    obs = spearman(xs, ys)
    perms = list(itertools.permutations(range(len(ys))))
    hits = sum(1 for p in perms if abs(spearman(xs, [ys[i] for i in p])) >= abs(obs) - 1e-12)
    return obs, hits / len(perms), len(perms)


def d3(tag):
    p = f"results/order_averaged_h2h__{tag}.csv"
    r = [x for x in csv.DictReader(open(p, encoding="utf-8")) if x["quantity"].startswith("D3")]
    assert len(r) == 1, p
    return float(r[0]["value"]), float(r[0]["lo95"]), float(r[0]["hi95"]), r[0]["reading"].strip()


def d1d2(tag):
    """L5: selection's own gain and the meter's, from the same pass. The appendix has said for
    three workloads that "the meter is what the workload changes"; if opponent strength is the
    variable behind the split it should act through the meter, and that is a prediction the
    existing sentence makes about numbers nobody had correlated."""
    p = f"results/order_averaged_h2h__{tag}.csv"
    out = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        for prefix in ("D1", "D2"):
            if r["quantity"].startswith(prefix):
                out[prefix] = float(r["value"])
    assert set(out) == {"D1", "D2"}, p
    return out["D1"], out["D2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    rows = []
    for name, kind, tag, opp_dir in ARMS:
        pp = f"results/order_averaged_h2h_per_prompt__{tag}.csv"
        if not os.path.exists(pp):
            print(f"[skip] {name}: {pp} not written yet")
            continue
        u = [float(r["u_anchor_k0"]) for r in csv.DictReader(open(pp, encoding="utf-8"))]
        anchor_win = statistics.fmean(u)
        g, lo, hi, reading = d3(tag)
        gs, gm = d1d2(tag)
        head = min(anchor_win, 1 - anchor_win)     # L4: the room a difference of win rates has
        opp, anc, gen = opponent(opp_dir)
        rows.append(dict(workload=name, task_type=kind, tag=tag, n_prompts=len(u),
                         opponent=opp, anchor=anc, opponent_generator=gen,
                         anchor_win=round(anchor_win, 6),
                         opponent_strength=round(1 - anchor_win, 6),
                         d3=g, d3_lo95=lo, d3_hi95=hi, reading=reading, g_sel=gs, g_met=gm,
                         headroom=round(head, 6), d3_normalised=round(g / head, 6)))

    # THE AXIS IS ONE OPPONENT SEEN BY DIFFERENT WORKLOADS. Asserted, not assumed.
    seen = {(r["opponent"], r["anchor"], r["opponent_generator"]) for r in rows if r["opponent"]}
    assert len(seen) <= 1, (
        f"the arms do not share an opponent, an anchor and a generator: {sorted(seen)}. "
        "Opponent strength would then be a statement about the opponent, not the workload.")

    out = os.path.join(a.out, "opponent_by_workload.csv")
    if not seen and os.path.exists(out):
        prev = list(csv.DictReader(open(out, encoding="utf-8")))
        assert not any(p.get("opponent") for p in prev), (
            f"{out} records who the opponent was and this run cannot see output/ to confirm it; "
            "refusing to overwrite those columns with blanks. Run this where the arms live.")

    rows.sort(key=lambda r: r["opponent_strength"])
    xs = [r["opponent_strength"] for r in rows]
    rho, p, n = exact_p(xs, [r["d3"] for r in rows])
    rho_n, p_n, _ = exact_p(xs, [r["d3_normalised"] for r in rows])
    rho_s, p_s, _ = exact_p(xs, [r["g_sel"] for r in rows])
    rho_m, p_m, _ = exact_p(xs, [r["g_met"] for r in rows])

    def band(r):
        return "CONSISTENT" if r <= -0.7 else ("REFUTED" if r >= 0.3 else "UNRESOLVED")

    w = max(len(r["workload"]) for r in rows)
    print(f"{'workload':<{w}} {'task type':<14} {'n':>5} {'strength':>9} {'D3':>9} "
          f"{'lo95':>9} {'hi95':>9} {'D3/head':>9}")
    for r in rows:
        print(f"{r['workload']:<{w}} {r['task_type']:<14} {r['n_prompts']:>5} "
              f"{r['opponent_strength']:>9.4f} {r['d3']:>+9.4f} {r['d3_lo95']:>+9.4f} "
              f"{r['d3_hi95']:>+9.4f} {r['d3_normalised']:>+9.4f}")
    print(f"\nL3  strength vs D3          rho = {rho:+.3f}  exact p = {p:.4f} over {n} "
          f"permutations   **{band(rho)}**")
    print(f"L4  strength vs D3/headroom rho = {rho_n:+.3f}  exact p = {p_n:.4f}"
          f"                        **{band(rho_n)}**")
    l5 = ("THE METER AGAIN" if abs(rho_m) > abs(rho_s) + 0.2 else
          "SELECTION, NOT THE METER" if abs(rho_s) > abs(rho_m) + 0.2 else "UNRESOLVED")
    print(f"L5  strength vs g_sel        rho = {rho_s:+.3f}  exact p = {p_s:.4f}")
    print(f"L5  strength vs g_met        rho = {rho_m:+.3f}  exact p = {p_m:.4f}"
          f"                        **{l5}**")
    print("\nNo significance is claimed: at n=%d the smallest attainable two-sided p is %.4f."
          % (len(rows), 2 / n))

    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["rho", "exact_p", "verdict",
                                                                 "rho_normalised", "exact_p_norm",
                                                                 "verdict_normalised",
                                                                 "rho_gsel", "rho_gmet", "l5"])
        wr.writeheader()
        for r in rows:
            wr.writerow(dict(r, rho=round(rho, 6), exact_p=round(p, 6), verdict=band(rho),
                             rho_normalised=round(rho_n, 6), exact_p_norm=round(p_n, 6),
                             verdict_normalised=band(rho_n),
                             rho_gsel=round(rho_s, 6), rho_gmet=round(rho_m, 6), l5=l5))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
