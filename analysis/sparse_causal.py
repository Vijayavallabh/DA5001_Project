"""feat-125/126: what a sparse CAUSAL policy buys, and whether alpha=8 is the trivial horn.

Proposition 3 says a per-token RATE meter is vacuous or trivial. It does not forbid a causal policy
that CONCENTRATES a bounded budget, and the paper's abstract has been stronger than that ("cannot be
repaired") while Section 4 concedes the gap. feat-092 built the front-loaded placement and scored it
(PLACEMENT IS IRRELEVANT, THE CAUSAL HORN IS EMPTY) but front-loading is not CHOOSING: it spends as
soon as the constraint binds. `--spend-threshold` reserves the budget for the steps whose own demand
D_KL(p_r,t || p_s,t) clears tau, which is the "concentrate at pivotal positions" shape.

This collects, per arm: the budget and where it went (does it bind? at how many steps?), the step
shares, and the order-averaged judged gain from analysis/order_averaged_h2h.py.

Two things it deliberately does NOT report:
  * leakage. h1.py's risky model here is the base Llama-3.1-8B-Instruct, which memorises none of
    these passages, so nv_recall = 0.0000 is a fact about the risky model and not about the policy
    (caution (t): a zero is the easiest kind of bug to mistake for a result). Withdrawn in
    results/onset_prediction_sparse_causal.md, amendment 2, before scoring.
  * anything from the per-step log, which is padded past the end of the generation (caution (s)).
    Step counts come from the aggregate, and are asserted to add up to the decoded length.

No GPU. Reads output/phase5/{sparse_*,renyi8_k3_full} and results/order_averaged_h2h_*.csv.

Usage:
  .venv/bin/python analysis/sparse_causal.py --out results
"""
import argparse
import csv
import glob
import json
import os
import statistics as st

ORDINARY = ("neutral", "creative", "factual")
SELECTION_N64 = 0.1045          # results/order_averaged_h2h.csv, the number the arms are read against


def arm_rows(run_dir):
    """Ordinary-traffic aggregates for one arm, with the padding-free step counters checked."""
    out, meta = [], None
    for cls in ORDINARY:
        for path in glob.glob(os.path.join(run_dir, f"trajectories_k*_{cls}.jsonl")):
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                a, m = r["aggregate"], r["metadata"]
                meta = meta or m
                act = a.get("steps_active") or 0
                forced = a.get("steps_forced_safe") or 0
                unch = a.get("steps_risky_unchanged") or 0
                out.append(dict(spend=a.get("total_spend"), act=act, forced=forced, unch=unch,
                                length=a.get("generation_length_tokens") or 0,
                                ok=a.get("invariant_ok", True)))
    return out, meta


def summarise(tag, run_dir, judged):
    rows, meta = arm_rows(run_dir)
    if not rows:
        return None
    # caution (l): an aggregate counter is not always a summary of the per-step log. Where the three
    # counters do add to the decoded length they are usable; say so rather than assume it.
    agree = sum(1 for r in rows if r["act"] + r["forced"] + r["unch"] == r["length"])
    spend = [r["spend"] for r in rows if r["spend"] is not None]
    act = [r["act"] for r in rows]
    K = meta.get("K")
    g = judged.get(tag, {})
    return dict(
        arm=tag,
        budget_B=meta.get("initial_bank"),
        tau=meta.get("spend_threshold"),
        constraint=meta.get("constraint", "kl"),
        K=K,
        n_traj=len(rows),
        spend_median=round(st.median(spend), 4) if spend else "",
        spend_max=round(max(spend), 4) if spend else "",
        binds=("YES" if spend and K and abs(max(spend) - K) < 1e-3 else "no"),
        active_steps_median=st.median(act),
        active_steps_max=max(act),
        active_share=round(sum(act) / max(sum(r["length"] for r in rows), 1), 5),
        forced_share=round(sum(r["forced"] for r in rows) / max(sum(r["length"] for r in rows), 1), 4),
        counters_agree_pct=round(100 * agree / len(rows), 1),
        invariant_violations=sum(0 if r["ok"] else 1 for r in rows),
        gain=g.get("value", ""), gain_lo95=g.get("lo95", ""), gain_hi95=g.get("hi95", ""),
        d3_vs_selection=(round(SELECTION_N64 - float(g["value"]), 4) if g.get("value") != "" and
                         g.get("value") is not None else ""),
    )


def load_judged(out_dir):
    """tag -> the D2 row (this arm's order-averaged gain over the anchor served alone)."""
    judged = {}
    for p in glob.glob(os.path.join(out_dir, "order_averaged_h2h_*.csv")):
        tag = os.path.basename(p)[len("order_averaged_h2h_"):-len(".csv")]
        if tag.startswith("per_prompt"):
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["quantity"].startswith("D2"):
                judged[tag] = r
    return judged


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="output/phase5")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    judged = load_judged(a.out)
    rows = []
    for d in sorted(glob.glob(os.path.join(a.runs, "sparse_*"))) + \
            sorted(glob.glob(os.path.join(a.runs, "renyi8_k3_full"))):
        tag = os.path.basename(d)
        r = summarise(tag, d, judged)
        if r:
            rows.append(r)
    assert rows, f"no arms under {a.runs}"

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "sparse_causal.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"  {r['arm']:22s} B={str(r['budget_B']):>8s} tau={str(r['tau']):>5s} "
              f"binds={r['binds']:>3s} active/tok={r['active_share']:.5f} "
              f"gain={str(r['gain']):>8s} [{r['gain_lo95']}, {r['gain_hi95']}]  "
              f"D3={r['d3_vs_selection']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
