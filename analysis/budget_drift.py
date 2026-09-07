"""Plan v4 / feat-037: the price of a budgeted decoder is not a property of the two models.

The frontier theorem fixes two of its three boundaries from the safe model alone: the certificate
goes vacuous at the surprisal rate s(x), and reproduction becomes possible at the Lindley running
maximum k_crit(x). The third boundary -- the budget below which utility degrades -- looks like it
should follow from the divergence between the risky and safe models on ordinary traffic. It does
not, and this script measures why.

Two facts, both from logs the decoder already wrote, no GPU:

  1. On the risky model's own rollout (k=-1, theta=1 everywhere) the per-step divergence
     KL(p_r || p_s) averages 4-11 nats depending on the prompt class. On a budgeted rollout the
     steps where the risky model is served UNCHANGED cost about 0.7-0.8 nats. The decoder is
     paying 5-14x less per unaltered step than the model's own trajectory implies.

  2. That gap is not explained by the decoder selecting cheap steps. Compare what it pays against
     the mean of the cheapest F-fraction of the unconstrained distribution, where F is the
     unaltered fraction it actually achieved: the ratio is 0.9-10.9, i.e. mostly it pays MORE than
     rank selection predicts, because banking lets it save for an expensive step rather than
     always taking the cheapest one.

So the realised price is a property of the rollout the budget induces, not of the model pair. Any
attempt to derive the utility boundary from a static comparison of p_r and p_s -- the only kind of
comparison a deployer can make before serving -- is measuring the wrong process. This is the
reason the utility boundary in the frontier theorem is reported as measured rather than proved.

Writes results/budget_drift.csv.
Usage: .venv/bin/python analysis/budget_drift.py --run output/sweep_chat --out results
"""
import argparse, csv, json, os, statistics as st


def per_step(path):
    """(a_t, theta) for every logged step of one trajectory file."""
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        for s in json.loads(line)["per_step_log"]:
            if s.get("a_t") is not None:
                out.append((s["a_t"], s.get("bd")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="output/sweep_chat")
    ap.add_argument("--classes", nargs="+",
                    default=["neutral", "factual", "creative", "attack_train", "test", "val"])
    ap.add_argument("--budgets", nargs="+", default=["0.5", "1"])
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    rows = []
    for cls in a.classes:
        base = sorted(v for v, _ in per_step(os.path.join(a.run, f"trajectories_k-1_{cls}.jsonl")))
        if not base:
            continue
        n = len(base)
        for k in a.budgets:
            steps = per_step(os.path.join(a.run, f"trajectories_k{k}_{cls}.jsonl"))
            free = [v for v, bd in steps if bd is not None and bd > 0.999]
            if not steps or not free:
                continue
            F = len(free) / len(steps)
            paid = st.mean(free)
            selection_only = st.mean(base[: max(1, int(F * n))])
            rows.append({
                "prompt_class": cls, "k": float(k), "n_steps": len(steps),
                "unconstrained_mean_nats": st.mean(base),
                "unconstrained_median_nats": st.median(base),
                "unaltered_fraction": F,
                "paid_per_unaltered_step": paid,
                "selection_only_prediction": selection_only,
                "price_gap_vs_unconstrained": st.mean(base) / paid,
                "selection_ratio": paid / selection_only,
            })

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "budget_drift.csv")
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print(f"{'class':13s} {'k':>4s} {'uncon.mean':>11s} {'paid|free':>10s} {'gap':>6s} {'sel.ratio':>10s}")
    for r in rows:
        print(f"{r['prompt_class']:13s} {r['k']:4.1f} {r['unconstrained_mean_nats']:11.3f} "
              f"{r['paid_per_unaltered_step']:10.4f} {r['price_gap_vs_unconstrained']:6.1f}x "
              f"{r['selection_ratio']:10.2f}")
    print(f"\nwrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
