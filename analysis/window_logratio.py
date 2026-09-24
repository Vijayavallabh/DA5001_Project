"""What a windowed pathwise meter would have to spend to serve the risky model on ordinary text.

A referee (2026-09-24, sixth round) proposes a per-token design the paper's dichotomy does not cover:
cap the per-step max-divergence by the budget left in a sliding window and deduct only the REALISED
log-ratio log p_r(y_t) - log p_s(y_t). Every w-token window is then certified at the window's budget,
and whether that budget binds on ordinary traffic is an empirical question. This measures it.

On the risky model's own draws (output/sweep_plain, k=-1, the three ordinary classes, every seed) the
per-step log records p_r and p_s of the token actually served, so the realised log-ratio of every
50-token window is a sum over the log. Also reported: the per-step KL D(p_r,t || p_s,t) summed over
the same windows (a_t in the log), which is what serving p_r costs in expectation, and each
trajectory's largest window, which is the budget a windowed meter would need never to bind on it.

No GPU. Steps after the first end-of-text token are dropped (the harness pads to T_max; caution (s)).

Usage: .venv/bin/python analysis/window_logratio.py --out results
"""
import argparse
import csv
import json
import math
import os

EOS = {128001, 128009}
CLASSES = ("neutral", "factual", "creative")


def windows(xs, w):
    """Sums of every contiguous length-w window of xs (empty if len(xs) < w)."""
    if len(xs) < w:
        return []
    s = sum(xs[:w])
    out = [s]
    for i in range(w, len(xs)):
        s += xs[i] - xs[i - w]
        out.append(s)
    return out


def quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def trajectory_steps(rec):
    """(realised log-ratio, per-step KL) for each decode step before the first EOS."""
    out = []
    for st in rec["per_step_log"]:
        ps, pr = st.get("p_s_prob"), st.get("p_risky_prob")
        if ps is None or pr is None or ps <= 0 or pr <= 0:
            raise ValueError(f"step without both probabilities: {st}")
        out.append((math.log(pr) - math.log(ps), float(st["a_t"])))
        if st.get("sampled_token_id") in EOS:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="output/sweep_plain")
    ap.add_argument("--w", type=int, default=50)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    ratio_w, kl_w, traj_max, n_traj, n_short, steps = [], [], [], 0, 0, 0
    for cls in CLASSES:
        for line in open(os.path.join(a.run_dir, f"trajectories_k-1_{cls}.jsonl")):
            r = json.loads(line)
            st = trajectory_steps(r)
            n_traj += 1
            steps += len(st)
            rw = windows([x for x, _ in st], a.w)
            if not rw:
                n_short += 1
                continue
            ratio_w += rw
            kl_w += windows([y for _, y in st], a.w)
            traj_max.append(max(rw))
    s_w = float({r["event"]: r for r in csv.DictReader(open(os.path.join(a.out, "window_vacuity.csv")))}
                [f"{a.w}-token window"]["S_median_nats"])
    rows = []
    for name, xs in (("realised log-ratio per window", ratio_w), ("per-step KL summed per window", kl_w),
                     ("largest realised window per trajectory", traj_max)):
        rows.append(dict(quantity=name, w=a.w, n=len(xs), median=round(quantile(xs, 0.5), 4),
                         p90=round(quantile(xs, 0.9), 4), p99=round(quantile(xs, 0.99), 4),
                         max=round(max(xs), 4), frac_ge_S_w=round(sum(x >= s_w for x in xs) / len(xs), 6),
                         S_w=s_w))
    rows.append(dict(quantity="trajectories (shorter than w, excluded)", w=a.w, n=n_traj,
                     median=n_short, p90="", p99="", max="", frac_ge_S_w="", S_w=s_w))
    rows.append(dict(quantity="decode steps before end-of-text", w=a.w, n=steps, median="", p90="",
                     p99="", max="", frac_ge_S_w="", S_w=s_w))
    path = os.path.join(a.out, "window_logratio.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for r in rows:
        print(r)
    print("wrote", path)


if __name__ == "__main__":
    main()
