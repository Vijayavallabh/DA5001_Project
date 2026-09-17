"""Measured wall-clock price of the two serving paths, against the analytical model that replaced it.

`analysis/serving_cost.py` prices selection at n=64 as 61.3x the metered decoder, from parameter
counts times token counts: (P_s + P_r)(L_p + T) with P_s = 1.7586 and P_r = 7.6156 billion. That is
a FLOP proxy, and it sits in the abstract where a deployer reads a price. A reviewer objected that
batched n-sampling amortises heavily and that GPU-seconds per served token is the honest unit.

This parses the timings that scripts/run_serving_latency.sh wrote and divides by the SERVED tokens
(one completion per prompt in both paths), using `generation_length_tokens` -- the decoded length,
so it is immune to the per_step_log padding trap of caution (s).

Two findings the analytical model gets wrong, both in results/onset_prediction_serving_latency.md:
the measured ratio is 35.4x rather than 61.3x, and the reward model is 9.3% of selection's wall-clock
rather than the majority -- scoring 64 candidates is one batched forward pass while drawing them is
64 x 204 sequential decode steps.

No GPU. Reads output/logs/serving_latency.log and the trajectory directories it names.

Usage:
  .venv/bin/python analysis/serving_latency.py --out results
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics as st

ANALYTICAL = 61.29          # analysis/serving_cost.py, the figure the paper quotes


def served_tokens(d):
    """Tokens actually delivered: one completion per prompt, whatever n was drawn."""
    per = {}
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            per.setdefault(r["metadata"]["prompt_id"], []).append(
                r["aggregate"]["generation_length_tokens"])
    return sum(v[0] for v in per.values()), len(per)


def parse(log):
    sel, met = [], []
    for line in open(log, encoding="utf-8"):
        m = re.search(r"\[lat\] SEL rep=(\d+) draws_s=([\d.]+) reward_s=([\d.]+) dir=(\S+)", line)
        if m:
            sel.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4)))
        m = re.search(r"\[lat\] MET rep=(\d+) decode_s=([\d.]+) dir=(\S+)", line)
        if m:
            met.append((int(m.group(1)), float(m.group(2)), m.group(3)))
    return sel, met


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default="output/logs/serving_latency.log")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    sel, met = parse(a.log)
    assert sel and met, f"no timings in {a.log}"
    tok_sel, n_sel = served_tokens(sel[0][3])
    tok_met, n_met = served_tokens(met[0][2])
    assert tok_sel == tok_met, (
        f"denominators differ ({tok_sel} vs {tok_met}); the ratio is then not the wall-clock ratio "
        "and the registration's construction has been broken")

    draws = st.mean(s[1] for s in sel)
    reward = st.mean(s[2] for s in sel)
    total = draws + reward
    m = st.mean(x[1] for x in met)
    rows = [
        dict(quantity="selection n=64, anchor draws", seconds=round(draws, 3),
             per_served_token=round(draws / tok_sel, 6), reps=" ".join(f"{s[1]:.1f}" for s in sel)),
        dict(quantity="selection n=64, reward pass", seconds=round(reward, 3),
             per_served_token=round(reward / tok_sel, 6), reps=" ".join(f"{s[2]:.1f}" for s in sel)),
        dict(quantity="selection n=64, total", seconds=round(total, 3),
             per_served_token=round(total / tok_sel, 6),
             reps=" ".join(f"{s[1] + s[2]:.1f}" for s in sel)),
        dict(quantity="metered k=10", seconds=round(m, 3),
             per_served_token=round(m / tok_met, 6), reps=" ".join(f"{x[1]:.1f}" for x in met)),
        dict(quantity="ratio, measured", seconds="", per_served_token=round(total / m, 3), reps=""),
        dict(quantity="ratio, analytical (serving_cost.py)", seconds="",
             per_served_token=ANALYTICAL, reps=""),
        dict(quantity="reward share of selection wall-clock", seconds="",
             per_served_token=round(reward / total, 4), reps=""),
        dict(quantity="ratio if the scorer were free", seconds="",
             per_served_token=round(draws / m, 3), reps=""),
        dict(quantity="served tokens (both arms)", seconds="", per_served_token=tok_sel,
             reps=f"{n_sel} prompts"),
    ]
    path = os.path.join(a.out, "serving_latency.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"  {r['quantity']:42s} {str(r['seconds']):>10s} s   {r['per_served_token']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
