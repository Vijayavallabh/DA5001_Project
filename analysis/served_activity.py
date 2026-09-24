"""Table 3's binding column, recomputed on exactly the trajectories the judge scored (no GPU).

A referee (2026-09-24, sixth round) found Table 3's "binds" (5.5% at k=1) disagreeing with Appendix G's
beta (0.2057 at k=1). Both were right about different things: Table 3's column came from
analysis/budget_calibration.py, the aggregate counters pooled over every class of output/sweep_plain,
protected passages included (and caution (l) says those counters are unreliable in that run), while
beta counts every step the bucket constrains, strict blends plus tokens the anchor writes outright,
averaged per trajectory. This fixes one definition and one population for the table:

  active  = decode steps whose served token comes from a STRICT blend of the two models
            (0 < bd < 1 in the per-step log), over all decode steps, end-of-text padding removed
            (dap.stats.strip_pad_steps), pooled over
  the judged trajectories = the lowest-seed trajectory of each of the 500 ordinary prompts, which is
            what analysis/order_averaged_h2h.py serves to the judge (lowest_seed).

Usage: .venv/bin/python analysis/served_activity.py --out results
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dap.stats import strip_pad_steps  # noqa: E402

CLASSES = ("neutral", "factual", "creative")
ROWS = [  # (block, k, directory, filename token) -- Table 3's token-level rows
    ("8B-Instruct, continuing text", 0.5, "output/sweep_plain", "0.5"),
    ("8B-Instruct, continuing text", 1.0, "output/sweep_plain", "1"),
    ("8B-Instruct, continuing text", 10.0, "output/phase2/conc_all", "10"),
    ("70B base", 0.5, "output/phase5/imit_llama70b", "0.5"),
    ("70B base", 1.0, "output/phase5/imit_llama70b", "1"),
    ("70B base", 20.0, "output/phase5/imit_llama70b", "20"),
    ("8B-Instruct, chat template", 1.0, "output/sweep_chat", "1"),
    ("8B-Instruct, chat template", 10.0, "output/feat184/chat_k10", "10"),
    # feat-196: the chat block's remaining budgets (k=0.5 is the committed sweep's own arm)
    ("8B-Instruct, chat template", 0.5, "output/sweep_chat", "0.5"),
    ("8B-Instruct, chat template", 2.0, "output/feat196/chat_grid", "2"),
    ("8B-Instruct, chat template", 3.0, "output/feat196/chat_grid", "3"),
    ("8B-Instruct, chat template", 5.0, "output/feat196/chat_grid", "5"),
]


def judged_logs(d, tok):
    """prompt_id -> per-step log of its lowest-seed trajectory, over the ordinary classes."""
    best = {}
    for cls in CLASSES:
        for line in open(os.path.join(d, f"trajectories_k{tok}_{cls}.jsonl")):
            r = json.loads(line)
            m = r["metadata"]
            if m.get("constraint", "kl") != "kl":
                continue
            p, s = m["prompt_id"], m["seed"]
            if p not in best or s < best[p][0]:
                best[p] = (s, r["per_step_log"])
    return {p: v[1] for p, v in best.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = []
    for block, k, d, tok in ROWS:
        logs = judged_logs(d, tok)
        assert len(logs) == 500, (d, tok, len(logs))
        act = frc = tot = 0
        for log in logs.values():
            log = strip_pad_steps(log)
            act += sum(1 for e in log if 1e-6 < e.get("bd", 0.0) < 1 - 1e-6)
            frc += sum(1 for e in log if e.get("bd", 0.0) <= 1e-6)
            tot += len(log)
        out.append(dict(block=block, k=k, run_dir=d, prompts=len(logs), decode_steps=tot,
                        active_share=round(act / tot, 6), forced_share=round(frc / tot, 6)))
    path = os.path.join(a.out, "served_activity.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(r)
    print("wrote", path)


if __name__ == "__main__":
    main()
