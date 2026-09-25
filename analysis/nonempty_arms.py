"""feat-209 (results/onset_prediction_nonempty_tables.md): the non-empty rule as a servable arm. No GPU.

For each selection pool, serve the highest-reward draw whose recovered text is non-empty (ties to the
lowest index; if every draw is empty, the committed pick), exactly analysis/nonempty_rule.py's `alt`,
and write it as trajectories_knonempty_<class>.jsonl so analysis/order_averaged_h2h.py --extra-dir can
judge it beside the committed pick in the same pass. The rule is a selection rule over the same n draws,
so Proposition 1 certifies it at log n unchanged.

Usage: .venv/bin/python analysis/nonempty_arms.py --out output/feat209
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_decoding import load_candidates  # noqa: E402
from analysis.selection_scaling import load_rewards  # noqa: E402

POOLS = {"pool": ("output/phase5/sel_anchor64", "results/selection_rewards64.csv"),
         "t07": ("output/feat195/t07_pool64", "results/selection_rewards64_t07.csv"),
         "comma7b": ("output/phase5/sel_comma7b_64", "results/selection_rewards64_comma7b.csv")}
CLASSES = ("neutral", "factual", "creative")


def picks(rewards, texts):
    """(committed, non-empty) rank for one prompt."""
    order = sorted(range(len(rewards)), key=lambda i: (-rewards[i], i))
    alt = next((i for i in order if (texts[i] or "").strip()), order[0])
    return order[0], alt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/feat209")
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    summary = []
    for tag, (d, rw) in POOLS.items():
        cands, rewards = load_candidates(d, deecho=True), load_rewards(rw)
        prefix = {}
        for cls in CLASSES:
            for line in open(os.path.join(d, f"trajectories_k0_{cls}.jsonl"), encoding="utf-8"):
                r = json.loads(line)
                prefix.setdefault(r["metadata"]["prompt_id"], (cls, r["prefix_analysis"]["prefix_text"]))
        pids = sorted(p for p in rewards if len(rewards[p]) >= a.n)
        assert len(pids) == 500, (tag, len(pids))
        by = {c: [] for c in CLASSES}
        changed = 0
        for p in pids:
            texts = [c[3] for c in cands[p][:a.n]]
            pick, alt = picks(rewards[p][:a.n], texts)
            if alt != pick:
                assert not texts[pick].strip()
                changed += 1
            cls, pt = prefix[p]
            gen = texts[alt]
            by[cls].append(dict(metadata=dict(prompt_id=p, seed=0, trajectory_id=0, k="nonempty", split=cls,
                                              rule="highest-reward non-empty draw", n=a.n, rank=alt, pool=d),
                                prefix_analysis=dict(prefix_text=pt),
                                aggregate=dict(generation=gen, full_text=pt + gen)))
        os.makedirs(os.path.join(a.out, tag), exist_ok=True)
        for cls, recs in by.items():
            with open(os.path.join(a.out, tag, f"trajectories_knonempty_{cls}.jsonl"), "w", encoding="utf-8") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
        summary.append(dict(pool=tag, prompts=len(pids), picks_changed=changed))
        print(f"[nonempty] {tag}: {changed} of {len(pids)} picks change", flush=True)
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)


if __name__ == "__main__":
    main()
