"""feat-198 (results/onset_prediction_scorer_family.md): what a scorer from another family serves. No GPU.

Reads the committed Qwen reward cache and the gemma-2-27b-it cache over the same 32,000 draws of the
headline pool and writes results/scorer_family.csv: G1 (on how many prompts the served draw changes)
and, descriptively, how long and how often empty each scorer's served draws are. Picks are the argmax
at n = 64 with ties broken by index, as selection serves them; texts are the recovered served text
(load_candidates(..., deecho=True)), which is what the judge read.

Usage: .venv/bin/python analysis/scorer_family.py --out results
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_decoding import load_candidates  # noqa: E402
from analysis.selection_scaling import load_rewards  # noqa: E402


def pick(r, n=64):
    r = r[:n]
    return max(range(len(r)), key=lambda i: (r[i], -i))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel-dir", default="output/phase5/sel_anchor64")
    ap.add_argument("--committed", default="results/selection_rewards64.csv")
    ap.add_argument("--other", default="results/selection_rewards64_gemma27b.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    q, g = load_rewards(a.committed), load_rewards(a.other)
    cands = load_candidates(a.sel_dir, deecho=True)
    pids = sorted(set(q) & set(g))
    assert len(pids) == 500, len(pids)
    rows = [dict(scorer="both", quantity="prompts whose served draw changes", value=sum(
        pick(q[p]) != pick(g[p]) for p in pids), n=len(pids))]
    for name, r in (("Qwen2.5-7B-Instruct", q), ("gemma-2-27b-it", g)):
        texts = [cands[p][pick(r[p])][3] for p in pids]
        words = sorted(len(t.split()) for t in texts)
        rows.append(dict(scorer=name, quantity="empty served texts", value=sum(not t.strip() for t in texts),
                         n=len(pids)))
        rows.append(dict(scorer=name, quantity="median words served", value=words[len(words) // 2], n=len(pids)))
    path = os.path.join(a.out, "scorer_family.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print("wrote", path)


if __name__ == "__main__":
    main()
