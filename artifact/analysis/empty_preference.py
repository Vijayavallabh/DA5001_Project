"""How often reward selection serves an EMPTY completion, and what the judge makes of it. No GPU.

The committed pointwise reward (Qwen2.5-7B-Instruct, log p(Yes) - log p(No)) scores an empty draw
above a typical one, a property found only once the recording defect was repaired (caution (bc)):
with the prompt's tail still in `generation` an empty draw did not look empty. This reads, on the
headline pool with its committed cache and picks, the reward of empty and non-empty draws, the
served-empty rate at each n, and the order-averaged judge-B level of the prompts served empty.
Usage: .venv/bin/python analysis/empty_preference.py --out results
"""
import argparse
import csv
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", default="output/phase5/sel_anchor64")
    ap.add_argument("--rewards", default="results/selection_rewards64.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    from analysis.selection_decoding import load_candidates
    from analysis.selection_scaling import load_rewards
    c = load_candidates(a.gen_dir, deecho=True)
    r = load_rewards(a.rewards)
    pids = sorted(p for p in r if p in c)
    u = {}
    for x in "ab":   # frontier_levels' per-prompt levels: selection n = 1..64 on this pool
        for row in csv.DictReader(open(os.path.join(a.out, f"frontier_levels_per_prompt_{x}.csv"))):
            for k, v in row.items():
                if k.startswith("u_sel_n"):
                    u.setdefault(int(k[7:]), {})[row["prompt_id"]] = float(v)
    emp = [r[p][i] for p in pids for i in range(64) if not c[p][i][3].strip()]
    full = [r[p][i] for p in pids for i in range(64) if c[p][i][3].strip()]
    rows = [dict(quantity="draws empty", n=64, value=round(len(emp) / (len(emp) + len(full)), 4),
                 count=len(emp), of=len(emp) + len(full)),
            dict(quantity="median reward, empty draws", n=64, value=round(st.median(emp), 2), count=len(emp)),
            dict(quantity="median reward, non-empty draws", n=64, value=round(st.median(full), 2),
                 count=len(full))]
    for n in (1, 8, 64):
        served = {p: not c[p][max(range(n), key=lambda i: r[p][i])][3].strip() for p in pids}
        k = sum(served.values())
        le = [u[n][p] for p in pids if served[p]]
        ln = [u[n][p] for p in pids if not served[p]]
        rows += [dict(quantity="served empty", n=n, value=round(k / len(pids), 4), count=k, of=len(pids)),
                 dict(quantity="judged level, served empty", n=n, value=round(sum(le) / len(le), 4), count=len(le)),
                 dict(quantity="judged level, served non-empty", n=n, value=round(sum(ln) / len(ln), 4),
                      count=len(ln))]
    path = os.path.join(a.out, "empty_preference.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["quantity", "n", "value", "count", "of"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for x in rows:
        print(x)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
