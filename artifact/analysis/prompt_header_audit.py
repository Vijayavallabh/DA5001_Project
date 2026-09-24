"""Is the workload split the prompt TEMPLATE rather than the workload?

A reviewer's obvious objection to the scoping result: our own corpus and every external benchmark
do not get the same prompt. `dap/shared.py` prepends `Complete the prefix:\\n` to copyright-domain
prompts, and AlpacaEval, MT-Bench and CoTaEval-QA are deliberately routed through the **factual**
slot so that their numbers stay their own. So "ours versus theirs" has also, all along, been
"header versus no header", and nothing in the paper had separated them.

It separates itself, on data already on disk and with no new compute. Our own `850`-prompt workload
is not uniform: its `neutral` and `creative` classes carry the header and its `factual` class --- a
majority of it, `500` prompts --- does not, because it goes through the same slot AlpacaEval does.
feat-173's per-class decomposition therefore already contains a header-free arm of our own corpus.

This reads the header off what the models were actually SERVED (`aggregate.full_text`), never off a
metadata field: `metadata.prompt_text` does not exist in these records, so a check written against
it returns `""` and reports `0/500` for every class --- a false negative that looks exactly like a
result. Caution (au): read the generation, not the field you expected to be there.

Usage:
  .venv/bin/python analysis/prompt_header_audit.py --out results
"""
import argparse
import csv
import json
import os

HEADER = "Complete the prefix:"

# (label, trajectories file, the per-class d3 row in results/workload_predictor.csv)
ARMS = [
    ("ours: neutral",  "output/wscope/a_conc/trajectories_k10_neutral.jsonl",  "ours: neutral"),
    ("ours: creative", "output/wscope/a_conc/trajectories_k10_creative.jsonl", "ours: creative"),
    ("ours: factual",  "output/wscope/a_conc/trajectories_k10_factual.jsonl",  "ours: factual"),
    ("AlpacaEval",     "output/mixpow/conc_all/trajectories_k10_factual.jsonl", "AlpacaEval"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    pred = {r["workload"]: r for r in csv.DictReader(
        open("results/workload_predictor.csv", encoding="utf-8"))}

    rows = []
    for label, path, key in ARMS:
        assert os.path.exists(path), f"{path} missing; run this where the generations are"
        n = hdr = 0
        for line in open(path, encoding="utf-8"):
            t = json.loads(line)["aggregate"].get("full_text") or ""
            n += 1
            hdr += t.startswith(HEADER)
        assert n, path
        p = pred.get(key)
        assert p, f"{key} is not in workload_predictor.csv"
        rows.append(dict(arm=label, n=n, n_with_header=hdr,
                         header=round(hdr / n, 4),
                         d3=p["d3"], lo95=p["lo95"], hi95=p["hi95"], winner=p["winner"]))

    out = os.path.join(a.out, "prompt_header_audit.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{'arm':<16} {'header':>8} {'d3':>9} {'lo':>9} {'hi':>9}  winner")
    for r in rows:
        print(f"{r['arm']:<16} {r['header']:>7.1%} {float(r['d3']):>+9.4f} "
              f"{float(r['lo95']):>+9.4f} {float(r['hi95']):>+9.4f}  {r['winner']}")

    free = [r for r in rows if r["header"] == 0.0]
    winners = {r["winner"] for r in free}
    print(f"\n{len(free)} header-free arms, winners {sorted(winners)}")
    print("HEADER EXCLUDED: two header-free arms disagree in sign with intervals excluding zero"
          if len(winners) > 1 else
          "HEADER NOT EXCLUDED: every header-free arm reads the same way")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
