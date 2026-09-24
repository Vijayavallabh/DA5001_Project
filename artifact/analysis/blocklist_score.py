"""feat-143: score the decode-time MemFree arms on the two leakage events.

Same metrics, same code path as analysis/selection_extraction.py -- nv_recall and rouge_l_score
from dap.stats, both against the DECODED target (caution (b)), so the numbers are comparable with
results/selection_extraction*.csv rather than merely similar to them.

H2 is the literal event the rule was designed for; H3 the non-literal one it was not.

Usage:
  .venv/bin/python analysis/blocklist_score.py --run output/memfree/protected --out results
"""
import argparse
import csv
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dap.stats import lcs_word, nv_recall, rouge_l_score  # noqa: E402


def load(run, arm):
    out = []
    for f in sorted(os.listdir(run)):
        if f.startswith(f"trajectories_k{arm}_") and f.endswith(".jsonl"):
            for line in open(os.path.join(run, f), encoding="utf-8"):
                out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--arms", default="-1,memfree")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    rows, per_rows = [], []
    for arm in a.arms.split(","):
        recs = load(a.run, arm)
        assert recs, f"no trajectories for arm {arm} in {a.run}"
        per = []
        for r in recs:
            tgt = r["source_record"].get("reference")
            assert tgt, f"no decoded target on {r['metadata']['prompt_id']}"
            g = r["aggregate"]["generation"]
            per.append(dict(prompt_id=r["metadata"]["prompt_id"],
                            arm=arm,
                            novel=r["source_record"].get("novel", ""),
                            # feat-165: the two rules name their own "did it bind" counter
                            # differently -- MemFree blocks a token, TokenSwap changes which one
                            # is served -- and this scorer reports whichever the run recorded.
                            blocked_steps=r["aggregate"].get(
                                "blocked_steps", r["aggregate"].get("changed_steps")),
                            nv_recall=round(nv_recall(g, tgt), 4),
                            lcs_word=lcs_word(g, tgt),
                            rouge_l=round(rouge_l_score(g, tgt), 4)))
        per_rows += per
        ge5 = sum(1 for x in per if x["rouge_l"] >= 0.5)
        ge3 = sum(1 for x in per if x["rouge_l"] >= 0.3)
        rows.append(dict(
            arm=arm, n_passages=len(per),
            blocked_steps=sum(x["blocked_steps"] or 0 for x in per),
            nv_recall_mean=round(st.mean(x["nv_recall"] for x in per), 4),
            nv_recall_max=round(max(x["nv_recall"] for x in per), 4),
            lcs_word_mean=round(st.mean(x["lcs_word"] for x in per), 2),
            rouge_l_mean=round(st.mean(x["rouge_l"] for x in per), 4),
            rouge_ge_0p3_count=ge3, rouge_ge_0p5_count=ge5,
            ge_0p01_pct=round(100 * sum(1 for x in per if x["nv_recall"] >= 0.01) / len(per), 1),
        ))
        print(f"[bls] {arm:>8s}  n={len(per)}  nv_recall mean={rows[-1]['nv_recall_mean']:.4f} "
              f"max={rows[-1]['nv_recall_max']:.4f}  lcs_word={rows[-1]['lcs_word_mean']:.2f}  "
              f"ROUGE-L>=0.5: {ge5}/{len(per)}  >=0.3: {ge3}/{len(per)}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    for name, data in (("blocklist_decode", rows), ("blocklist_decode_per_passage", per_rows)):
        p = os.path.join(a.out, f"{name}{a.tag and '_' + a.tag}.csv")
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
        print(f"[bls] wrote {p}", flush=True)


if __name__ == "__main__":
    main()
