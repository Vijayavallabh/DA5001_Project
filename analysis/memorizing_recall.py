"""feat-008: summarise extraction from the memorising risky model at k=-1 (greedy and sampled E1 runs).

Usage: .venv/bin/python analysis/memorizing_recall.py --run greedy=output/memorizing_check_greedy --run sampled=output/memorizing_check --out results
Writes results/memorizing_model_recall.csv: per (run, split) mean/median nv-recall, share with nv-recall >= 0.8,
mean LCS (words/chars), mean ROUGE-L, n.
"""
import argparse, csv, glob, json, os, re, statistics as st

FNAME = re.compile(r"trajectories_k-1_(?P<split>[a-z_]+)\.jsonl$")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="name=dir (repeatable)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    rows = []
    for spec in args.run:
        name, d = spec.split("=", 1)
        for path in sorted(glob.glob(os.path.join(d, "trajectories_k-1_*.jsonl"))):
            split = FNAME.search(path)["split"]
            A = [json.loads(l)["aggregate"] for l in open(path) if l.strip()]
            if not A or split in ("neutral", "factual", "creative"):
                continue
            nv = [a["nv_recall"] for a in A]
            rows.append(dict(run=name, split=split, n=len(A), nv_recall_mean=round(st.mean(nv), 4), nv_recall_median=round(st.median(nv), 4),
                             nv_recall_ge_0p8_pct=round(100 * sum(x >= 0.8 for x in nv) / len(nv), 2),
                             lcs_word_mean=round(st.mean(a["lcs_word"] for a in A), 2), lcs_char_mean=round(st.mean(a["lcs_char"] for a in A), 1),
                             acs_word_mean=round(st.mean(a["acs_word"] for a in A), 2), rouge_l_mean=round(st.mean(a["rouge_l"] for a in A), 4),
                             gen_len_mean=round(st.mean(a["generation_length_tokens"] for a in A), 1)))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "memorizing_model_recall.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
