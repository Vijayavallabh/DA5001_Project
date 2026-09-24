"""Score results/deecho_rejudge_note.md: does each older judged reading survive de-echoing?

A reading is its interval's sign (POSITIVE / NEGATIVE / COVERS ZERO); it survives iff the de-echoed
pass gives the same sign as the pass on record. No GPU.

  --coverage   (host B, where output/ lives) gate G0: for every h1.py directory the passes read,
               the fraction of records whose full_text decomposes as prompt + generation, and the
               fraction that carried an echo. -> results/deecho_rejudge_coverage.csv
  default      -> results/deecho_rejudge.csv

Usage: .venv/bin/python analysis/deecho_rejudge.py --out results
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dap.shared import served_generation  # noqa: E402

# (pass, judge, suffix on record, de-echoed suffix). Judge B's committed rung is feat-184's A1.
PASSES = [
    ("panel k=10", "D Qwen2.5-72B", "__qwen72b", "__qwen72b_deecho"),
    ("panel k=10", "E Mixtral-8x7B", "__mixtral", "__mixtral_deecho"),
    ("panel k=10", "F Qwen2.5-14B", "__qwen14b", "__qwen14b_deecho"),
    ("panel k=10", "G gemma-2-27b", "__gemma27b", "__gemma27b_deecho"),
    ("panel k=10", "B Phi-3.5-mini", "", "_deecho"),
    ("ladder opp Llama-3.1-8B", "B Phi-3.5-mini", "", "_deecho"),
    ("ladder opp Qwen2.5-0.5B", "B Phi-3.5-mini", "__opp_qwen05b", "__opp_qwen05b_deecho"),
    ("ladder opp Qwen2.5-1.5B", "B Phi-3.5-mini", "__opp_qwen15b", "__opp_qwen15b_deecho"),
    ("ladder opp Qwen2.5-3B", "B Phi-3.5-mini", "__opp_qwen3b", "__opp_qwen3b_deecho"),
    ("ladder opp Qwen2.5-14B", "B Phi-3.5-mini", "__opp2", "__opp2_deecho"),
    ("ladder opp Llama-3.1-8B", "C Llama-3.1-8B", "__opp_committed_judgeC", "__opp_committed_judgeC_deecho"),
    ("ladder opp Qwen2.5-0.5B", "C Llama-3.1-8B", "__opp_qwen05b_judgeC", "__opp_qwen05b_judgeC_deecho"),
    ("ladder opp Qwen2.5-1.5B", "C Llama-3.1-8B", "__opp_qwen15b_judgeC", "__opp_qwen15b_judgeC_deecho"),
    ("ladder opp Qwen2.5-3B", "C Llama-3.1-8B", "__opp_qwen3b_judgeC", "__opp_qwen3b_judgeC_deecho"),
    ("workload AlpacaEval k=1", "B Phi-3.5-mini", "__mixpowk_judgeB", "__mixpowk_judgeB_deecho"),
    ("workload MT-Bench k=1.0", "B Phi-3.5-mini", "__mtb_conc_bind", "__mtb_conc_bind_deecho"),
    ("workload Gutenberg k=0.9", "B Phi-3.5-mini", "__gutenberg_conc_bind", "__gutenberg_conc_bind_deecho"),
    ("workload unseen books k=1.0", "B Phi-3.5-mini", "__unseenbooks_conc_bind", "__unseenbooks_conc_bind_deecho"),
    ("workload CoTaEval-QA k=1.4", "B Phi-3.5-mini", "__cotaeval_qa_conc_bind", "__cotaeval_qa_conc_bind_deecho"),
    ("workload ours (Arm C) k=0.9", "B Phi-3.5-mini", "__wscope_c", "__wscope_c_deecho"),
]
QUANT = {"D1": "D1 selection gain", "D2": "D2 metered gain", "D3": "D3 difference of gains"}

# h1.py directories the passes read (the Qwen opponents are blocklist_decode, no full_text, no echo)
DIRS = ["output/xfer/sel_anchor64", "output/xfer/conc_all", "output/sweep_plain",
        "output/mixpow/sel_anchor64", "output/mixpow/conc_k10", "output/mixpow/baseline",
        "output/mtb/sel_anchor256", "output/mtb/conc_bind", "output/mtb/baseline",
        "output/wscope/a_sel", "output/wscope/c_conc", "output/wscope/a_baseline"] + [
        f"output/{c}/{d}" for c in ("gutenberg", "unseenbooks", "cotaeval_qa")
        for d in ("sel_anchor64", "conc_bind", "baseline")]
_SPECIAL = re.compile(r"<\|[^|>]*\|>")


def sign(lo, hi):
    return "POSITIVE" if lo > 0 else "NEGATIVE" if hi < 0 else "COVERS ZERO"


def read(path):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        q = r["quantity"][:2]
        if q in QUANT:
            out[q] = (float(r["value"]), float(r["lo95"]), float(r["hi95"]))
    assert set(out) == set(QUANT), path
    return out


def score(res):
    rows = []
    for p, j, old, new in PASSES:
        fo = os.path.join(res, f"order_averaged_h2h{old}.csv")
        fn = os.path.join(res, f"order_averaged_h2h{new}.csv")
        if not os.path.exists(fn):
            print(f"[skip] {p} / {j}: {fn} not written")
            continue
        o, n = read(fo), read(fn)
        for q in QUANT:
            ro, rn = sign(*o[q][1:]), sign(*n[q][1:])
            rows.append(dict({"pass": p}, judge=j, quantity=QUANT[q],
                             value_old=o[q][0], lo_old=o[q][1], hi_old=o[q][2], reading_old=ro,
                             value_deecho=n[q][0], lo=n[q][1], hi=n[q][2], reading_deecho=rn,
                             survives="YES" if ro == rn else "NO"))
    return rows


def coverage():
    rows = []
    for d in DIRS:
        tot = dec = echo = 0
        for f in sorted(glob.glob(os.path.join(d, "trajectories_*.jsonl"))):
            for ln in open(f, encoding="utf-8"):
                if not ln.strip():
                    continue
                r = json.loads(ln)
                a, pre = r["aggregate"], r["prefix_analysis"]["prefix_text"]
                full, gen, prompt = a.get("full_text") or "", a.get("generation") or "", _SPECIAL.sub("", pre or "")
                tot += 1
                dec += bool(prompt and full.startswith(prompt) and gen.endswith(full[len(prompt):]))
                echo += served_generation(a, pre) != gen
        rows.append(dict(dir=d, records=tot, decomposes=round(dec / tot, 6) if tot else "",
                         echo_frac=round(echo / tot, 6) if tot else "",
                         g0="PASS" if tot and dec / tot >= 0.99 else "FAIL"))
        print(rows[-1])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--dir", default="results", help="where the per-pass CSVs are read from")
    a = ap.parse_args()
    rows, name = (coverage(), "deecho_rejudge_coverage.csv") if a.coverage else (score(a.dir), "deecho_rejudge.csv")
    with open(os.path.join(a.out, name), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    if not a.coverage:
        for r in rows:
            if r["quantity"].startswith("D3"):
                print(f"{r['pass']:<30} {r['judge']:<16} {r['value_old']:+.4f} {r['reading_old']:<12} -> "
                      f"{r['value_deecho']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] {r['reading_deecho']:<12} {r['survives']}")
        pr = [r for r in rows if r["pass"].startswith("panel") and r["quantity"].startswith("D3")]
        print(f"panel r = {sum(r['reading_deecho'] == 'POSITIVE' for r in pr)} of {len(pr)} (registered r = 4 of 5)")
    print(f"wrote {os.path.join(a.out, name)}")


if __name__ == "__main__":
    main()
