"""Score feat-201 (results/onset_prediction_blockwise.md): selection spent in installments. No GPU.

Reads each arm's judge pass (results/order_averaged_h2h{,_per_prompt}_blockwise_<arm>.csv), its served
records and block log (output/feat201/<arm>/), the gate file results/blockwise_gate.csv, the queue logs
(output/logs/feat201_q*.log) and the memoriser arm's per-passage file, and writes results/blockwise.csv
with the registered gates and readings. The bands were committed before any registered draw; this
script reads them, it does not choose them. An order-averaged level is a deterministic function of the
prompt, the two texts and the judge, so arms judged in different passes on one host are compared per
prompt (caution (ap)).

Usage: .venv/bin/python analysis/blockwise_scoring.py --out results
"""
import argparse
import csv
import glob
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.blockwise_selection import CLASSES, certificate  # noqa: E402

VALUE = {"blk10n64": (10, 64), "blk25n64": (25, 64), "blk50n64": (50, 64), "blk200n64": (200, 64),
         "blk100n8": (100, 8), "blk67n4": (67, 4), "blk34n2": (34, 2)}
OTHER = {"blk50n64_reward": (50, 64), "blk10n64_planner": (10, 64), "blk200n1": (200, 1)}
N_BOOT = 10000


def summary(d, arm):
    return {r["quantity"][:2]: r for r in csv.DictReader(open(os.path.join(d, f"order_averaged_h2h_blockwise_{arm}.csv")))}


def levels(d, arm):
    return {r["prompt_id"]: r for r in
            csv.DictReader(open(os.path.join(d, f"order_averaged_h2h_per_prompt_blockwise_{arm}.csv")))}


def boot(diffs, rng):
    n = len(diffs)
    xs = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return xs[int(0.025 * N_BOOT)], xs[int(0.975 * N_BOOT)]


def texts(outdir, arm):
    out = {}
    for cls in CLASSES:
        for line in open(os.path.join(outdir, arm, f"trajectories_k{arm}_{cls}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            out[r["metadata"]["prompt_id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--outdir", default="output/feat201")
    ap.add_argument("--logs", default="output/logs")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    rows, rng = [], random.Random(201)

    def add(band, arm, quantity, value, lo="", hi="", reading="", n=""):
        rows.append(dict(band=band, arm=arm, quantity=quantity, value=value, lo95=lo, hi95=hi, reading=reading, n=n))

    arms = list(VALUE) + list(OTHER)
    S = {x: summary(a.results, x) for x in arms}
    U = {x: levels(a.results, x) for x in arms}
    T = {x: texts(a.outdir, x) for x in arms}

    # G0: coverage, length, block logs
    for x in arms:
        blog = list(csv.DictReader(open(os.path.join(a.outdir, x, f"blocks_{x}.csv"))))
        per_block = {}
        for r in blog:
            per_block.setdefault(int(r["block"]), set()).add(r["prompt_id"])
        nested = all(per_block[b] <= per_block[b - 1] for b in per_block if b > 0)
        ok = (len(T[x]) == 500 and max(r["aggregate"]["generation_length_tokens"] for r in T[x].values()) <= 200
              and len(per_block.get(0, ())) == 500 and nested and len(U[x]) == 500)
        add("G0", x, "500 records, <= 200 tokens, block log nested and complete, 500 judged", float(ok),
            reading="PASS" if ok else "FAIL", n=len(T[x]))
    mem = list(csv.DictReader(open(os.path.join(a.results, "blockwise_extraction_blk10n64_memoriser.csv"))))
    base = {r["prompt_id"]: r for r in csv.DictReader(open(os.path.join(a.results, "selection_extraction_per_passage.csv")))}
    ok = len(mem) == 100 and {r["prompt_id"] for r in mem} == set(base)
    add("G0", "blk10n64_memoriser", "100 passages, the committed extraction arm's", float(ok),
        reading="PASS" if ok else "FAIL", n=len(mem))

    # G1: the sampling law, as the gate script wrote it
    g1 = next(r for r in csv.DictReader(open(os.path.join(a.results, "blockwise_gate.csv"))) if r["source"] == "G1")
    add("G1", "blk200n1", "anchor surprisal of served tokens against the pool's rank-0 draws",
        g1["nats_per_token"], reading=g1["nats_per_token"].rsplit(": ", 1)[-1])

    # G2: the judge path, against feat-198's pass on the same host
    ref = {r["prompt_id"]: r for r in
           csv.DictReader(open(os.path.join(a.results, "order_averaged_h2h_per_prompt_scorer_gemma27b.csv")))}
    for x in arms:
        same = sum(U[x][p]["u_metered_k10"] == ref[p]["u_metered_k10"] and U[x][p]["u_anchor_k0"] == ref[p]["u_anchor_k0"]
                   for p in ref)
        add("G2", x, "meter and anchor levels equal feat-198's, prompt by prompt", same,
            reading="PASS" if same == len(ref) == 500 else "FAIL", n=len(ref))

    # G3: the resolved sampling configuration each generation printed
    lines = [ln for f in glob.glob(os.path.join(a.logs, "feat201_q*.log")) for ln in open(f, errors="replace")
             if "effective sampling" in ln]
    pure = sum("temperature=1.0 top_k=0 top_p=1.0" in ln for ln in lines)
    shipped = sum("temperature=0.6" in ln and "top_p=0.9" in ln for ln in lines)
    ok = pure == len(arms) + 1 and shipped == 1          # + the memoriser arm; the probe is the one shipped
    add("G3", "all", "pure-sampling lines / shipped-probe lines", f"{pure} / {shipped}",
        reading="PASS" if ok else "FAIL", n=len(lines))

    def u(x, p):
        return float(U[x][p][f"u_{x}"])

    pids = sorted(U["blk200n64"])

    def contrast(band, x, y, labels):
        d = [u(x, p) - u(y, p) for p in pids]
        lo, hi = boot(d, rng)
        add(band, f"{x} - {y}", "paired difference of order-averaged levels", round(sum(d) / len(d), 4),
            round(lo, 4), round(hi, 4), labels[0] if lo > 0 else labels[1] if hi < 0 else "TIE", len(d))

    # B1: each value-scored arm gains over the anchor control (the pass's own D4)
    for x in VALUE:
        r = S[x]["D4"]
        add("B1", x, "gain over the anchor control (D4)", float(r["value"]), float(r["lo95"]), float(r["hi95"]),
            r["reading"], r["n"])
    for L in (10, 25, 50):
        contrast("B2", f"blk{L}n64", "blk200n64", ("INSTALLMENTS WIN", "ONCE WINS"))
    for x in ("blk100n8", "blk67n4", "blk34n2"):
        contrast("B3", x, "blk200n64", ("INSTALLMENTS WIN", "ONCE WINS"))
    r = S["blk10n64_planner"]["D4"]
    add("B4", "blk10n64_planner", "gain over the anchor control (D4)", float(r["value"]), float(r["lo95"]),
        float(r["hi95"]), r["reading"], r["n"])
    contrast("B4", "blk10n64_planner", "blk10n64", ("PLANNER WINS", "VALUE WINS"))
    contrast("B5", "blk50n64", "blk50n64_reward", ("VALUE WINS", "REWARD WINS"))

    rec = [float(r["recall"]) for r in mem]
    add("B6", "blk10n64_memoriser", "near-verbatim recall, mean over 100 passages", round(sum(rec) / len(rec), 4),
        reading="ZERO" if max(rec) == 0 else "NONZERO", n=len(rec))
    add("B6", "blk10n64_memoriser", "near-verbatim recall, maximum", max(rec), n=len(rec))
    add("B6", "blk10n64_memoriser", "passages with recall >= 0.01", sum(x >= 0.01 for x in rec), n=len(rec))
    rou = [float(r["rouge"]) for r in mem]
    para = [float(r["rouge_n64"]) for r in csv.DictReader(open(os.path.join(a.results, "selection_extraction_paraphrase_per_passage.csv")))]
    add("desc", "blk10n64_memoriser", "ROUGE-L mean (whole-output adversarial arm at n=64 beside it)",
        round(sum(rou) / len(rou), 4), reading=f"whole output {sum(para) / len(para):.4f}", n=len(rou))
    add("desc", "baselines", "memoriser alone (k=-1) / anchor alone (n=1), committed, same passages and seeds",
        f"{statistics.mean(float(r['risky_alone_recall']) for r in base.values()):.4f} / "
        f"{statistics.mean(float(r['recall_n1']) for r in base.values()):.4f}", n=len(base))

    # descriptive: certificate, served length and empties, D5 against the committed selection, the null arm
    for x in arms:
        L, n = {**VALUE, **OTHER}[x]
        whole, window = certificate(200, L, n)
        words = sorted(len(r["aggregate"]["generation"].split()) for r in T[x].values())
        empty = sum(not r["aggregate"]["generation"].strip() for r in T[x].values())
        add("desc", x, "certificate nats (whole output / any 50-token window)", f"{whole:.2f} / {window:.2f}")
        add("desc", x, "median words served / empty served", f"{words[len(words) // 2]} / {empty}", n=len(words))
        r = S[x]["D5"]
        add("desc", x, "minus the committed selection at n=64 (D5)", float(r["value"]), float(r["lo95"]),
            float(r["hi95"]), n=r["n"])
        blog = list(csv.DictReader(open(os.path.join(a.outdir, x, f"blocks_{x}.csv"))))
        add("desc", x, "reward calls (candidates scored)", len(blog) * n, n=len(blog))
    r = S["blk200n1"]["D4"]
    add("desc", "blk200n1", "the anchor alone through this code path, gain over the anchor control",
        float(r["value"]), float(r["lo95"]), float(r["hi95"]), r["reading"], r["n"])

    path = os.path.join(a.out, "blockwise.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print("wrote", path)


if __name__ == "__main__":
    main()
