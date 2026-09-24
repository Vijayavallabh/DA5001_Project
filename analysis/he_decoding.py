"""Score feat-195 (results/onset_prediction_he_decoding.md): the head-to-head at the authors' decoding
settings, temperature 0.7 and repetition penalty 1.1. No GPU.

Reads the four judge passes J1-J4 (results/order_averaged_h2h{,_per_prompt}_t07_*.csv), the reward
cache of the temperature-0.7 pool and the generated arms, and writes results/he_decoding.csv with the
registered gates and readings. The bands were committed before any token was decoded; this script
reads them, it does not choose them. Table 2's rows for the block are computed where every other row
of that table is, analysis/served_opponent.py.

Usage: .venv/bin/python analysis/he_decoding.py --out results
"""
import argparse
import csv
import json
import os

PASSES = {"J1": "t07_8b_k10", "J2": "t07_8b_k1", "J3": "t07_70b_k20", "J4": "t07_70b_k1"}
CLASSES = ("neutral", "factual", "creative")


def summary(d, tag):
    return {r["quantity"][:2]: r for r in csv.DictReader(open(os.path.join(d, f"order_averaged_h2h_{tag}.csv")))}


def per_prompt(d, tag):
    return {r["prompt_id"]: r for r in
            csv.DictReader(open(os.path.join(d, f"order_averaged_h2h_per_prompt_{tag}.csv")))}


def lowest_seed_texts(run_dir, tok):
    """prompt_id -> (metadata, served text) of the lowest-seed trajectory, ordinary classes only: the
    text is read by the judge's own loader (analysis/utility.load_arm, de-echoed), never re-parsed."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from analysis.order_averaged_h2h import lowest_seed
    from analysis.utility import load_arm
    text = lowest_seed(load_arm(run_dir, tok, "kl", deecho=True))
    meta = {}
    for cls in CLASSES:
        for line in open(os.path.join(run_dir, f"trajectories_k{tok}_{cls}.jsonl"), encoding="utf-8"):
            m = json.loads(line)["metadata"]
            if m["prompt_id"] not in meta or m["seed"] < meta[m["prompt_id"]]["seed"]:
                meta[m["prompt_id"]] = m
    assert set(meta) == set(text), (len(meta), len(text))
    return {p: (meta[p], text[p][1]) for p in text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--out", default="results")
    ap.add_argument("--t07", default="output/feat195/t07_8b")
    ap.add_argument("--committed", default="output/sweep_plain")
    ap.add_argument("--rewards", default="results/selection_rewards64_t07.csv")
    a = ap.parse_args()
    rows = []

    def add(band, quantity, value, lo="", hi="", reading="", n=""):
        rows.append(dict(band=band, quantity=quantity, value=value, lo95=lo, hi95=hi, reading=reading, n=n))

    s = {j: summary(a.dir, t) for j, t in PASSES.items()}
    pp = {j: per_prompt(a.dir, t) for j, t in PASSES.items()}

    # G0: every pass covers the 500 prompts; the pool holds 64 draws per prompt, the cache 32,000 rows
    for j, d in pp.items():
        add("G0", f"{j}: prompts judged", len(d), reading="PASS" if len(d) == 500 else "FAIL")
    rw = list(csv.DictReader(open(a.rewards)))
    per = {}
    for r in rw:
        per.setdefault(r["prompt_id"], set()).add(int(r["rank"]))
    ok = len(rw) == 32000 and len(per) == 500 and all(v == set(range(64)) for v in per.values())
    add("G0", "reward cache rows (64 ranks x 500 prompts)", len(rw), reading="PASS" if ok else "FAIL")

    # G1: the settings took effect, and the k=-1 text is not the committed temperature-1.0 opponent's
    new = lowest_seed_texts(a.t07, "-1")
    old = lowest_seed_texts(a.committed, "-1")
    meta_ok = all(m.get("temperature") == 0.7 and m.get("repetition_penalty") == 1.1 for m, _ in new.values())
    add("G1", "every k=-1 record carries temperature 0.7 and repetition_penalty 1.1", float(meta_ok),
        reading="PASS" if meta_ok else "FAIL", n=len(new))
    common = sorted(set(new) & set(old))
    diff = sum(new[p][1] != old[p][1] for p in common) / len(common)
    add("G1", "k=-1 text differs from the committed opponent's", round(diff, 4),
        reading="PASS" if diff >= 0.99 and len(common) == 500 else "FAIL", n=len(common))

    # G2: selection's per-prompt levels identical across J1-J4 (same text, same opponent, greedy judge)
    ref = pp["J1"]
    same = all(pp[j][p]["u_sel_n64"] == ref[p]["u_sel_n64"] for j in pp for p in ref)
    add("G2", "selection levels identical in J1-J4", float(same), reading="PASS" if same else "FAIL", n=len(ref))

    # the readings, as the judge script labels them
    def rd(j, q, band, what):
        r = s[j][q]
        add(band, f"{j} {what}: {r['quantity']}", float(r["value"]), float(r["lo95"]), float(r["hi95"]),
            r["reading"], r["n"])
    rd("J1", "D3", "H1", "8B k=10")
    rd("J1", "D5", "H2", "8B k=0.5")
    rd("J3", "D3", "H3", "70B k=20")
    rd("J2", "D4", "H4", "the 70B alone")
    rd("J3", "D5", "H5", "70B k=0.5")
    rd("J2", "D3", "desc", "8B k=1")
    rd("J4", "D3", "desc", "70B k=1")
    rd("J1", "D1", "desc", "selection")

    # K/S_w at 0.7 is relative to the WARPED anchor; the registration forbids reading it off the 1.0
    # anchor. analysis/regimes.py --temperature 0.7 --repetition-penalty 1.1 wrote the per-passage
    # surprisal; S_w is 50 times its median per-token rate, as results/window_vacuity.csv does at 1.0.
    import statistics
    rg = list(csv.DictReader(open(os.path.join(a.dir, "regimes_copybench_t07.csv"))))
    sw = 50 * statistics.median(float(r["nats_per_tok"]) for r in rg)
    add("S_w", "50-token window, warped anchor (0.7, 1.1), median over passages", round(sw, 4), n=len(rg))
    for k in (0.5, 1, 10, 20):
        add("K/S_w", f"k={k:g}, K = 200k over the warped anchor's S_w", round(200 * k / sw, 4))
    for j, q in (("J1", "D4"), ("J3", "D4")):
        rd(j, q, "desc", "the k=0.5 meter over its anchor")

    out = os.path.join(a.out, "he_decoding.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print("wrote", out)


if __name__ == "__main__":
    main()
