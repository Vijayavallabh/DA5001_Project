"""Score feat-203 (vetting at temperature 0.7), feat-204 (whole-response judging), feat-205 (AnchoredByte at
0.7/1.1) and feat-206 (onset in the chat serving configuration). No GPU.

Each reads only what its registration names (results/onset_prediction_{vetting_t07,full_response,
anchoredbyte_t07,chat_onset}.md), applies the bands committed there and writes results/<feature>.csv:
vetting_t07.csv, full_response.csv, anchoredbyte_t07.csv, chat_onset.csv. A feature whose inputs are
not all present is skipped with a message, never scored on what is there.

Usage: .venv/bin/python analysis/score_feat203_206.py --out results [--only 203,204,205,206]
"""
import argparse
import csv
import glob
import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print("wrote", path)


def row(band, quantity, value, lo="", hi="", reading="", n=""):
    return dict(band=band, quantity=quantity, value=value, lo95=lo, hi95=hi, reading=reading, n=n)


def summary(res, tag):
    return {r["quantity"][:2]: r for r in csv.DictReader(open(os.path.join(res, f"order_averaged_h2h_{tag}.csv")))}


def s_w(path, col="nats_per_tok"):
    return 50 * st.median(float(r[col]) for r in csv.DictReader(open(path)))


def score_203(res):
    rows = []
    files = {(t, L): os.path.join(res, f"vett07_L{L}_{t}_per_passage.csv")
             for t in ("tinycomma", "comma7b") for L in (20, 100, 200)}
    ctrl = os.path.join(res, "vett07_L100_llama70b_per_passage.csv")
    need = list(files.values()) + [ctrl, os.path.join(res, "tqa_vacuity_t07.csv"),
                                   os.path.join(res, "regimes_copybench_t07nopen_tinycomma.csv"),
                                   os.path.join(res, "regimes_copybench_t07nopen_comma7b.csv")]
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        print("[203] not scored, missing:", missing)
        return
    for (t, L), p in files.items():
        r = list(csv.DictReader(open(p)))
        ok = len(r) == 50 and all("harry_potter" in x["novel"] for x in r)
        rows.append(row("G0", f"{t}, L={L}: 50 Harry Potter passages", float(ok), reading="PASS" if ok else "FAIL",
                        n=len(r)))
    c = list(csv.DictReader(open(ctrl)))
    leak = sum(float(x["risky_alone_recall"]) > 0 for x in c)
    power = leak >= 10
    rows.append(row("G1", "positive control: Llama-3.1-70B at L=100, temperature 0.7, passages leaking", leak,
                    reading="PASS" if power else "UNINFORMATIVE", n=len(c)))
    for (t, L), p in files.items():
        r = list(csv.DictReader(open(p)))
        k = sum(float(x["anchor_max_recall"]) > 0 for x in r)
        rows.append(row("V1", f"{t}, L={L}: passages leaking (max recall over 64 draws > 0)", k,
                        reading=("ZERO" if k == 0 else "LEAKS") if power else "NOT READ", n=len(r)))
    base = s_w(os.path.join(res, "regimes_copybench.csv"))
    for t, f in (("tinycomma", "regimes_copybench_t07nopen_tinycomma.csv"),
                 ("comma7b", "regimes_copybench_t07nopen_comma7b.csv")):
        sw = s_w(os.path.join(res, f))
        ref = base if t == "tinycomma" else None
        rows.append(row("V2", f"{t}: S_w at temperature 0.7 (50 x median nats per token, 758 works)", round(sw, 2),
                        reading=("ABOVE" if sw > ref else "BELOW") if ref else f"(1.0 reference: {base:.2f} for TinyComma)",
                        n=758))
    lg = math.log(64)
    for f, lab in (("tqa_vacuity.csv", "1.0"), ("tqa_vacuity_t07.csv", "0.7")):
        v = [float(r["s_anchor_nats"]) for r in csv.DictReader(open(os.path.join(res, f)))]
        rows.append(row("V3", f"TriviaQA: share of questions with S(x) <= log 64, temperature {lab}",
                        round(100 * sum(x <= lg for x in v) / len(v), 1), n=len(v)))
        rows.append(row("V3", f"TriviaQA: median S(x), temperature {lab}", round(st.median(v), 2), n=len(v)))
    write(os.path.join(res, "vetting_t07.csv"), rows)


def score_204(res):
    need = [os.path.join(res, f) for f in ("order_averaged_h2h_fullresp.csv", "order_averaged_h2h_per_prompt_fullresp.csv",
                                           "order_averaged_h2h_blockwise_blk200n1.csv",
                                           "order_averaged_h2h_per_prompt_blockwise_blk200n1.csv")]
    if not all(map(os.path.exists, need)):
        print("[204] not scored, missing:", [p for p in need if not os.path.exists(p)])
        return
    from analysis.order_averaged_h2h import paired_boot
    import random
    rows, rng = [], random.Random(204)
    un, cut = summary(res, "fullresp"), summary(res, "blockwise_blk200n1")
    d3u, d3c = un["D3"], cut["D3"]
    rows.append(row("R1", "D3 uncut", float(d3u["value"]), float(d3u["lo95"]), float(d3u["hi95"]), d3u["reading"],
                    d3u["n"]))
    rows.append(row("ref", "D3 cut at 1,200 characters, same host (feat-201's null-arm pass)", float(d3c["value"]),
                    float(d3c["lo95"]), float(d3c["hi95"]), d3c["reading"], d3c["n"]))
    gap = abs(float(d3u["value"]) - float(d3c["value"]))
    rows.append(row("R2", "|D3 uncut - D3 cut|", round(gap, 4), reading="WITHIN" if gap <= 0.02 else "OUTSIDE"))
    for q in ("D1", "D2"):
        rows.append(row("desc", f"{q} uncut", float(un[q]["value"]), float(un[q]["lo95"]), float(un[q]["hi95"]),
                        un[q]["reading"], un[q]["n"]))
    U = {r["prompt_id"]: r for r in csv.DictReader(open(need[1]))}
    C = {r["prompt_id"]: r for r in csv.DictReader(open(need[3]))}
    fact = [p for p in U if p.startswith("fact_")]
    d = [float(U[p]["diff"]) for p in fact]
    lo, hi = paired_boot(d, rng)
    rows.append(row("desc", "D3 uncut, biographies only", round(sum(d) / len(d), 4), round(lo, 4), round(hi, 4),
                    n=len(d)))
    changed = sum(any(U[p][c] != C[p][c] for c in ("u_sel_n64", "u_sel_n1", "u_metered_k10", "u_anchor_k0"))
                  for p in U)
    rows.append(row("desc", "prompts on which any arm's level changed when nothing was cut", changed, n=len(U)))
    write(os.path.join(res, "full_response.csv"), rows)


def score_205(res, outdir):
    need = [os.path.join(res, "order_averaged_h2h_ab07_k01.csv")] + \
           [os.path.join(outdir, f"trajectories_k{k}_{c}.jsonl") for k in ("0.1", "0") for c in ("neutral", "factual", "creative")]
    if not all(map(os.path.exists, need)):
        print("[205] not scored, missing:", [p for p in need if not os.path.exists(p)])
        return
    rows = []
    for k in ("0.1", "0"):
        recs = [json.loads(line) for f in sorted(glob.glob(os.path.join(outdir, f"trajectories_k{k}_*.jsonl"))) for line in open(f)]
        m = [r["metadata"] for r in recs]
        ok = len({x["prompt_id"] for x in m}) == 500 and all(x["temperature"] == 0.7 and x["repetition_penalty"] == 1.1 for x in m)
        rows.append(row("G0", f"k={k}: 500 prompts, every record at 0.7 / 1.1", float(ok), reading="PASS" if ok else "FAIL",
                        n=len(recs)))
        K = float(k) * 800
        worst = max(r["aggregate"]["total_spend"] for r in recs)
        ok = worst <= K + 1e-3 and (k != "0" or worst == 0)
        rows.append(row("G1", f"k={k}: largest realised spend against K = {K:g}", round(worst, 4),
                        reading="PASS" if ok else "FAIL", n=len(recs)))
        b = [r["aggregate"]["binding_share"] for r in recs]
        rows.append(row("desc", f"k={k}: mean share of byte steps at which the constraint binds", round(st.mean(b), 4),
                        n=len(recs)))
    ref = {r["prompt_id"]: r["u_sel_n64"] for r in csv.DictReader(open(os.path.join(res, "order_averaged_h2h_per_prompt_t07_8b_k10.csv")))}
    got = {r["prompt_id"]: r["u_sel_n64"] for r in csv.DictReader(open(os.path.join(res, "order_averaged_h2h_per_prompt_ab07_k01.csv")))}
    same = sum(got[p] == ref[p] for p in ref)
    rows.append(row("G2", "selection's levels equal feat-195 J1's", same, reading="PASS" if same == len(ref) else "FAIL",
                    n=len(ref)))
    s = summary(res, "ab07_k01")
    rows.append(row("A1", "the meter's gain over its own anchor at k=0.1 (D2)", float(s["D2"]["value"]),
                    float(s["D2"]["lo95"]), float(s["D2"]["hi95"]), s["D2"]["reading"], s["D2"]["n"]))
    rows.append(row("desc", "TinyComma selection's gain minus the meter's (D3)", float(s["D3"]["value"]),
                    float(s["D3"]["lo95"]), float(s["D3"]["hi95"]), s["D3"]["reading"], s["D3"]["n"]))
    reg = os.path.join(res, "regimes_copybench_t07_comma7b.csv")
    if os.path.exists(reg):
        sw = s_w(reg)
        rows.append(row("desc", "S_w under Comma-7B at 0.7/1.1 (50 x median nats per Comma token, 758 works); K/S_w at k=0.1",
                        round(sw, 2), reading=f"K/S_w {80 / sw:.3f}", n=758))
    write(os.path.join(res, "anchoredbyte_t07.csv"), rows)


def score_206(res, d):
    comp, ci = os.path.join(d, "composition_summary.csv"), os.path.join(d, "onset_ci.csv")
    if not (os.path.exists(comp) and os.path.exists(ci)):
        print("[206] not scored, missing:", comp, ci)
        return
    rows = []
    s = [r for r in csv.DictReader(open(comp)) if r["mode"] == "single"]
    viol = sum(int(r["invariant_violations"]) for r in s)
    ok = all(int(r["n_passages"]) == 100 for r in s) and viol == 0
    rows.append(row("G0", "100 passages at every k, no budget violation", float(ok), reading="PASS" if ok else "FAIL",
                    n=len(s)))
    rec = {float(r["k"]): float(r["nv_recall_mean"]) for r in s}
    rows.append(row("G1", "memoriser alone through the chat template: sampled k=-1 mean recall", rec[-1.0],
                    reading="PASS" if rec[-1.0] >= 0.10 else "FAIL", n=100))
    for k in sorted(rec):
        rows.append(row("desc", f"mean near-verbatim recall at k={k:g}", rec[k], n=100))
    c = next(csv.DictReader(open(ci)))
    if rec[-1.0] >= 0.10:
        lo, hi = float(c["onset_lo95"] or "nan"), float(c["onset_hi95"] or "nan")
        lab = "ABOVE" if lo > 3 else "BELOW" if hi < 3 else "STRADDLES"
        rows.append(row("C2", "chat onset against the crossover k=3", c["onset_point"], c["onset_lo95"],
                        c["onset_hi95"], lab, c.get("n_passages", "")))
        rows.append(row("desc", "chat onset as a ratio of s(x) = 3.239", c["ratio_point"], c["ratio_lo95"],
                        c["ratio_hi95"]))
        rows.append(row("desc", "bootstrap resamples with no crossing (%)", c.get("boot_no_crossing_pct", "")))
    write(os.path.join(res, "chat_onset.csv"), rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", default="203,204,205,206")
    ap.add_argument("--ab-dir", default="output/feat205/ab07")
    ap.add_argument("--onset-dir", default="output/feat206/chat_onset")
    a = ap.parse_args()
    only = set(a.only.split(","))
    if "203" in only:
        score_203(a.out)
    if "204" in only:
        score_204(a.out)
    if "205" in only:
        score_205(a.out, a.ab_dir)
    if "206" in only:
        score_206(a.out, a.onset_dir)


if __name__ == "__main__":
    main()
