"""Score feat-210 (results/onset_prediction_windowed_meter.md): the sliding-window meter and every
mechanism at a matched certificate. No GPU: reads the generations, the leakage sweeps and the judge
passes of analysis/matched_h2h.py, and writes

  results/windowed_arms.csv     per generation arm: coverage, invariant, binding shares, worst span
  results/windowed_leakage.csv  per configuration and window budget: near-verbatim recall
  results/windowed_meter.csv    gates G0-G2, predictions P1-P7 with their verdicts, and the registered
                                descriptive readings (onset, sensitivity of the k=0.5 null, ...)

Usage: .venv/bin/python analysis/score_feat210.py --out results
"""
import argparse
import csv
import glob
import json
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a_patch.pathwise import max_span_sum  # noqa: E402
from analysis.order_averaged_h2h import paired_boot  # noqa: E402

F = "output/feat210"
CLASSES = ("neutral", "factual", "creative")
# arm -> (dir, filename token, window budget or None, whole-output pathwise K or None)
GEN = {"win_0": ("win_plain", "0", None, None), "win_4.16": ("win_plain", "4.1589", 4.1589, None),
       "win_12.48": ("win_plain", "12.4767", 12.4767, None), "win_24.95": ("win_plain", "24.9534", 24.9534, None),
       "win_40": ("win_plain", "40", 40.0, None), "win_125": ("win_plain", "125", 125.0, None),
       "pw_83.18": ("pw_plain", "0.415888", None, 83.1776), "pw_33.27": ("pw_plain", "0.166355", None, 33.271),
       "frontpw_4.16": ("front_plain", "1e-09", None, 4.158883 + 2e-7),
       "winc_0": ("win_chat", "0", None, None), "winc_12.48": ("win_chat", "12.4767", 12.4767, None),
       "winc_24.95": ("win_chat", "24.9534", 24.9534, None), "winc_40": ("win_chat", "40", 40.0, None),
       "winc_125": ("win_chat", "125", 125.0, None)}
P = {  # registered differences -> predicted label
    "P1": [("sel_n64 - win_4.16", "CONFIRMED"), ("sel_n64 - frontpw_4.16", "CONFIRMED")],
    "P2": [("sel_n64 - win_24.95", "CONFIRMED"), ("sel_n64 - win_40", "CONFIRMED"), ("sel_n64 - win_125", "CONFIRMED")],
    "P3": [("blk10n64 - pw_83.18", "CONFIRMED"), ("blk25n64 - pw_33.27", "CONFIRMED")],
    "P4": [("blk10n64 - win_24.95", "CONFIRMED"), ("blk25n64 - win_12.48", "CONFIRMED")],
    "P5": [("sel_n64 - winc_125", "REFUTED"), ("sel_n64 - winc_12.48", "CONFIRMED")]}
Z80 = 1.959964 + 0.841621        # two-sided alpha 0.05, power 0.80, normal approximation


def arm_stats(name):
    d, tok, W, K = GEN[name]
    recs = []
    for c in CLASSES:
        for line in open(os.path.join(F, d, f"trajectories_k{tok}_{c}.jsonl")):
            recs.append(json.loads(line))
    spans, tots, bad, act, forced, free = [], [], 0, 0, 0, 0
    for r in recs:
        a = r["aggregate"]
        n = a["steps_forced_safe"] + a["steps_active"] + a["steps_risky_unchanged"]   # true length, caution (ah)
        rs = [s["r_t"] for s in r["per_step_log"][:n] if s.get("r_t") is not None]
        span = max_span_sum(rs, 50)
        spans.append(span)
        tots.append(sum(rs))
        act, forced, free = act + a["steps_active"], forced + a["steps_forced_safe"], free + a["steps_risky_unchanged"]
        if W is not None and span > W + 1e-3:
            bad += 1
        if K is not None and sum(rs) > K + 1e-3:
            bad += 1
    steps = max(1, act + forced + free)
    return dict(arm=name, dir=d, token=tok, W=W if W is not None else "", K_output=K if K is not None else "",
                trajectories=len(recs), prompts=len({r["metadata"]["prompt_id"] for r in recs}),
                violations=bad, active_pct=round(100 * act / steps, 2), forced_pct=round(100 * forced / steps, 2),
                risky_unchanged_pct=round(100 * free / steps, 2),
                worst_span_median=round(st.median(spans), 2), worst_span_max=round(max(spans), 2),
                total_ratio_median=round(st.median(tots), 2))


def rows_of(path):
    return list(csv.DictReader(open(path)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--results", default="results", help="where the judge passes and the committed pass are read from")
    ap.add_argument("--seed", type=int, default=7717)
    a = ap.parse_args()
    out = []

    def add(**kw):
        out.append(kw)
        print("  ".join(f"{k}={v}" for k, v in kw.items()))

    # ---- generation arms: G0 coverage, G1 invariant, binding shares --------------------------
    arms = [arm_stats(n) for n in GEN]
    with open(os.path.join(a.out, "windowed_arms.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(arms[0])); w.writeheader(); w.writerows(arms)
    add(band="G0", quantity="generation arms covering all 500 prompts", value=sum(r["prompts"] == 500 for r in arms),
        n=len(arms), reading="PASS" if all(r["prompts"] == 500 for r in arms) else "FAIL")
    add(band="G1", quantity="trajectories over their certified budget (window or output)",
        value=sum(r["violations"] for r in arms), n=sum(r["trajectories"] for r in arms),
        reading="PASS" if not any(r["violations"] for r in arms) else "FAIL")

    # ---- leakage ------------------------------------------------------------------------------
    leak = []
    for cfg in ("plain", "chat"):
        for r in rows_of(os.path.join(F, f"leak_{cfg}", "composition_summary.csv")):
            leak.append(dict(config=cfg, W=float(r["k"]), n_passages=int(r["n_passages"]),
                             nv_recall_mean=float(r["nv_recall_mean"]), nv_recall_ge_0p8_pct=float(r["nv_recall_ge_0p8_pct"]),
                             lcs_word_mean=float(r["lcs_word_mean"]), violations=int(r["invariant_violations"])))
    with open(os.path.join(a.out, "windowed_leakage.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(leak[0])); w.writeheader(); w.writerows(leak)
    add(band="G0", quantity="leakage runs covering the 100 passages", value=sum(r["n_passages"] == 100 for r in leak),
        n=len(leak), reading="PASS" if all(r["n_passages"] == 100 for r in leak) else "FAIL")
    add(band="G1", quantity="leakage queries over their certified budget", value=sum(r["violations"] for r in leak),
        n=len(leak), reading="PASS" if not any(r["violations"] for r in leak) else "FAIL")
    for cfg in ("plain", "chat"):
        rs = {r["W"]: r for r in leak if r["config"] == cfg}
        low = [W for W in rs if 0 < W <= 40]
        ok_low = all(rs[W]["nv_recall_mean"] < 0.01 for W in low)
        ok_hi = rs[125.0]["nv_recall_mean"] >= 0.01
        add(band="P6", quantity=f"{cfg}: recall < 0.01 at every W <= 40 and >= 0.01 at W = 125",
            value=f"{max(rs[W]['nv_recall_mean'] for W in low):.3f} / {rs[125.0]['nv_recall_mean']:.3f}",
            n=100, reading="right" if ok_low and ok_hi else "wrong")
        onset = min((W for W in rs if W > 0 and rs[W]["nv_recall_mean"] >= 0.01), default=None)
        add(band="descriptive", quantity=f"{cfg}: windowed onset, smallest grid W with mean recall >= 0.01",
            value=onset, n=100, reading=f"memoriser alone {rs[-1.0]['nv_recall_mean']:.4f}, anchor {rs[0.0]['nv_recall_mean']:.4f}")

    # ---- judged: G2 and predictions -------------------------------------------------------------
    new = {r["prompt_id"]: r for r in rows_of(os.path.join(a.results, "matched_h2h_per_prompt_matched_plain_B.csv"))}
    old = {r["prompt_id"]: r for r in rows_of(os.path.join(a.results, "order_averaged_h2h_per_prompt_deecho.csv"))}
    eq = {x: sum(float(new[p][f"u_{x}"]) == float(old[p][f"u_{x}"]) for p in old)
          for x in ("sel_n64", "sel_n1", "metered_k10", "anchor_k0")}
    add(band="G2", quantity="committed arms' per-prompt levels reproduced in the plain judge-B pass",
        value=min(eq.values()), n=len(old), reading="PASS" if min(eq.values()) == len(old) else "FAIL")
    res = {}
    for pas in ("plain_B", "plain_G", "chat_B", "chat_G"):
        f = os.path.join(a.results, f"matched_h2h_matched_{pas}.csv")
        if os.path.exists(f):
            res[pas] = {(r["quantity"], r["arm"]): r for r in rows_of(f)}
    for band, preds in P.items():
        pas = "chat_B" if band == "P5" else "plain_B"
        for arm, want in preds:
            r = res[pas][("difference", arm)]
            add(band=band, quantity=f"{arm} ({pas})", value=r["value"], lo95=r["lo95"], hi95=r["hi95"],
                n=r["n"], reading=r["reading"], predicted=want, verdict="right" if r["reading"] == want else "wrong")
    if "plain_G" in res:
        signs = []
        for band in ("P1", "P2", "P3", "P4"):
            for arm, _ in P[band]:
                b, g = float(res["plain_B"][("difference", arm)]["value"]), float(res["plain_G"][("difference", arm)]["value"])
                signs.append((arm, (b > 0) == (g > 0)))
        add(band="P7", quantity="differences of P1-P4 with the same sign under judge G", value=sum(s for _, s in signs),
            n=len(signs), reading="right" if all(s for _, s in signs) else "wrong",
            predicted="all", verdict="; ".join(x for x, s in signs if not s) or "")

    # ---- descriptive: sensitivity of the k=0.5 null, known difference, empties ----------------
    rng = random.Random(a.seed)
    known = float(res["plain_B"][("gain", "opp_r1 - anchor_k0")]["value"])
    half_sel = float(res["plain_B"][("gain", "sel_n64 - sel_n1")]["value"]) / 2
    for arm, ctrl in (("met_k0.5", "anchor_k0"), ("frontkl_4.16", "anchor_k0"), ("win_4.16", "win_0"),
                      ("frontpw_4.16", "win_0"), ("pw_33.27", "win_0")):
        if f"u_{arm}" not in next(iter(new.values())):
            continue
        d = [float(new[p][f"u_{arm}"]) - float(new[p][f"u_{ctrl}"]) for p in new]
        m, se = st.mean(d), st.stdev(d) / len(d) ** 0.5
        xs = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(10000))
        lo90, hi90 = xs[500], xs[9500]
        add(band="descriptive", quantity=f"sensitivity of {arm} - {ctrl}: MDE at 80% power, 90% CI, TOST at the known "
            f"difference {known:+.4f} and at half selection's gain {half_sel:+.4f}", value=round(m, 4),
            lo90=round(lo90, 4), hi90=round(hi90, 4), n=len(d),
            reading=f"MDE80 {Z80 * se:.4f}; equivalent within {known:.3f}: {'yes' if -known < lo90 and hi90 < known else 'no'}; "
                    f"within {half_sel:.3f}: {'yes' if -half_sel < lo90 and hi90 < half_sel else 'no'}")
    for pas in res:
        for (q, arm), r in res[pas].items():
            if q == "level" and r.get("n_empty") not in ("", "0", None):
                add(band="descriptive", quantity=f"{pas}: level of {arm}'s empty texts", value=r["level_empty"],
                    n=r["n_empty"], reading=f"arm level {r['value']}, order consistency {r['consistency']}")
    # ---- post hoc (seventh round, after the registered rows): empties counted as losses ----------
    # Judge G scores an EMPTY answer above the plain-served opponent (0.62-0.75), judge B near parity,
    # so an arm that serves more empties is flattered by G. Every registered difference re-read with an
    # empty text scored 0 in both orders, for every arm in it, controls included.
    EMPTY = "da39a3ee5e6b"                          # sha1("")[:12], the key matched_h2h.py stores
    ctrl = {"sel_n64": "sel_n1", "nonempty": "sel_n1", "blk10n64": "blk200n1", "blk25n64": "blk200n1",
            "metered_k10": "anchor_k0", "met_k0.5": "anchor_k0"}
    for pas in ("plain_B", "plain_G", "chat_B", "chat_G"):
        vf = os.path.join(a.results, f"matched_h2h_verdicts_matched_{pas}.csv")
        pf = os.path.join(a.results, f"matched_h2h_per_prompt_matched_{pas}.csv")
        if not (os.path.exists(vf) and os.path.exists(pf)):
            continue
        empty = {(r["arm"], r["prompt_id"]) for r in rows_of(vf) if r["text_sha"] == EMPTY}
        per = {r["prompt_id"]: r for r in rows_of(pf)}

        def u(arm, p):
            return 0.0 if (arm, p) in empty else float(per[p][f"u_{arm}"])

        for band, preds in P.items():
            if (band == "P5") != pas.startswith("chat"):
                continue
            for arm, _ in preds:
                x, y = arm.split(" - ")
                cx = ctrl.get(x, "winc_0" if x.startswith("winc") else "win_0")
                cy = ctrl.get(y, "winc_0" if y.startswith("winc") else "win_0")
                d = [(u(x, p) - u(cx, p)) - (u(y, p) - u(cy, p)) for p in per]
                lo, hi = paired_boot(d, rng)
                lab = "UNRESOLVED" if lo <= 0 <= hi else ("CONFIRMED" if lo > 0 else "REFUTED")
                add(band="post hoc", quantity=f"{arm} ({pas}), empty texts scored as losses", value=round(st.mean(d), 4),
                    lo95=round(lo, 4), hi95=round(hi, 4), n=len(d), reading=lab)
    with open(os.path.join(a.out, "windowed_meter.csv"), "w", newline="") as fh:
        cols = ["band", "quantity", "value", "lo95", "hi95", "lo90", "hi90", "n", "reading", "predicted", "verdict"]
        w = csv.DictWriter(fh, fieldnames=cols, restval=""); w.writeheader(); w.writerows(out)
    print("wrote", os.path.join(a.out, "windowed_meter.csv"))


if __name__ == "__main__":
    main()
