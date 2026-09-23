"""feat-172: score results/onset_prediction_offsupport_ladder.md. Gate first, then bands.

Written 2026-09-23 while the reward passes were still running, so no number from either arm existed.

  Arm A  off-support: the audited anchor at n=256 on AlpacaEval's 805 prompts.
  Arm B  on-support:  the same on our own 850 prompts (registered 2026-09-22 12:20).

THE GATE (registered): ranks 0-63 of each n=256 reward cache must be BIT-IDENTICAL to the committed
n=64 cache it extends -- mixpow_rewards64.csv (51,520 floats) for A, wscope_rewards64_a.csv (54,400)
for B. analysis/score_n128.py:reward_gate is feat-129/134's gate, reused rather than re-written. A
failure is INAPPLICABLE, not a result, and this script then prints no band.

THE BANDS, as registered:
  B1 (A)  D3 against the k=1 metered arm at n=128 and 256, order-averaged, judge B. CATCHES if D3 >= 0
          at either; CLOSES if still negative but |D3(256)| < half the committed n=64 |D3|; CEILING if
          |D3(256)| has not fallen by at least a quarter. The registration leaves the gap between the
          last two unnamed; it is printed as BETWEEN and labelled unregistered.
  B2 (A)  paired g(256)-g(128) within the pass: STILL CLIMBING if > 0 with its interval excluding
          zero, SATURATED if the interval contains zero. g(128)-g(64) is reported beside it. A
          negative interval clear of zero has no registered name and is printed as TURNS OVER,
          labelled unregistered.
  B3 (A)  log n at the crossing or at the grid's end, against the metered arm's realised spend on
          this corpus (utility_price.spends on its own trajectories), as a ratio.
  B5 (B)  as B2, on support.
  B6      the two shapes side by side, never their levels (caution (ap)).

Reads the committed n=64 D3 from results/order_averaged_h2h__mixpowk_judgeB.csv (caution (at)).
Run on host B, where output/mixpow/conc_k10 lives (B3 needs it; everything else is in results/).

Usage: .venv/bin/python analysis/score_offsupport.py --out results
"""
import argparse
import csv
import glob
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.opponent_strength import d3_row  # noqa: E402
from analysis.score_n128 import reward_gate, rows  # noqa: E402
from analysis.selection_decoding import boot_mean  # noqa: E402

JUDGE_B = "Phi-3.5-mini-instruct"
ARMS = {"A": dict(name="off-support, AlpacaEval", new="offsup_rewards256.csv",
                  old="mixpow_rewards64.csv", tag="_offsup"),
        "B": dict(name="on-support, our 850 prompts", new="onsup_rewards256.csv",
                  old="wscope_rewards64_a.csv", tag="_onsup")}
METERED_DIR = "output/mixpow/conc_k10"     # holds k=1, the binding cell (names are swapped there)


def shape(res, tag):
    """Paired g(128)-g(64) and g(256)-g(128) under judge B, within one pass."""
    per = rows(os.path.join(res, f"selection_scaling_per_prompt{tag}.csv"))
    if per is None:
        return None
    pp = [r for r in per if JUDGE_B in r["judge"]]
    assert pp and "u_n256" in pp[0], "the pass did not reach n=256"
    out = {}
    for hi, lo in ((128, 64), (256, 128)):
        d = [float(r[f"u_n{hi}"]) - float(r[f"u_n{lo}"]) for r in pp]
        a, b = boot_mean(d, random.Random(20260923))
        out[(hi, lo)] = (sum(d) / len(d), a, b, len(d))
    g, a, b, _ = out[(256, 128)]
    out["verdict"] = ("STILL CLIMBING" if a > 0 else "SATURATED" if a <= 0 <= b
                      else "TURNS OVER (unregistered)")
    return out


def _band(arm, quantity, d):
    a, b = boot_mean(d, random.Random(20260923))
    return dict(arm=arm, quantity=quantity, value=round(sum(d) / len(d), 4), lo95=round(a, 4),
                hi95=round(b, 4), n=len(d), reading="POST HOC")


def post_hoc(res):
    """NOT REGISTERED. Added 2026-09-23 after the bands were read, and every row says POST HOC.

    Both registered climbs sit about one half-width from zero, the regime where feat-131's paired
    difference did not replicate (caution (ap)), so two checks are reported beside them:
      g(256)-g(64) single-order per arm -- the two doublings from the paper's n, not the last one;
      Arm A under order averaging, which draws nothing and so pairs exactly across the committed
        n=64 pass and this arm's n=128/256 passes -- refused unless u_sel_n1 agrees on every prompt.
    """
    out = []
    for arm, A in ARMS.items():
        per = rows(os.path.join(res, f"selection_scaling_per_prompt{A['tag']}.csv"))
        if per:
            pp = [r for r in per if JUDGE_B in r["judge"]]
            out.append(_band(arm, "g(256)-g(64)", [float(r["u_n256"]) - float(r["u_n64"]) for r in pp]))
    files = {64: "mixpowk_judgeB", 128: "offsup_n128", 256: "offsup_n256"}
    per = {n: rows(os.path.join(res, f"order_averaged_h2h_per_prompt__{t}.csv")) for n, t in files.items()}
    if all(per.values()):
        by = {n: {r["prompt_id"]: r for r in v} for n, v in per.items()}
        pids = sorted(set.intersection(*(set(v) for v in by.values())))
        assert all(len({by[n][p]["u_sel_n1"] for n in by}) == 1 for p in pids), \
            "u_sel_n1 differs across the order-averaged passes; they do not pair"
        for hi, lo in ((128, 64), (256, 128), (256, 64)):
            out.append(_band("A", f"g({hi})-g({lo}) order-averaged",
                             [float(by[hi][p]["gain_sel"]) - float(by[lo][p]["gain_sel"]) for p in pids]))
    return out


def realised_spend():
    """Mean realised sequence divergence of the k=1 metered arm on AlpacaEval, read from its own
    trajectories; None off host B."""
    files = glob.glob(os.path.join(METERED_DIR, "trajectories_k*_factual.jsonl"))
    if not files:
        return None
    ks = {os.path.basename(f)[len("trajectories_k"):-len("_factual.jsonl")] for f in files}
    ks.discard("-1")
    ks.discard("0")
    assert len(ks) == 1, f"{METERED_DIR} holds several budgets: {sorted(ks)}"
    from analysis.utility_price import spends
    s = spends([METERED_DIR], ks.pop())
    return sum(s) / len(s) if s else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    res, out = a.results, []

    gates = {}
    for arm, A in ARMS.items():
        ok, bad = reward_gate(rows(os.path.join(res, A["new"])), rows(os.path.join(res, A["old"])))
        gates[arm] = ok
        status = "NOT YET" if ok is None else "PASS" if ok else "FAIL -> INAPPLICABLE"
        print(f"G  Arm {arm} ({A['name']}): ranks 0-63 of {A['new']} == {A['old']}: {status}")
        if ok is False:
            for x in bad[:5]:
                print(f"     {x}")

    shapes = {}
    for arm, A in ARMS.items():
        if not gates[arm]:
            print(f"   Arm {arm}: no band is read")
            continue
        s = shape(res, A["tag"])
        if s is None:
            print(f"   Arm {arm}: gate passed; not judged yet")
            continue
        shapes[arm] = s
        band = "B2" if arm == "A" else "B5"
        for (hi, lo), (g, l, h, n) in ((k, v) for k, v in s.items() if k != "verdict"):
            print(f"{band} Arm {arm}: g({hi}) - g({lo}) = {g:+.4f} [{l:+.4f}, {h:+.4f}] over {n}")
            out.append(dict(arm=arm, quantity=f"g({hi})-g({lo})", value=round(g, 4),
                            lo95=round(l, 4), hi95=round(h, 4), n=n, reading=s["verdict"]
                            if (hi, lo) == (256, 128) else ""))
        print(f"{band} Arm {arm}: **{s['verdict']}**")

    if gates["A"]:
        ref = d3_row("__mixpowk_judgeB")
        d = {n: d3_row(f"__offsup_n{n}") for n in (128, 256)}
        if ref is None:
            print("B1 Arm A: the committed n=64 reference, order_averaged_h2h__mixpowk_judgeB.csv, "
                  "is missing; B1 cannot be read")
        elif all(d.values()):
            d64 = float(ref["value"])
            v = {n: float(r["value"]) for n, r in d.items()}
            if any(x >= 0 for x in v.values()):
                b1 = "CATCHES"
            elif abs(v[256]) < abs(d64) / 2:
                b1 = "CLOSES"
            elif abs(v[256]) > 0.75 * abs(d64):
                b1 = "CEILING"
            else:
                b1 = "BETWEEN (unregistered gap)"
            for n, r in d.items():
                print(f"B1 Arm A: D3 at n={n} = {v[n]:+.4f} [{float(r['lo95']):+.4f}, "
                      f"{float(r['hi95']):+.4f}]   (n=64 on record: {d64:+.4f})")
                out.append(dict(arm="A", quantity=f"D3 n={n}", value=v[n],
                                lo95=float(r["lo95"]), hi95=float(r["hi95"]), n=r["n"],
                                reading=b1 if n == 256 else ""))
            print(f"B1 Arm A: **{b1}**")
            cross = min((n for n in (128, 256) if v[n] >= 0), default=256)
            spend = realised_spend()
            cert = math.log(cross)
            print(f"B3 Arm A: certificate log {cross} = {cert:.3f} nats"
                  + (f" against the meter's realised {spend:.2f}, a ratio of {spend / cert:.1f}"
                     if spend else " (meter's realised spend readable only on host B)"))
            out.append(dict(arm="A", quantity=f"certificate log {cross}", value=round(cert, 4),
                            lo95="", hi95="", n="", reading=(round(spend, 4) if spend else "")))
        else:
            print("B1 Arm A: gate passed; the n=128/256 head-to-heads are not judged yet")

    if len(shapes) == 2:
        print(f"B6: off-support {shapes['A']['verdict']}, on-support {shapes['B']['verdict']} "
              "(shapes only; levels are never set against each other)")

    if shapes:
        for r in post_hoc(res):
            print(f"POST HOC Arm {r['arm']}: {r['quantity']} = {r['value']:+.4f} "
                  f"[{r['lo95']:+.4f}, {r['hi95']:+.4f}] over {r['n']}")
            out.append(r)

    if out:
        path = os.path.join(a.out, "offsupport_ladder.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
