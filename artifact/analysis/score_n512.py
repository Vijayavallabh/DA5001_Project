"""feat-181: score results/onset_prediction_n512_ladder.md. Gates first, then bands.

Written 2026-09-23 while the extension draws were still generating, so no n > 256 number existed.

  G1   ranks 0-255 of each n=512 reward cache bit-identical to the committed n=256 cache
       (analysis/score_n128.py:reward_gate with top=256). A failure is INAPPLICABLE.
  G2'  every prompt holds ranks 0..511 exactly once in its cache (G2 proper, on the trajectory files,
       ran on host B inside scripts/run_n512_post.sh; this re-reads it where the cache is committed).
  C2   paired g(512)-g(256), judge B, within the n=512 pass: STILL CLIMBING / SATURATED / TURNS OVER.
  C3   paired g(512)-g(64):  CLIMBS PAST 64 / NOT RESOLVED / FALLS.
       Both carry point / half-width, and below 1.7 the label MARGINAL (registered, caution (ap)).
  C1   (Arm A) order-averaged D3 at n=512: CATCHES / STILL BEHIND / UNRESOLVED; beside it the
       order-averaged g(512)-g(256) and g(512)-g(64), paired exactly across passes.
  C4   log 512 against the meter's realised spend, read from feat-172's committed B3 row.

Usage: .venv/bin/python analysis/score_n512.py --out results
"""
import argparse
import csv
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.opponent_strength import d3_row  # noqa: E402
from analysis.score_n128 import reward_gate, rows  # noqa: E402
from analysis.selection_decoding import boot_mean  # noqa: E402

JUDGE_B = "Phi-3.5-mini-instruct"
ARMS = {"A": dict(name="off-support, AlpacaEval", tag="n512a", old="offsup_rewards256.csv"),
        "B": dict(name="on-support, our 850 prompts", tag="n512b", old="onsup_rewards256.csv")}
MARGINAL = 1.7


def band(d):
    a, b = boot_mean(d, random.Random(20260923))
    g = sum(d) / len(d)
    hw = (b - a) / 2
    return g, a, b, (g / hw if hw > 0 else float("inf"))


def label(lo, hi, up, flat, down, ratio):
    word = up if lo > 0 else down if hi < 0 else flat
    return word + (" (MARGINAL)" if word != flat and abs(ratio) < MARGINAL else "")


def g2_cache(cache, n=512):
    by = {}
    for r in cache:
        by.setdefault(r["prompt_id"], []).append(int(r["rank"]))
    return [p for p, ks in by.items() if sorted(ks) != list(range(n))]


def ladder(res, tag):
    per = rows(os.path.join(res, f"selection_scaling_per_prompt_{tag}.csv"))
    if per is None:
        return None
    pp = [r for r in per if JUDGE_B in r["judge"]]
    assert pp and "u_n512" in pp[0], "the pass did not reach n=512"
    out = {}
    for hi, lo, names in ((512, 256, ("STILL CLIMBING", "SATURATED", "TURNS OVER")),
                          (512, 64, ("CLIMBS PAST 64", "NOT RESOLVED", "FALLS"))):
        g, a, b, ratio = band([float(r[f"u_n{hi}"]) - float(r[f"u_n{lo}"]) for r in pp])
        out[(hi, lo)] = (g, a, b, ratio, len(pp), label(a, b, names[0], names[1], names[2], ratio))
    return out


def order_averaged(res):
    files = {64: "mixpowk_judgeB", 256: "offsup_n256", 512: "n512a_n512"}
    per = {n: rows(os.path.join(res, f"order_averaged_h2h_per_prompt__{t}.csv")) for n, t in files.items()}
    if not all(per.values()):
        return None
    by = {n: {r["prompt_id"]: r for r in v} for n, v in per.items()}
    pids = sorted(set.intersection(*(set(v) for v in by.values())))
    assert all(len({by[n][p]["u_sel_n1"] for n in by}) == 1 for p in pids), \
        "u_sel_n1 differs across the order-averaged passes; they do not pair"
    return {(hi, lo): band([float(by[hi][p]["gain_sel"]) - float(by[lo][p]["gain_sel"]) for p in pids])
            + (len(pids),) for hi, lo in ((512, 256), (512, 64))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    res, out, ok = a.results, [], {}

    for arm, A in ARMS.items():
        new = rows(os.path.join(res, f"{A['tag']}_rewards512.csv"))
        g1, bad = reward_gate(new, rows(os.path.join(res, A["old"])), top=256)
        g2 = g2_cache(new) if new else None
        ok[arm] = bool(g1) and g2 == []
        print(f"G1 Arm {arm}: {'NOT YET' if g1 is None else 'PASS' if g1 else 'FAIL -> INAPPLICABLE'}"
              f"   G2' {'NOT YET' if g2 is None else 'PASS' if not g2 else f'FAIL on {len(g2)} prompts'}")

    for arm, A in ARMS.items():
        if not ok[arm]:
            print(f"   Arm {arm}: no band is read")
            continue
        lad = ladder(res, A["tag"])
        if lad is None:
            print(f"   Arm {arm}: gates passed; not judged yet")
            continue
        for (hi, lo), (g, l, h, ratio, n, word) in lad.items():
            b = "C2" if lo == 256 else "C3"
            print(f"{b} Arm {arm}: g({hi}) - g({lo}) = {g:+.4f} [{l:+.4f}, {h:+.4f}], "
                  f"{ratio:.2f} half-widths -> **{word}**")
            out.append(dict(arm=arm, band=b, quantity=f"g({hi})-g({lo})", value=round(g, 4),
                            lo95=round(l, 4), hi95=round(h, 4), half_widths=round(ratio, 2), n=n,
                            reading=word))

    if ok["A"]:
        d = d3_row("__n512a_n512")
        if d is None:
            print("C1 Arm A: gates passed; the n=512 head-to-head is not judged yet")
        else:
            v, l, h = float(d["value"]), float(d["lo95"]), float(d["hi95"])
            word = "CATCHES" if v >= 0 else "STILL BEHIND" if h < 0 else "UNRESOLVED"
            print(f"C1 Arm A: D3 at n=512 = {v:+.4f} [{l:+.4f}, {h:+.4f}] -> **{word}**")
            out.append(dict(arm="A", band="C1", quantity="D3 n=512", value=v, lo95=l, hi95=h,
                            half_widths="", n=d["n"], reading=word))
            oa = order_averaged(res)
            for (hi, lo), (g, l2, h2, ratio, n) in (oa or {}).items():
                print(f"C1 Arm A: order-averaged g({hi}) - g({lo}) = {g:+.4f} [{l2:+.4f}, {h2:+.4f}]")
                out.append(dict(arm="A", band="C1", quantity=f"g({hi})-g({lo}) order-averaged",
                                value=round(g, 4), lo95=round(l2, 4), hi95=round(h2, 4),
                                half_widths=round(ratio, 2), n=n, reading=""))
            b3 = next((r for r in rows(os.path.join(res, "offsupport_ladder.csv")) or []
                       if r["quantity"] == "certificate log 256"), None)
            if b3 and b3["reading"]:
                spend, cert = float(b3["reading"]), math.log(512)
                print(f"C4 Arm A: log 512 = {cert:.3f} nats against the meter's realised {spend:.2f}, "
                      f"a ratio of {spend / cert:.1f}")
                out.append(dict(arm="A", band="C4", quantity="certificate log 512", value=round(cert, 4),
                                lo95="", hi95="", half_widths="", n="", reading=round(spend / cert, 2)))

    if out:
        path = os.path.join(a.out, "n512_ladder.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
