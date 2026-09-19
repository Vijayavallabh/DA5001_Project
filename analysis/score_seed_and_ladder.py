"""Score feat-138 (seed replication) and feat-140 (single-host ladder) against their committed bands.

Zero GPU. Written BEFORE either arm produced a number and mutation-tested before the data existed
(caution (v)), so the reading rules here are the pre-registrations' and not rules chosen after
seeing the answer.

Constants are IMPORTED from analysis/score_breadth_ladders.py rather than retyped -- MAX_SEED_MOVE,
HALF_WIDTHS_FOR_STABLE, BAND_SEED, N_PROMPTS, GRID. A constant copied into a second file is a
constant that can drift from the one it is supposed to be, which is caution (j) one level up.

WHAT EACH ARM SET ASKS.

  feat-138 re-draws four arms measured on host B, on host B, changing exactly one thing: the seed.
  Its question is whether the VERDICT survives a fresh pool, and its distance bound is a real gate
  because both draws are the same pipeline on the same machine.

  feat-140 measures four anchors on host B that exist only on the local box, so the whole non-Comma
  ladder lives on one machine. Its distance to the local reading is REPORTED and never gated:
  caution (as) established that at bf16 no two hosts compute the same reward and no gate can certify
  that they do.
"""
import argparse
import csv
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.score_breadth_ladders import (BAND_SEED, GRID, HALF_WIDTHS_FOR_STABLE,  # noqa: E402
                                            MAX_SEED_MOVE, N_PROMPTS, JUDGE_B, band, rows)

I1_TOL = 0.05        # mean words GIVEN NON-EMPTY, relative
I2_Z_BLOCK = 3.0     # empty-fraction two-proportion z; feat-138 only (same host, same batch)

# ---- feat-138: seeds 42 43 44 -> 52 53 54, same host, same pipeline ---------------------------
SEED_ARMS = (
    dict(name="comma1t", label="Comma-7B (1T)", ref="comma1thb", new_tag="comma1t64s52",
         ref_delta=0.0970, ref_hw=2.34, predict="CLIMBS"),
    dict(name="pleias350m", label="Pleias-350M", ref="pleias350mhb", new_tag="pleias350m64s52",
         ref_delta=0.0090, ref_hw=0.22, predict="SATURATED BY 8"),
    dict(name="kl3m170m", label="KL3M-170M", ref="kl3m170mhb", new_tag="kl3m170m64s52",
         ref_delta=0.0130, ref_hw=0.38, predict="SATURATED BY 8"),
    dict(name="kl3m520m", label="KL3M-520M", ref="kl3m520mhb", new_tag="kl3m520m64s52",
         ref_delta=-0.0070, ref_hw=0.19, predict="SATURATED BY 8"),
)

# ---- feat-140: the anchors that existed only on the local box ---------------------------------
# Pleias-1.2B/3B and KL3M-1.7B carry a local reading; KL3M-3.7B deliberately carries NONE, because
# feat-135 had not landed when this was registered and a band invented for an unmeasured anchor is
# a band chosen to be met (P4).
LADDER_ARMS = (
    dict(name="pleias12bhb", label="Pleias-1.2B", ref_stats="pleias12b_local",
         new_tag="pleias12bhb64", ref_delta=0.0360, predict="SATURATED BY 8"),
    dict(name="pleias3bhb", label="Pleias-3B", ref_stats="pleias3b_local",
         new_tag="pleias3bhb64", ref_delta=0.0030, predict="SATURATED BY 8"),
    dict(name="kl3m17bhb", label="KL3M-1.7B", ref_stats="kl3m17b_local",
         new_tag="kl3m17bhb64", ref_delta=0.0650, predict="SATURATED BY 8"),
    dict(name="kl3m37bhb", label="KL3M-3.7B", ref_stats="kl3m37b_local",
         new_tag="kl3m37bhb64", ref_delta=None, predict=None),
)


def verdict_of(lo, hi):
    return "CLIMBS" if lo > 0 else "TURNS OVER" if hi < 0 else "SATURATED BY 8"


def two_prop_z(e1, n1, e2, n2):
    """Two-proportion z. Caution (v): an absolute tolerance on a rate is not scale-free -- 0.03 is
    unfalsifiable at KL3M (0.000-0.002) and tighter than the arm-to-arm spread at Comma-7B."""
    if not n1 or not n2:
        return float("nan")
    p1, p2 = e1 / n1, e2 / n2
    p = (e1 + e2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se if se > 0 else 0.0


def stat_row(out, name):
    r = rows(os.path.join(out, f"arm_rank0_stats_{name}.csv"))
    return r[0] if r else None


def i1_length(new, ref):
    """BLOCKS. Mean words GIVEN NON-EMPTY, not the raw mean (feat-136's G0b gated the wrong one)."""
    if new is None or ref is None:
        return None, "rank-0 stats are missing for one side; I1 cannot be evaluated"
    a, b = float(new["mean_words_nonempty"]), float(ref["mean_words_nonempty"])
    rel = abs(a - b) / b if b else float("inf")
    return rel <= I1_TOL, (f"{a:.1f} words given non-empty against {b:.1f} "
                           f"({100 * rel:+.1f}%, tolerance {100 * I1_TOL:.0f}%)")


def i2_empties(new, ref, blocking):
    """Empty fraction, two-proportion z, on the total and per class."""
    if new is None or ref is None:
        return None, "rank-0 stats are missing for one side"
    z = two_prop_z(int(new["n_empty"]), int(new["n_prompts"]),
                   int(ref["n_empty"]), int(ref["n_prompts"]))
    parts = [f"total {float(new['empty_frac']):.3f} vs {float(ref['empty_frac']):.3f} (z={z:+.2f})"]
    worst = abs(z)
    for cls in ("neutral", "creative", "factual"):
        if f"empty_{cls}" in new and f"empty_{cls}" in ref and new.get(f"n_{cls}"):
            zc = two_prop_z(int(new[f"empty_{cls}"]), int(new[f"n_{cls}"]),
                            int(ref[f"empty_{cls}"]), int(ref[f"n_{cls}"]))
            parts.append(f"{cls} z={zc:+.2f}")
            worst = max(worst, abs(zc))
    ok = (worst <= I2_Z_BLOCK) if blocking else None
    return ok, "; ".join(parts) + (f"  [worst |z|={worst:.2f}]" if blocking else "  [reported only]")


def i3_coverage(summary, per):
    if summary is None or per is None:
        return False, "the scoring CSVs are not there yet; the arm has not finished"
    jb = [r for r in summary if JUDGE_B in r["judge"]]
    if not jb:
        return False, "judge B is absent"
    have = sorted({int(float(r["n"])) for r in jb})
    if have != sorted(GRID):
        return False, f"grid is {have}, not {sorted(GRID)}"
    pp = [r for r in per if JUDGE_B in r["judge"]]
    if len(pp) != N_PROMPTS:
        return False, f"{len(pp)} prompts, not {N_PROMPTS}"
    return True, f"grid {have} on {len(pp)} prompts"


def i4_pipeline(new, ref):
    """BLOCKS (feat-140). Caution (at) made into a gate: both sides must be the same pipeline."""
    if new is None or ref is None:
        return None, "rank-0 stats are missing for one side"
    a = (new.get("target_model"), new.get("anchor_model"))
    b = (ref.get("target_model"), ref.get("anchor_model"))
    if a[0] != a[1] or b[0] != b[1]:
        return False, f"one side is not self-paired: this {a}, reference {b}"
    return a[0] == b[0], f"both self-paired on {a[0]}" if a[0] == b[0] else \
        f"PIPELINE MISMATCH: this {a[0]}, reference {b[0]}"


def score_one(out, arm, kind, recs):
    label = arm["label"]
    print(f"\n--- {label}  ({arm['name']}, {kind})")
    summary = rows(os.path.join(out, f"selection_scaling_{arm['new_tag']}.csv"))
    per = rows(os.path.join(out, f"selection_scaling_per_prompt_{arm['new_tag']}.csv"))
    ok3, msg3 = i3_coverage(summary, per)
    print(f"  I3 coverage: {'PASS' if ok3 else 'FAIL'} -- {msg3}")
    if not ok3:
        print("  NOT SCORED. The band is not computed, and is not computed 'just to see'.")
        recs.append(dict(arm=arm["name"], set=kind, verdict="NOT SCORED", note=msg3))
        return

    new_s = stat_row(out, arm["name"] if kind == "feat-138" else arm["name"])
    ref_s = stat_row(out, arm.get("ref") or arm.get("ref_stats"))
    if kind == "feat-140":
        ok4, msg4 = i4_pipeline(new_s, ref_s)
        print(f"  I4 pipeline: {'PASS' if ok4 else 'FAIL' if ok4 is False else 'n/a'} -- {msg4}")
        if ok4 is False:
            print("  NOT SCORED -- INVALID rather than failed (caution (w)/(at)).")
            recs.append(dict(arm=arm["name"], set=kind, verdict="NOT SCORED", note=msg4))
            return

    ok1, msg1 = i1_length(new_s, ref_s)
    print(f"  I1 length:   {'PASS' if ok1 else 'FAIL' if ok1 is False else 'n/a'} -- {msg1}")
    if ok1 is False:
        print("  NOT SCORED: length given non-empty disagrees, which is the statistic a wrong")
        print("  model, corpus or truncation moves. The band is not computed.")
        recs.append(dict(arm=arm["name"], set=kind, verdict="NOT SCORED", note=msg1))
        return

    ok2, msg2 = i2_empties(new_s, ref_s, blocking=(kind == "feat-138"))
    print(f"  I2 empties:  {'PASS' if ok2 else 'FAIL' if ok2 is False else 'REPORTED'} -- {msg2}")
    if ok2 is False:
        print("  NOT SCORED: with host, pipeline and batch size all held, only the seed moves this,")
        print("  so a shift this large means something else changed.")
        recs.append(dict(arm=arm["name"], set=kind, verdict="NOT SCORED", note=msg2))
        return

    pp = [r for r in per if JUDGE_B in r["judge"]]
    g, lo, hi, hw, ratio, n = band(pp)
    v = verdict_of(lo, hi)
    marginal = ratio < HALF_WIDTHS_FOR_STABLE
    print(f"  paired g(64) - g(8) over {n} prompts: {g:+.4f} [{lo:+.4f}, {hi:+.4f}]  -> {v}")
    print(f"  distance from zero: {ratio:.2f} half-widths "
          f"({'MARGINAL' if marginal else 'stable'} at {HALF_WIDTHS_FOR_STABLE:.1f})")

    agrees = move = ""
    if arm["predict"] is not None:
        agrees = "YES" if v == arm["predict"] else "NO"
        print(f"  predicted {arm['predict']} -> read {v}: verdict agrees? {agrees}")
        if agrees == "NO":
            print("  THE PREDICTION FAILED. That is the finding and is reported as such.")
    else:
        print("  no verdict was predicted for this anchor (P4): it had no prior when registered.")

    if arm["ref_delta"] is not None:
        move = abs(g - arm["ref_delta"])
        if kind == "feat-138":
            within = move <= MAX_SEED_MOVE
            print(f"  G2 distance: |{g:+.4f} - {arm['ref_delta']:+.4f}| = {move:.4f} "
                  f"vs {MAX_SEED_MOVE:.4f} -> {'WITHIN' if within else 'BEYOND'} "
                  f"the largest seed move on record")
            if not within:
                print("    Beyond every seed move on record. Reported as a distance that exceeded")
                print("    the range, separately from whether the verdict agreed (G2 never")
                print("    overrides G1).")
        else:
            print(f"  cross-host distance (REPORTED, never gated -- caution (as)): "
                  f"|{g:+.4f} - {arm['ref_delta']:+.4f}| = {move:.4f}, context {MAX_SEED_MOVE:.4f}")

    recs.append(dict(arm=arm["name"], set=kind, delta=round(g, 4), lo95=round(lo, 4),
                     hi95=round(hi, 4), half_widths=round(ratio, 2), verdict=v,
                     predicted=arm["predict"] or "", verdict_agrees=agrees,
                     ref_delta=arm["ref_delta"] if arm["ref_delta"] is not None else "",
                     move=(round(move, 4) if move != "" else ""),
                     marginal=("yes" if marginal else "no"), note=""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = a.out
    print("feat-138 (seed replication, same host) and feat-140 (the non-Comma ladder on one host)")
    print(f"  constants imported, not retyped: MAX_SEED_MOVE={MAX_SEED_MOVE}, "
          f"boundary={HALF_WIDTHS_FOR_STABLE}, BAND_SEED={BAND_SEED}")
    recs = []
    print("\n================ feat-138: does the verdict survive a fresh draw? ================")
    for arm in SEED_ARMS:
        score_one(out, arm, "feat-138", recs)
    print("\n================ feat-140: the non-Comma ladder on ONE host ================")
    for arm in LADDER_ARMS:
        score_one(out, arm, "feat-140", recs)

    scored = [r for r in recs if r["verdict"] != "NOT SCORED"]
    print("\n=== H2 on a single host, read by the committed rule ===")
    nc = [r for r in scored if r["set"] == "feat-140" or r["arm"] != "comma1t"]
    clears = [r for r in nc if r["arm"] != "comma1t" and r["half_widths"] >= HALF_WIDTHS_FOR_STABLE
              and r["verdict"] == "CLIMBS"]
    if not scored:
        print("  nothing is scored yet; H2 is not read.")
    elif clears:
        print(f"  H2 REFUTED: {[r['arm'] for r in clears]} clears "
              f"{HALF_WIDTHS_FOR_STABLE} half-widths and CLIMBS. That is the finding.")
    else:
        print(f"  H2 survives: of {len(nc)} non-Comma readings scored here, none clears "
              f"{HALF_WIDTHS_FOR_STABLE} half-widths while climbing.")

    p = os.path.join(out, "seed_and_ladder_scoring.csv")
    cols = ["arm", "set", "delta", "lo95", "hi95", "half_widths", "verdict", "predicted",
            "verdict_agrees", "ref_delta", "move", "marginal", "note"]
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow(r)
    print(f"\nwrote {p} ({len(scored)} of {len(recs)} arms scored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
