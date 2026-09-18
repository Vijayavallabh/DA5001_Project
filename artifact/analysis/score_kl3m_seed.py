"""Score results/onset_prediction_kl3m_seed.md against its committed bands. Zero GPU.

Written BEFORE the arm produced a number, so the reading rules here are the pre-registration's
rather than rules chosen after seeing the answer.

The question: feat-130 read PARTIAL on the strength of exactly one climbing anchor, KL3M-1.7B at
+0.0650 [+0.0270, +0.1030], which also carries a reduced warrant. This re-draws it with disjoint
seeds and asks whether CLIMBS survives.

THERE IS DELIBERATELY NO BIT-IDENTITY GATE. Both of this session's earlier pre-registrations wrote a
reproduction gate that could not pass (cautions (ap) and (u)); here one would be incoherent rather
than merely wrong, because the arm is an independent draw BY CONSTRUCTION and nothing is supposed to
match. The integrity checks are distributional and were fixed in the pre-registration: the same 500
prompts, and an n=1 empty fraction within 0.03 of the committed arm's 0.002. Judged LEVELS are never
compared across passes -- that is the error caution (ap) exists to prevent -- and the band is the
PAIRED difference computed within one pass on both sides.

Usage: .venv/bin/python analysis/score_kl3m_seed.py --out results
"""
import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_decoding import boot_mean, load_candidates  # noqa: E402

JUDGE_B = "Phi-3.5-mini-instruct"
ORIG_TAG = "_kl3m17b64"          # feat-130's arm, seeds 42 43 44
REP_TAG = "_kl3m17bseed52"       # this arm, seeds 52 53 54
COMMITTED_ORIG = (0.0650, 0.0270, 0.1030)
COMMITTED_EMPTY_FRAC = 0.002     # selection_breadth.csv, KL3M-1.7B, empty_frac_n1
EMPTY_TOL = 0.03
# The precedent the pre-registration quotes: the audited anchor's own seed replication left the
# paired difference at +0.0880 under both seed sets while its levels moved.
PRECEDENT = 0.0000


def rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8"))) if os.path.exists(path) else None


def paired(out, tag, seed=20260918):
    """The paired g(64) - g(8) within ONE pass. Both gains subtract the same u_n1, so the
    difference is exactly u_n64 - u_n8 and no cross-pass quantity enters."""
    per = rows(os.path.join(out, f"selection_scaling_per_prompt{tag}.csv"))
    if not per:
        return None
    pp = [r for r in per if JUDGE_B in r["judge"]]
    if not pp or "u_n64" not in pp[0]:
        return None
    d = [float(r["u_n64"]) - float(r["u_n8"]) for r in pp]
    g = sum(d) / len(d)
    lo, hi = boot_mean(d, random.Random(seed))
    return g, lo, hi, len(d)


def empty_fraction(gen_dir):
    """Share of prompts whose DRAW 0 is blank. load_candidates sorts by seed and the seeds are
    (h << 16) | j, so entry 0 is draw 0 -- the string the n=1 arm serves."""
    if not os.path.isdir(gen_dir):
        return None
    cands = load_candidates(gen_dir)
    if not cands:
        return None
    blank = sum(1 for v in cands.values() if not v[0][3].strip())
    return blank / len(cands), len(cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--gen-dir", default="output/phase5/sel_kl3m17b_64_seed52")
    a = ap.parse_args()

    print("results/onset_prediction_kl3m_seed.md -- does KL3M-1.7B's CLIMBS survive a fresh draw?")
    print(f"  feat-130 on record (seeds 42 43 44): {COMMITTED_ORIG[0]:+.4f} "
          f"[{COMMITTED_ORIG[1]:+.4f}, {COMMITTED_ORIG[2]:+.4f}]")
    print(f"  precedent: the audited anchor's own seed replication moved its LEVELS and left its "
          f"paired difference unchanged to four decimals (|difference| = {PRECEDENT:.4f})")

    print("\n=== integrity checks (distributional; there is no bit-identity gate, by design) ===")
    ef = empty_fraction(a.gen_dir)
    if ef is None:
        print(f"  NOT YET SCOREABLE: {a.gen_dir} has no merged generations")
        return
    frac, n_prompts = ef
    ok_prompts = n_prompts == 500
    ok_empty = abs(frac - COMMITTED_EMPTY_FRAC) <= EMPTY_TOL
    print(f"  prompts: {n_prompts} ({'PASS' if ok_prompts else 'FAIL -- expected 500'})")
    print(f"  n=1 empty fraction: {frac:.4f} against the committed {COMMITTED_EMPTY_FRAC:.4f}, "
          f"tolerance {EMPTY_TOL} -> {'PASS' if ok_empty else 'FAIL'}")
    if not (ok_prompts and ok_empty):
        print("  Per the pre-registration, the arm does not clear its integrity checks. Chase it.")
        return

    rep = paired(a.out, REP_TAG)
    if rep is None:
        print(f"\n  NOT YET SCOREABLE: selection_scaling_per_prompt{REP_TAG}.csv is absent or has "
              f"no n=64 column.")
        return
    g, lo, hi, n = rep
    verdict = ("REPLICATES" if lo > 0 else "INVERTS" if hi < 0 else "DOES NOT REPLICATE")
    print(f"\n=== the committed band ===")
    print(f"  paired g(64) - g(8) over {n} prompts: {g:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print(f"  VERDICT: {verdict}")

    orig = paired(a.out, ORIG_TAG)
    dist = abs(g - orig[0]) if orig else None
    if orig:
        print(f"\n=== committed secondary ===")
        print(f"  feat-130 recomputed here: {orig[0]:+.4f} [{orig[1]:+.4f}, {orig[2]:+.4f}]")
        print(f"  |D_rep - D_orig| = {dist:.4f}, against the precedent's {PRECEDENT:.4f}")
    new = rows(os.path.join(a.out, f"selection_scaling{REP_TAG}.csv")) or []
    for r in sorted(new, key=lambda r: (r["judge"], int(float(r["n"])))):
        print(f"    {r['judge'][:26]:<26} n={int(float(r['n'])):<3} gain={float(r['gain']):+.4f}")

    out = dict(arm="KL3M-1.7B seed replication", seeds="52 53 54", n_prompts=n,
               gain_diff=round(g, 4), lo95=round(lo, 4), hi95=round(hi, 4), verdict=verdict,
               orig_gain_diff=round(orig[0], 4) if orig else None,
               distance=round(dist, 4) if dist is not None else None,
               empty_frac_n1=round(frac, 4), integrity="PASS")
    p = os.path.join(a.out, "kl3m_seed_scoring.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out)); w.writeheader(); w.writerow(out)
    print(f"\nwrote {p}")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
