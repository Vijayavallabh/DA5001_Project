"""Score a seed replication against its committed bands. Zero GPU.

Two arms, ONE rule:

  --anchor kl3m17b  feat-131, results/onset_prediction_kl3m_seed.md. feat-130 read PARTIAL on the
                    strength of exactly one climbing anchor, KL3M-1.7B at +0.0650 [+0.0270,
                    +0.1030], which also carried a reduced warrant. Re-drawn with disjoint seeds:
                    DOES NOT REPLICATE.
  --anchor comma7b  feat-132, results/onset_prediction_comma7b_seed.md. After feat-131 the
                    breadth-at-n=64 claim rests on TinyComma and Comma-7B; Comma-7B has never been
                    re-drawn, and at 2.43 interval half-widths it is also the out-of-sample test of
                    the stability criterion feat-131 produced.

Each anchor's constants were committed in its own pre-registration BEFORE that arm produced a
number, so the reading rules here are the pre-registrations' rather than rules chosen after seeing
the answer. Scoring both through one code path is deliberate: it is what makes the two readings
comparable rather than merely similar.

THERE IS DELIBERATELY NO BIT-IDENTITY GATE. Both of this session's earlier pre-registrations wrote a
reproduction gate that could not pass (cautions (ap) and (u)); here one would be incoherent rather
than merely wrong, because the arm is an independent draw BY CONSTRUCTION and nothing is supposed to
match. The integrity checks are distributional and were fixed in the pre-registration: the same 500
prompts, and an n=1 empty fraction within 0.03 of the committed arm's own. Judged LEVELS are never
compared across passes -- that is the error caution (ap) exists to prevent -- and the band is the
PAIRED difference computed within one pass on both sides.

Usage: .venv/bin/python analysis/score_kl3m_seed.py --anchor comma7b --out results
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
EMPTY_TOL = 0.03

# One rule, two arms. feat-132 re-draws Comma-7B the same way feat-131 re-drew KL3M-1.7B, and the
# strongest available design is for BOTH to be read by this code path rather than by two scorers
# that merely look alike. So the anchor's constants move into a table and everything below is
# unchanged; `--anchor kl3m17b` is the default, so feat-131's committed invocation is unchanged.
# Its output gains two derived columns, `ratio_orig` and `ratio_rep` -- the criterion's own quantity,
# computed from numbers already in that file -- and no band, verdict or measured value moves. The
# re-run that confirmed this is in the commit that added feat-132.
#
# `precedent` is what the SAME comparison gave at the audited anchor, whose two seed draws left the
# paired difference at +0.0880 to four decimals while its levels moved. For feat-132 the KL3M
# non-replication is a second precedent and is printed beside it.
ANCHORS = {
    "kl3m17b": dict(
        name="KL3M-1.7B seed replication",
        orig_tag="_kl3m17b64",              # feat-130's arm, seeds 42 43 44
        rep_tag="_kl3m17bseed52",           # feat-131, seeds 52 53 54
        committed_orig=(0.0650, 0.0270, 0.1030),
        # Left exactly as feat-131 committed it, from selection_breadth.csv's n=8 column. feat-132's
        # mutation test showed that is the wrong protocol -- the reference must be measured on the
        # arm being replicated -- and this anchor passed only because its two arms agree (0.002
        # against 0.000). A committed band is not edited after the fact; see the comma7b entry.
        committed_empty_frac=0.002,
        gen_dir="output/phase5/sel_kl3m17b_64_seed52",
        out_csv="kl3m_seed_scoring.csv",
        log="results/onset_prediction_kl3m_seed.md",
        question="does KL3M-1.7B's CLIMBS survive a fresh draw?",
        precedents=(("audited anchor", 0.0000),),
    ),
    "comma7b": dict(
        name="Comma-7B seed replication",
        orig_tag="_comma7b64",              # the n=64 arm on record, seeds 42 43 44
        rep_tag="_comma7bseed52",           # feat-132, seeds 52 53 54
        committed_orig=(0.1010, 0.0590, 0.1420),
        committed_empty_frac=0.094,         # the n=64 arm ON RECORD, measured; NOT the n=8 column (caution (v))
        gen_dir="output/phase5/sel_comma7b_64_seed52",
        out_csv="comma7b_seed_scoring.csv",
        log="results/onset_prediction_comma7b_seed.md",
        question="does Comma-7B's climb to n=64 survive a fresh draw?",
        precedents=(("audited anchor", 0.0000), ("KL3M-1.7B", 0.0610)),
    ),
}


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
    ap.add_argument("--anchor", default="kl3m17b", choices=sorted(ANCHORS),
                    help="which seed replication to score; the default is feat-131's")
    ap.add_argument("--gen-dir", default=None,
                    help="override the anchor's merged generation directory")
    a = ap.parse_args()
    A = ANCHORS[a.anchor]
    gen_dir = a.gen_dir or A["gen_dir"]
    orig_band = A["committed_orig"]

    print(f"{A['log']} -- {A['question']}")
    print(f"  the arm on record (seeds 42 43 44): {orig_band[0]:+.4f} "
          f"[{orig_band[1]:+.4f}, {orig_band[2]:+.4f}]")
    for label, d in A["precedents"]:
        print(f"  precedent, the same comparison at the {label}: |difference| = {d:.4f}")

    print("\n=== integrity checks (distributional; there is no bit-identity gate, by design) ===")
    ef = empty_fraction(gen_dir)
    if ef is None:
        print(f"  NOT YET SCOREABLE: {gen_dir} has no merged generations")
        return
    frac, n_prompts = ef
    ok_prompts = n_prompts == 500
    ok_empty = abs(frac - A["committed_empty_frac"]) <= EMPTY_TOL
    print(f"  prompts: {n_prompts} ({'PASS' if ok_prompts else 'FAIL -- expected 500'})")
    print(f"  n=1 empty fraction: {frac:.4f} against the committed {A['committed_empty_frac']:.4f}, "
          f"tolerance {EMPTY_TOL} -> {'PASS' if ok_empty else 'FAIL'}")
    if not (ok_prompts and ok_empty):
        print("  Per the pre-registration, the arm does not clear its integrity checks. Chase it.")
        return

    rep = paired(a.out, A["rep_tag"])
    if rep is None:
        print(f"\n  NOT YET SCOREABLE: selection_scaling_per_prompt{A['rep_tag']}.csv is absent or "
              f"has no n=64 column.")
        return
    g, lo, hi, n = rep
    verdict = ("REPLICATES" if lo > 0 else "INVERTS" if hi < 0 else "DOES NOT REPLICATE")
    print("\n=== the committed band ===")
    print(f"  paired g(64) - g(8) over {n} prompts: {g:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print(f"  VERDICT: {verdict}")

    orig = paired(a.out, A["orig_tag"])
    dist = abs(g - orig[0]) if orig else None
    half = (hi - lo) / 2
    if orig:
        print("\n=== committed secondary ===")
        print(f"  the arm on record recomputed here: {orig[0]:+.4f} [{orig[1]:+.4f}, {orig[2]:+.4f}]")
        print(f"  |D_rep - D_orig| = {dist:.4f}, against " +
              ", ".join(f"{label} {d:.4f}" for label, d in A["precedents"]))
    # The stability criterion this session wrote into caution (ap): a paired difference is stable
    # where the effect is large relative to its OWN interval. Reported, never gated -- the verdict
    # above is the pre-registered reading and this ratio is what the criterion predicted from.
    o_half = (orig[2] - orig[1]) / 2 if orig else None
    if o_half:
        print(f"  g / half-width: {orig[0] / o_half:.2f} on the arm on record, "
              f"{g / half:.2f} in this draw")

    new = rows(os.path.join(a.out, f"selection_scaling{A['rep_tag']}.csv")) or []
    for r in sorted(new, key=lambda r: (r["judge"], int(float(r["n"])))):
        print(f"    {r['judge'][:26]:<26} n={int(float(r['n'])):<3} gain={float(r['gain']):+.4f}")

    out = dict(arm=A["name"], seeds="52 53 54", n_prompts=n,
               gain_diff=round(g, 4), lo95=round(lo, 4), hi95=round(hi, 4), verdict=verdict,
               orig_gain_diff=round(orig[0], 4) if orig else None,
               distance=round(dist, 4) if dist is not None else None,
               ratio_orig=round(orig[0] / o_half, 2) if o_half else None,
               ratio_rep=round(g / half, 2) if half else None,
               empty_frac_n1=round(frac, 4), integrity="PASS")
    p = os.path.join(a.out, A["out_csv"])
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out)); w.writeheader(); w.writerow(out)
    print(f"\nwrote {p}")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
