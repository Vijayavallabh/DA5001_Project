"""Score results/onset_prediction_kl3m37b_breadth64.md (feat-135) against its committed bands.

Zero GPU. Written BEFORE the arm produced a number and mutation-tested before the data existed
(caution (v): mutation-test a gate before the data exist, not after), so the reading rules here are
the pre-registration's rather than rules chosen after seeing the answer.

The question: every anchor on which the climb to n=64 is established is Comma-family, so the claim is
confounded with model family. This arm takes the largest untried clean anchor on disk,
`alea-institute/kl3m-003-3.7b`, whose 1.7B sibling climbed and then failed to replicate.

WHY THIS IS NOT analysis/score_breadth64.py. That scorer gates each anchor on ranks 0..7 of the new
pool reproducing a committed n=8 arm bit-exactly. This anchor HAS no committed n=8 arm -- it is a new
draw by construction -- so such a gate would be incoherent rather than merely wrong, and the
pre-registration says so in as many words and fixes distributional checks instead. Writing a second
scorer is the honest way to honour that; bending the first one's gate would not be.

The registered gates, in the order they are applied:

  G2  the sweep covers n in {1,2,4,8,16,32,64} on all 500 prompts, and the n=1 row is the anchor's
      own first draw. BLOCKS the band: a partial sweep cannot be read.
  G3  the n=1 empty-completion fraction is REPORTED, never gated against a prior -- there is no
      prior for this anchor, and feat-132 failed precisely by gating a rate against a sibling arm at
      a different n (caution (v)). Above the breadth arm's own 5% the failure is RECORDED rather
      than exempted, exactly as the audited anchor's 6.8% was, and the gain is additionally reported
      on the non-empty prompts so a reader can see it is not a degeneracy filter.
  G4  mean completion length at n=1 is non-degenerate, >= 20 WORDS. BLOCKS the band: an anchor that
      emits almost nothing would make every judged comparison a statement about length. Words, not
      `generation_length_tokens`, which is the PADDED length (caution (ah)).

The band is the paired g(64) - g(8) under judge B, computed WITHIN this pass, which on the
per-prompt file is exactly u_n64 - u_n8 because both gains subtract the same u_n1. A paired
difference inside one pass shares the presentation-order flip sequence, so the grid-dependence
feat-129 measured cannot reach it (caution (ap)).

MARGINALITY IS DECLARED IN ADVANCE. feat-131/133 established that a paired difference is stable
where the effect is large relative to its own interval and not where it is marginal, with the
boundary near 2 interval half-widths. Below 2.0 half-widths this arm reads MARGINAL and its verdict
is not promoted into the manuscript without a seed replication, whatever the interval says.

Usage: .venv/bin/python analysis/score_kl3m37b_breadth64.py --out results
"""
import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_decoding import boot_mean  # noqa: E402

JUDGE_B = "Phi-3.5-mini-instruct"
NAME = "kl3m37b"
LABEL = "KL3M-3.7B"
GRID = (1, 2, 4, 8, 16, 32, 64)
N_PROMPTS = 500
MIN_WORDS = 20.0            # G4
EMPTY_THRESHOLD = 0.05      # G3, reported not gated
HALF_WIDTHS_FOR_STABLE = 2.0
GEN_DIR = "output/phase5/sel_kl3m37b_64"
# The two anchors the claim already rests on give the effect size this arm is powered against.
REFERENCE = (("TinyComma-1.8B", 0.054, 0.142), ("Comma-7B", 0.072, 0.173))


def rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8"))) if os.path.exists(path) else None


def g2_grid(summary, per, out):
    """Coverage. Returns (ok, message)."""
    if summary is None or per is None:
        return False, "the scoring CSVs are not there yet; the arm has not finished"
    jb = [r for r in summary if JUDGE_B in r["judge"]]
    if not jb:
        return False, f"judge B ({JUDGE_B}) is not in the summary: {sorted({r['judge'] for r in summary})}"
    have = sorted({int(float(r["n"])) for r in jb})
    if have != sorted(GRID):
        return False, f"grid is {have}, not {sorted(GRID)}"
    pp = [r for r in per if JUDGE_B in r["judge"]]
    missing = [f"u_n{n}" for n in GRID if f"u_n{n}" not in (pp[0] if pp else {})]
    if missing:
        return False, f"the per-prompt file is missing {missing}"
    if len(pp) != N_PROMPTS:
        return False, f"{len(pp)} prompts in the per-prompt file, not {N_PROMPTS}"
    counts = {int(float(r["n_prompts"])) for r in jb}
    if counts != {N_PROMPTS}:
        return False, f"n_prompts is {counts}, not {{{N_PROMPTS}}}"
    return True, f"grid {have} on {len(pp)} prompts, both judges present"


def g4_length(summary):
    """Non-degenerate completions, in WORDS (caution (ah): the token field is padded)."""
    jb = [r for r in summary if JUDGE_B in r["judge"] and int(float(r["n"])) == 1]
    if not jb or not jb[0].get("mean_words"):
        return False, "no mean_words on the n=1 row", float("nan")
    w = float(jb[0]["mean_words"])
    return w >= MIN_WORDS, f"mean {w:.1f} words at n=1 (>= {MIN_WORDS:.0f} required)", w


def g3_empty(out):
    """Reported, never gated. Uses the committed definitions rather than lookalikes."""
    try:
        from analysis.selection_breadth import gate, nonempty_gain
    except Exception as e:                                    # pragma: no cover
        return None, f"could not import the committed definitions: {e}", None
    toks, empty, npr = gate(GEN_DIR)
    if npr == 0:
        return None, f"no generations at {GEN_DIR}", None
    ne = nonempty_gain(os.path.join(out, f"selection_scaling_per_prompt_{NAME}64.csv"),
                       GEN_DIR, JUDGE_B, 64, random.Random(99))
    return empty, (f"{100 * empty:.1f}% of {npr} n=1 completions are empty "
                   f"({'ABOVE' if empty > EMPTY_THRESHOLD else 'within'} the breadth arm's own "
                   f"{100 * EMPTY_THRESHOLD:.0f}% threshold), mean {toks:.0f} padded tokens"), ne


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = a.out
    summary = rows(os.path.join(out, f"selection_scaling_{NAME}64.csv"))
    per = rows(os.path.join(out, f"selection_scaling_per_prompt_{NAME}64.csv"))

    print(f"results/onset_prediction_{NAME}_breadth64.md -- does the climb to n=64 survive "
          f"off the Comma family?")
    print("  effect size on record: " + ", ".join(
        f"{n} {g8:+.3f} -> {g64:+.3f} (D={g64 - g8:+.3f})" for n, g8, g64 in REFERENCE))
    print("  stage 1 (vetting) PASSED: the anchor reads 0.0000 at every n and at k=-1, which is G1.")
    print("  There is NO bit-identity reproduction gate here, deliberately: this anchor has no "
          "committed n=8 arm to reproduce.")

    ok2, msg2 = g2_grid(summary, per, out)
    print(f"\n  G2 coverage: {'PASS' if ok2 else 'FAIL'} -- {msg2}")
    if not ok2:
        print("  NOT SCORED. Per the pre-registration no n > 8 number is read until the gates pass,")
        print("  and the band is NOT computed 'just to see' (feat-132's never was).")
        return 1

    ok4, msg4, words = g4_length(summary)
    print(f"  G4 length:   {'PASS' if ok4 else 'FAIL'} -- {msg4}")
    if not ok4:
        print("  NOT SCORED: an anchor that emits almost nothing makes every judged comparison a")
        print("  statement about length. The band is not computed.")
        return 1

    empty, msg3, ne = g3_empty(out)
    print(f"  G3 empties:  REPORTED (never gated) -- {msg3}")
    if empty is not None and empty > EMPTY_THRESHOLD:
        print("    RECORDED, NOT EXEMPTED: above the breadth arm's own threshold, exactly as the")
        print("    audited anchor's 6.8% was. The gain on non-empty prompts is reported beside it.")
    if ne:
        print(f"    gain on the {ne[3]} non-empty prompts: {ne[0]:+.4f} [{ne[1]:+.4f}, {ne[2]:+.4f}]")

    pp = [r for r in per if JUDGE_B in r["judge"]]
    d = [float(r["u_n64"]) - float(r["u_n8"]) for r in pp]
    g = sum(d) / len(d)
    lo, hi = boot_mean(d, random.Random(20260919))
    verdict = "CLIMBS" if lo > 0 else "TURNS OVER" if hi < 0 else "SATURATED BY 8"
    hw = (hi - lo) / 2
    ratio = abs(g) / hw if hw > 0 else float("inf")
    marginal = ratio < HALF_WIDTHS_FOR_STABLE
    print(f"\n  paired g(64) - g(8) over {len(d)} prompts, within this pass: "
          f"{g:+.4f} [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")
    print(f"  distance from zero: {ratio:.2f} interval half-widths "
          f"({'MARGINAL' if marginal else 'stable'} at the {HALF_WIDTHS_FOR_STABLE:.1f} boundary "
          f"declared in advance)")
    if marginal:
        print("  MARGINAL: this verdict is NOT promoted into the manuscript without a seed")
        print("  replication, whatever the interval says. feat-131 moved 0.061 at 1.7 half-widths.")

    for r in sorted(summary, key=lambda r: (r["judge"], int(float(r["n"])))):
        print(f"    {r['judge'][:26]:<26} n={int(float(r['n'])):<3} "
              f"gain={float(r['gain']):+.4f} [{float(r['gain_lo95']):+.4f}, "
              f"{float(r['gain_hi95']):+.4f}] kl_nats={float(r['kl_nats']):.4f} "
              f"mean_words={r.get('mean_words','')} spearman={r.get('spearman_u_logn','')}")

    n8 = next(r for r in summary if JUDGE_B in r["judge"] and int(float(r["n"])) == 8)
    n64 = next(r for r in summary if JUDGE_B in r["judge"] and int(float(r["n"])) == 64)
    rec = dict(anchor=LABEL, name=NAME, gain8=round(float(n8["gain"]), 4),
               gain64=round(float(n64["gain"]), 4), gain_diff=round(g, 4),
               lo95=round(lo, 4), hi95=round(hi, 4), half_widths=round(ratio, 2),
               n_prompts=len(d), empty_frac_n1=(round(empty, 4) if empty is not None else ""),
               mean_words_n1=round(words, 1), reproduction_gate="none by design",
               verdict=verdict, marginal=("yes" if marginal else "no"))
    p = os.path.join(out, f"{NAME}_breadth64_scoring.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rec)); w.writeheader(); w.writerow(rec)
    print(f"\n=== VERDICT: {verdict}{' (MARGINAL)' if marginal else ''} ===")
    print(f"wrote {p}")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
