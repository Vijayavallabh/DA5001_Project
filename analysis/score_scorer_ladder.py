"""feat-158: score the scorer-size ladder against the bands in
results/onset_prediction_scorer_ladder.md. Zero GPU.

Every arm here is a reward-only re-score of cached generations, so the strongest instrument check
available is free and is applied FIRST: majority vote never consults the scorer, so every one of its
seven cells must be identical to the committed arm's. If any cell moves, the generations are not the
ones on record and the arm is INVALID rather than a failed band (caution (at)).

Gates are read per arm and BEFORE that arm's band, which is the order the registration fixes.
"""
import argparse
import csv
import os

# arm -> (new CSV, committed reference CSV, human name, task, BAND CELL)
#
# The band cell is task-specific because the CLAIM is task-specific. Appendix I's TriviaQA
# concession is about n=16 -- where the committed interval is -0.038 [-0.068, -0.008] and excludes
# zero on the wrong side -- and about the Spearman, NOT about n=64, where the committed arm already
# reads -0.014 [-0.046, +0.018] and is NO EFFECT. Reading this band at n=64 would have asked
# whether a turn-over that is not there survives a larger scorer. Caught by mutation-testing this
# scorer against a synthetic arm before any data existed; recorded in the registration.
ARMS = {
    "tqa14": ("selection_verifiable_tqa_comma7b_qwen14b.csv",
              "selection_verifiable_tqa_comma7b.csv", "TriviaQA @ Qwen2.5-14B", "TriviaQA", 16),
    "gsm14": ("selection_verifiable_comma7b_qwen14b.csv",
              "selection_verifiable_comma7b.csv", "GSM8K @ Qwen2.5-14B", "GSM8K", 64),
    "cta72": ("selection_verifiable_cta14_comma7b_qwen72b.csv",
              "selection_verifiable_cta14_comma7b.csv", "CoTaEval @ Qwen2.5-72B", "CoTaEval", 64),
}
VOTE = "majority vote (self-consistency)"


def rows(out, name):
    p = os.path.join(out, name)
    return list(csv.DictReader(open(p, encoding="utf-8"))) if os.path.exists(p) else []


def pick(rs, arm_pred, n):
    for r in rs:
        if arm_pred(r["arm"]) and int(r["n"]) == n:
            return r
    return None


def g0_majority_identical(new, ref):
    """Every majority-vote cell, not just n=1. Returns (ok, message)."""
    a = {int(r["n"]): r["acc"] for r in new if r["arm"] == VOTE}
    b = {int(r["n"]): r["acc"] for r in ref if r["arm"] == VOTE}
    if not a or not b:
        return False, "G0 no majority-vote column to compare -- STRUCTURAL, no comparison was made"
    if set(a) != set(b):
        return False, (f"G0 grids differ, new {sorted(a)} vs committed {sorted(b)} "
                       f"-- STRUCTURAL, no comparison was made")
    bad = [n for n in sorted(a) if round(float(a[n]), 4) != round(float(b[n]), 4)]
    if bad:
        return False, (f"G0 majority vote MOVED at n={bad} "
                       f"({[(a[n], b[n]) for n in bad]}) -> FAIL "
                       f"(the generations are not the ones on record; arm INVALID)")
    return True, f"G0 majority vote identical at all {len(a)} cells -> PASS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out, table = [], {}
    for key, (new_f, ref_f, name, task, band_n) in ARMS.items():
        new, ref = rows(a.out, new_f), rows(a.out, ref_f)
        print(f"--- {name}   ({new_f})")
        if not new:
            print("    NO DATA")
            continue
        if not ref:
            print(f"    G0 no committed counterpart ({ref_f}) "
                  f"-- STRUCTURAL, no comparison was made")
            out.append(dict(arm=key, name=name, task=task, gate="FAIL-G0-STRUCTURAL"))
            continue
        ok0, msg0 = g0_majority_identical(new, ref)
        print("    " + msg0)

        sel = [r for r in new if r["arm"].startswith("pointwise reward")]
        risky = [r for r in new if r["arm"].startswith("risky model alone")]
        base = pick(new, lambda s: s.startswith("pointwise reward"), 1)
        if base is None:
            print("    G1 no n=1 selector row -- STRUCTURAL")
            out.append(dict(arm=key, name=name, task=task, gate="FAIL-G1-STRUCTURAL"))
            continue
        anchor1 = float(base["acc"])
        # G1 has a DIRECTION (caution (au)): the unconstrained model must be AT OR ABOVE the anchor.
        r_best = max((float(r["acc"]) for r in risky), default=float("nan"))
        g1 = r_best >= anchor1
        print(f"    G1 risky {r_best:.4f} >= anchor n=1 {anchor1:.4f} -> "
              f"{'PASS' if g1 else 'FAIL (parser or pipeline; arm INVALID)'}")
        ns = sorted({int(r["n"]) for r in sel})
        g2 = ns == sorted({int(r["n"]) for r in ref if r["arm"] == VOTE})
        print(f"    G2 selector grid {ns} matches the committed grid -> "
              f"{'PASS' if g2 else 'FAIL'}")
        if not (ok0 and g1 and g2):
            print("    band NOT computed (a gate blocks it; the registration forbids reading it)")
            out.append(dict(arm=key, name=name, task=task, gate="FAIL"))
            continue
        top = band_n
        assert top in ns, f"{name}: the band cell n={top} is not on the grid {ns}"
        r_top = pick(new, lambda s: s.startswith("pointwise reward"), top)
        pt, lo, hi = (float(r_top["gain"]), float(r_top["gain_lo95"]), float(r_top["gain_hi95"]))
        hw = (hi - lo) / 2
        hwr = abs(pt) / hw if hw else float("inf")
        v = "CLIMBS" if lo > 0 else "TURNS OVER" if hi < 0 else "NO EFFECT"
        marg = "MARGINAL" if hwr < 2.0 else "not marginal"
        print(f"    band  acc(1)={anchor1:.4f} acc({top})={float(r_top['acc']):.4f}  "
              f"gain {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]  {hwr:.2f} hw  {v}  {marg}")
        sp = float(r_top.get("spearman_acc_logn") or "nan")
        sp_ref = float((pick(ref, lambda s: s.startswith("pointwise reward"), top)
                        or {}).get("spearman_acc_logn") or "nan")
        print(f"    spearman(acc, log n) {sp:+.4f}  (committed arm {sp_ref:+.4f})")
        table[key] = v
        out.append(dict(arm=key, name=name, task=task, gate="PASS", n_top=top,
                        spearman=round(sp, 4), spearman_committed=round(sp_ref, 4),
                        acc_n1=round(anchor1, 4), acc_top=round(float(r_top["acc"]), 4),
                        gain=round(pt, 4), lo95=round(lo, 4), hi95=round(hi, 4),
                        half_widths=round(hwr, 2), verdict=v, marginal=marg))

    p = os.path.join(a.out, "scorer_ladder.csv")
    if out:
        keys = sorted({k for r in out for k in r})
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(out)
        print(f"\nwrote {p}")
    # The GSM8K control has teeth in both directions: if it stops climbing, the 14B pipeline is
    # what changed and BOTH 14B arms are INVALID rather than informative. Registered in advance so
    # a convenient TriviaQA result cannot be kept while its own control is thrown away.
    if "gsm14" in table and table["gsm14"] != "CLIMBS":
        print(f"\nINSTRUMENT FAILURE: the GSM8K control reads {table['gsm14']}, not CLIMBS. "
              f"Both 14B arms are INVALID (caution (w)); the CoTaEval 72B arm does not share "
              f"their pipeline and is read on its own.")
        table.pop("tqa14", None)
    if {"tqa14", "cta72"} <= set(table):
        row = [r for r in out if r.get("arm") == "tqa14"][0]
        # TriviaQA's turn-over "survives" only if BOTH halves of the committed claim do.
        tqa_over = table["tqa14"] == "TURNS OVER" and row["spearman"] < 0
        cta_over = table["cta72"] == "TURNS OVER"
        read = ("SCORER-INDEPENDENT AT SCALE" if tqa_over and cta_over else
                "SCORER-BOUND, GENERAL" if not tqa_over and not cta_over else "TASK-DEPENDENT")
        print(f"\nREGISTERED READING: TriviaQA@14B {table['tqa14']}, "
              f"CoTaEval@72B {table['cta72']} -> {read}")
        print(f"  H1 (TriviaQA's turn-over does NOT survive 14B): "
              f"{'CONFIRMED' if not tqa_over else 'REFUTED'}")
        print(f"  H2 (CoTaEval's turn-over DOES survive 72B): "
              f"{'CONFIRMED' if cta_over else 'REFUTED'}")


if __name__ == "__main__":
    main()
