"""Score feat-155/feat-156 against their committed bands. Zero GPU.

Gates are applied per anchor and BEFORE that anchor's band, which is the order the registrations
fix. The registered selector is the POINTWISE REWARD (Qwen2.5-7B on the fixed template); majority
vote is reported beside it because the paper's own judge-free headline uses it and because the
contrast between the two is the whole reward-overoptimisation question.

Bands: results/onset_prediction_cotaeval_news.md, results/onset_prediction_cotaeval_breadth.md.
"""
import argparse
import csv
import glob
import os

ANCHORS = {
    "cta_news": "Comma-7B (2T)", "cta_s5254": "Comma-7B (2T), seed 5254",
    "cta_comma1t": "Comma-7B (1T)", "cta_tc18b": "TinyComma-1.8B",
    "cta_kl3m37b": "KL3M-3.7B", "cta_kl3m17b": "KL3M-1.7B",
    "cta_pleias3b": "Pleias-3B", "cta_pleias12b": "Pleias-1.2B",
}
REGISTERED = "pointwise reward (Qwen2.5-7B)"
VOTE = "majority vote (self-consistency)"

# feat-157: the same protocol with the scorer swapped 7B -> 14B, at the four anchors that passed
# G1. Each arm names the DIRECTORY it is the counterpart of, so the instrument reference is derived
# by this script from that arm's own CSV and is never a constant typed into a pre-registration
# (caution (v), caution (at)).
SCORER14 = {
    "cta14_comma7b": ("Comma-7B (2T)", "cta_news"),
    "cta14_comma1t": ("Comma-7B (1T)", "cta_comma1t"),
    "cta14_tc18b": ("TinyComma-1.8B", "cta_tc18b"),
    "cta14_s5254": ("Comma-7B (2T), seed 5254", "cta_s5254"),
}
REGISTERED14 = "pointwise reward (Qwen2.5-14B)"


def rows(tag):
    p = f"results/selection_verifiable_{tag}.csv"
    return list(csv.DictReader(open(p, encoding="utf-8"))) if os.path.exists(p) else []


def pick(rs, arm, n):
    for r in rs:
        if r["arm"] == arm and int(r["n"]) == n:
            return r
    return None


def verdict(lo, hi, pt):
    if lo > 0:
        return "CLIMBS"
    if hi < 0:
        return "TURNS OVER"
    return "NO EFFECT"


def g0_instrument(tag, ref_tag, selector, ref_selector):
    """feat-157's instrument check: n=1 does not involve the scorer, so it must not move.

    The tolerance is the REFERENCE arm's own n=1 bootstrap interval, read out of its CSV by this
    function. Nothing is typed in. Returns (ok, message).
    """
    ref = pick(rows(ref_tag), ref_selector, 1)
    cur = pick(rows(tag), selector, 1)
    if ref is None or cur is None:
        return False, f"G0 no counterpart arm ({ref_tag}) -- STRUCTURAL, no comparison was made"
    lo, hi, got = float(ref["acc_lo95"]), float(ref["acc_hi95"]), float(cur["acc"])
    ok = lo <= got <= hi
    return ok, (f"G0 n=1 F1 {got:.4f} in {ref_tag}'s own [{lo:.4f}, {hi:.4f}] -> "
                f"{'PASS' if ok else 'FAIL (pipeline changed; arm INVALID)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--scorer-scale", action="store_true",
                    help="score feat-157 (Qwen2.5-14B scorer) instead of feat-155/156")
    a = ap.parse_args()
    if a.scorer_scale:
        return scorer_scale(a)
    out = []
    for tag, name in ANCHORS.items():
        rs = rows(tag)
        if not rs:
            print(f"--- {name:26s} NO DATA")
            continue
        base = pick(rs, REGISTERED, 1)
        risky = [r for r in rs if r["arm"].startswith("risky model alone")]
        f1_1 = float(base["acc"])
        r_best = max(float(r["acc"]) for r in risky) if risky else float("nan")
        g1 = f1_1 >= 0.10
        g2 = r_best > f1_1
        print(f"--- {name}")
        print(f"    G1 anchor n=1 F1 {f1_1:.4f} >= 0.10 -> {'PASS' if g1 else 'FAIL (UNUSABLE ON THIS TASK)'}")
        print(f"    G2 risky {r_best:.4f} > anchor {f1_1:.4f} -> {'PASS' if g2 else 'FAIL (INVALID)'}")
        if not (g1 and g2):
            print("    band NOT computed (a gate blocks it; the registration forbids reading it)")
            out.append(dict(tag=tag, anchor=name, gate="FAIL", f1_n1=round(f1_1, 4)))
            continue
        for rule, label in ((REGISTERED, "registered: pointwise reward"), (VOTE, "majority vote")):
            r64 = pick(rs, rule, 64)
            if not r64:
                continue
            pt, lo, hi = (float(r64["gain"]), float(r64["gain_lo95"]), float(r64["gain_hi95"]))
            hw = (hi - lo) / 2
            hwr = abs(pt) / hw if hw else float("inf")
            v = verdict(lo, hi, pt)
            marg = "MARGINAL" if hwr < 2.0 else "not marginal"
            print(f"    {label:30s} F1(1)={float(pick(rs, rule, 1)['acc']):.4f} "
                  f"F1(64)={float(r64['acc']):.4f}  gain {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
                  f"{hwr:.2f} hw  {v}  {marg}")
            out.append(dict(tag=tag, anchor=name, rule=rule, gate="PASS",
                            f1_n1=round(float(pick(rs, rule, 1)["acc"]), 4),
                            f1_n64=round(float(r64["acc"]), 4), gain=round(pt, 4),
                            lo95=round(lo, 4), hi95=round(hi, 4),
                            half_widths=round(hwr, 2), verdict=v, marginal=marg))
    p = os.path.join(a.out, "cotaeval_scoring.csv")
    if out:
        keys = sorted({k for r in out for k in r})
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(out)
        print(f"\nwrote {p}")
        climbs = [r for r in out if r.get("rule") == REGISTERED and r.get("verdict") == "CLIMBS"]
        print(f"LADDER CLAIM: {len(climbs)} anchors climb under the registered selector "
              f"(band: at least 2)")


def scorer_scale(a):
    """feat-157. Same gates, plus G0; the reading is the COMPARISON OF VERDICTS across the four."""
    out = []
    for tag, (name, ref_tag) in SCORER14.items():
        rs = rows(tag)
        print(f"--- {name}  ({tag}, counterpart of {ref_tag})")
        if not rs:
            print("    NO DATA")
            continue
        ok0, msg0 = g0_instrument(tag, ref_tag, REGISTERED14, REGISTERED)
        print("    " + msg0)
        base = pick(rs, REGISTERED14, 1)
        risky = [r for r in rs if r["arm"].startswith("risky model alone")]
        f1_1 = float(base["acc"])
        r_best = max(float(r["acc"]) for r in risky) if risky else float("nan")
        g1, g2 = f1_1 >= 0.10, r_best > f1_1
        print(f"    G1 anchor n=1 F1 {f1_1:.4f} >= 0.10 -> {'PASS' if g1 else 'FAIL'}")
        print(f"    G2 risky {r_best:.4f} > anchor {f1_1:.4f} -> {'PASS' if g2 else 'FAIL'}")
        if not (ok0 and g1 and g2):
            print("    band NOT computed (a gate blocks it; the registration forbids reading it)")
            out.append(dict(tag=tag, anchor=name, gate="FAIL", f1_n1=round(f1_1, 4)))
            continue
        for rule, label in ((REGISTERED14, "registered: pointwise reward (14B)"),
                            (VOTE, "majority vote")):
            r64 = pick(rs, rule, 64)
            if not r64:
                continue
            pt, lo, hi = float(r64["gain"]), float(r64["gain_lo95"]), float(r64["gain_hi95"])
            hw = (hi - lo) / 2
            hwr = abs(pt) / hw if hw else float("inf")
            v = verdict(lo, hi, pt)
            marg = "MARGINAL" if hwr < 2.0 else "not marginal"
            print(f"    {label:34s} F1(1)={float(pick(rs, rule, 1)['acc']):.4f} "
                  f"F1(64)={float(r64['acc']):.4f}  gain {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
                  f"{hwr:.2f} hw  {v}  {marg}")
            out.append(dict(tag=tag, anchor=name, rule=rule, gate="PASS",
                            f1_n1=round(float(pick(rs, rule, 1)["acc"]), 4),
                            f1_n64=round(float(r64["acc"]), 4), gain=round(pt, 4),
                            lo95=round(lo, 4), hi95=round(hi, 4),
                            half_widths=round(hwr, 2), verdict=v, marginal=marg))
    p = os.path.join(a.out, "cotaeval_scorer_scale.csv")
    if out:
        keys = sorted({k for r in out for k in r})
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(out)
        print(f"\nwrote {p}")
        reg = [r for r in out if r.get("rule") == REGISTERED14]
        over = [r for r in reg if r.get("verdict") == "TURNS OVER"]
        print(f"\nREGISTERED READING: {len(over)} of {len(reg)} anchors still TURN OVER under the "
              f"14B scorer -> "
              + ("SCORER-INDEPENDENT" if len(over) == 4 else
                 "SCORER-BOUND" if len(reg) - len(over) >= 2 else "UNRESOLVED"))


if __name__ == "__main__":
    main()
