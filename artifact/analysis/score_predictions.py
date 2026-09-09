"""Plan v5 / feat-056: score the three onset rules the way a pre-registration demands.

Three rules have been on the table for where near-verbatim extraction begins:

  q25       the population onset is the 25th percentile of the per-work requirement r(x),
            calibrated on the first two pairs;
  P1        the onset is the median of r(x) = s_s(x) - s_r(x) itself, which follows from
            psi_t(1) = 0 on the geodesic and has nothing fitted in it;
  constant  the onset is c * s(x) for a single c, also calibrated on the first two pairs.

Two of the three were calibrated on pairs 1 and 2, so scoring them there measures the fit and not
the rule. Only pairs measured AFTER their prediction was committed are a test, and this script
reports those separately and leads with them. Predictions come from results/onset_theory.csv
(written before each sweep ran); measurements and their bootstrap intervals come from
results/onset_ci.csv, which also carries the passage count, so when a pair has been measured twice
the higher-n measurement is the one scored.

Writes <out>/prediction_scores.csv. No GPU.

Usage:
  .venv/bin/python analysis/score_predictions.py --out results \
    --calibrated-on "TinyComma-1.8B + mem. Llama-3.1-8B" "Comma-7B + mem. Comma-7B"
"""
import argparse, csv, os, re, statistics as st, sys


def canonical(name):
    """The two manifests spell a pair differently and a re-measurement adds an (n=...) tag."""
    s = name.lower().replace("memorised", "mem.")
    s = re.sub(r"\s*\(n=\d+\)\s*$", "", s)
    return re.sub(r"\s+", " ", s).strip()


def load_measurements(path):
    """canonical pair -> row, keeping the measurement with the most passages."""
    best = {}
    for r in csv.DictReader(open(path)):
        if r["mode"] != "single" or not r["onset_point"]:
            continue
        k = canonical(r["pair"])
        n = int(r["n_passages"])
        if k not in best or n > int(best[k]["n_passages"]):
            best[k] = r
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--theory", default="results/onset_theory.csv")
    ap.add_argument("--ci", default="results/onset_ci.csv")
    ap.add_argument("--seed-words", default="results/onset_seed_words.csv",
                    help="per-pair seed length in words, from analysis/seed_effect.py")
    ap.add_argument("--words-split", type=float, default=10.0,
                    help="the gap between the two seed regimes; they are 7.3 and 13.0-14.4")
    ap.add_argument("--calibrated-on", nargs="+", required=True,
                    help="pairs the q25 and constant rules were fitted on; they are reported but "
                         "excluded from the held-out score")
    a = ap.parse_args()

    meas = load_measurements(a.ci)
    cal = {canonical(x) for x in a.calibrated_on}
    preds = [r for r in csv.DictReader(open(a.theory))
             if canonical(r["pair"]) in meas and not r["pair"].startswith("Ladder rung")]
    if not preds:
        raise SystemExit(f"[score] no pair in {a.theory} has a measurement in {a.ci}")

    # the constant was fitted as the mean measured ratio on the calibration pairs; refit it here
    # from those same pairs so the number in the paper and the number scored cannot drift apart.
    cal_ratios = [float(meas[canonical(r["pair"])]["onset_point"]) / float(r["s_safe_median"])
                  for r in preds if canonical(r["pair"]) in cal]
    c = st.mean(cal_ratios) if cal_ratios else float("nan")
    # The baseline a reviewer will ask for: quote a constant number of nats and do not rescale at
    # all. It is fitted on the same calibration pairs, so it is scored on the same footing, and it
    # can only make the s(x) rule look worse -- which is why it belongs here.
    cal_nats = [float(meas[canonical(r["pair"])]["onset_point"])
                for r in preds if canonical(r["pair"]) in cal]
    b = st.mean(cal_nats) if cal_nats else float("nan")

    rows = []
    for r in preds:
        m = meas[canonical(r["pair"])]
        y = float(m["onset_point"])
        lo, hi = float(m["onset_lo95"]), float(m["onset_hi95"])
        rules = {"q25 of r(x)": float(r["req_q25"]),
                 "P1: median s_s - s_r": float(r["pred_onset_median"]),
                 f"constant {c:.3f}*s(x)": c * float(r["s_safe_median"]),
                 f"baseline: {b:.3f} nats, no rescaling": b}
        for rule, p in rules.items():
            rows.append({"pair": r["pair"], "rule": rule,
                         "held_out": canonical(r["pair"]) not in cal,
                         "n_passages": int(m["n_passages"]),
                         "s_safe": round(float(r["s_safe_median"]), 4),
                         "s_risky": round(float(r["s_risky_median"]), 4),
                         "measured": round(y, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                         "predicted": round(p, 4), "error": round(p - y, 4),
                         "abs_error": round(abs(p - y), 4),
                         "rel_error_pct": round(100 * (p - y) / y, 2),
                         "inside_ci": int(lo <= p <= hi),
                         "measured_ratio": round(y / float(r["s_safe_median"]), 4)})

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "prediction_scores.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    rules = sorted({r["rule"] for r in rows})
    print(f"constant refitted on {len(cal_ratios)} calibration pair(s): c = {c:.4f}\n")
    for held in (True, False):
        sub = [r for r in rows if r["held_out"] is held]
        if not sub:
            continue
        print(f"--- {'HELD OUT (a test)' if held else 'CALIBRATION PAIRS (a fit, not a test)'} ---")
        print(f"{'pair':36s}{'n':>5s}{'measured':>10s}{'95% CI':>16s}"
              + "".join(f"{r.split(':')[0][:16]:>17s}" for r in rules))
        for pair in dict.fromkeys(r["pair"] for r in sub):
            pr = {r["rule"]: r for r in sub if r["pair"] == pair}
            any_ = next(iter(pr.values()))
            ci = f"[{any_['ci_lo']:.2f},{any_['ci_hi']:.2f}]"
            cells = "".join(
                f"{pr[r]['predicted']:11.2f}{'  in' if pr[r]['inside_ci'] else ' OUT':>6s}"
                for r in rules)
            print(f"{pair[:35]:36s}{any_['n_passages']:5d}{any_['measured']:10.3f}{ci:>16s}{cells}")
        print()
    print(f"{'rule':28s}{'held-out mean |err|':>21s}{'in CI':>8s}{'all mean |err|':>16s}{'in CI':>8s}")
    for rule in rules:
        h = [r for r in rows if r["rule"] == rule and r["held_out"]]
        al = [r for r in rows if r["rule"] == rule]
        hv = f"{st.mean(r['abs_error'] for r in h):.3f}" if h else "  -  "
        hc = f"{sum(r['inside_ci'] for r in h)}/{len(h)}" if h else " - "
        print(f"{rule:28s}{hv:>21s}{hc:>8s}"
              f"{st.mean(r['abs_error'] for r in al):16.3f}"
              f"{sum(r['inside_ci'] for r in al)}/{len(al):>7}")
    # P1 is not only a level: it says the ratio should fall as the memoriser leaves more residual
    # surprisal. Scoring the level and the direction separately is the point -- a rule can get one
    # right and the other wrong, and this one does.
    pr = [(1 - float(r["s_risky_median"]) / float(r["s_safe_median"]),
           float(meas[canonical(r["pair"])]["onset_point"]) / float(r["s_safe_median"]),
           r["pair"]) for r in preds]
    if len(pr) > 2:
        def ranks(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            out = [0] * len(v)
            for pos, i in enumerate(order):
                out[i] = pos
            return out
        rp, rm = ranks([x[0] for x in pr]), ranks([x[1] for x in pr])
        n = len(pr)
        rho = 1 - 6 * sum((x - y) ** 2 for x, y in zip(rp, rm)) / (n * (n * n - 1))
        print(f"\ndirection: P1 says onset/s(x) = 1 - s_r/s_s, so the ratio must fall as the "
              f"memoriser\n           leaves more residual surprisal. Spearman over {n} pairs: "
              f"rho = {rho:+.2f}")
        for pred_ratio, meas_ratio, pair in sorted(pr, key=lambda x: -x[0]):
            print(f"  {pair[:38]:40s} predicted {pred_ratio:.3f}   measured {meas_ratio:.3f}")

    # Conditioning on the adversary's context. The seven pairs split exactly in two by how many
    # words 20 tokens buy (7.3 against 13.0-14.4, no overlap), and the split was identified and
    # registered in results/onset_prediction_seed.md before the intervention arms ran. The
    # grouping itself is read off these same measurements, so it is not a test -- the intervention
    # arms are. What it does show is how much of the residual is the protocol rather than the law.
    sw = {}
    if os.path.exists(a.seed_words):
        sw = {canonical(r["pair"]): float(r["seed_words"]) for r in csv.DictReader(open(a.seed_words))}
    if sw:
        pts = [(canonical(r["pair"]), float(r["s_safe_median"]),
                float(meas[canonical(r["pair"])]["onset_point"])) for r in preds]
        blocks = [("all pairs", pts),
                  (f"matched context (> {a.words_split:g} words)",
                   [p for p in pts if sw.get(p[0], 0) > a.words_split]),
                  (f"short context (<= {a.words_split:g} words)",
                   [p for p in pts if sw.get(p[0], 1e9) <= a.words_split])]
        out = []
        print(f"\nleave-one-out, conditioning on how many words the adversary is handed:")
        print(f"  {'subset':34s}{'n':>3s}{'s(x) span':>11s}{'ratio cv':>10s}"
              f"{'k/s(x) rule':>13s}{'constant nats':>15s}")
        for lab, sub in blocks:
            if len(sub) < 3:
                continue
            rs = [o / x for _, x, o in sub]
            def loo(rescale):
                e = []
                for i in range(len(sub)):
                    o_ = [j for j in range(len(sub)) if j != i]
                    c = st.mean((sub[j][2] / sub[j][1]) if rescale else sub[j][2] for j in o_)
                    e.append(abs((c * sub[i][1] if rescale else c) - sub[i][2]))
                return st.mean(e)
            a_, b_ = loo(True), loo(False)
            out.append(dict(subset=lab, n=len(sub),
                            s_x_span=round(max(x for _, x, _ in sub) / min(x for _, x, _ in sub), 3),
                            ratio_lo=round(min(rs), 4), ratio_hi=round(max(rs), 4),
                            ratio_cv_pct=round(100 * st.stdev(rs) / st.mean(rs), 2),
                            loo_abs_err_rescaled=round(a_, 4), loo_abs_err_constant=round(b_, 4),
                            improvement=round(b_ / a_, 2)))
            print(f"  {lab:34s}{len(sub):3d}{max(x for _,x,_ in sub)/min(x for _,x,_ in sub):10.2f}x"
                  f"{100*st.stdev(rs)/st.mean(rs):9.1f}%{a_:12.3f}n{b_:14.3f}n"
                  f"   ({b_/a_:.1f}x better)")
        # Two multiplicity checks, because a 5-of-7 subgroup is exactly the shape of a finding
        # that appears by chance. (a) Is our five the tightest five, or merely a tight five?
        # (b) If the grouping label were assigned at random, how often would the two groups differ
        # by this much? Both are exact enumerations, not asymptotics, at n = 7.
        import itertools
        rr = [o / x for _, x, o in pts]
        ours = tuple(i for i, p_ in enumerate(pts) if sw.get(p_[0], 0) > a.words_split)
        if 3 <= len(ours) < len(pts):
            cv = lambda idx: st.stdev([rr[i] for i in idx]) / st.mean([rr[i] for i in idx])
            subs = list(itertools.combinations(range(len(pts)), len(ours)))
            tighter = sum(1 for c in subs if cv(c) <= cv(ours) + 1e-12)
            m = len(pts) - len(ours)
            obs = abs(st.mean([rr[i] for i in range(len(pts)) if i not in ours])
                      - st.mean([rr[i] for i in ours]))
            labellings = list(itertools.combinations(range(len(pts)), m))
            ge = sum(1 for c in labellings
                     if abs(st.mean([rr[i] for i in c])
                            - st.mean([rr[i] for i in range(len(pts)) if i not in c])) >= obs - 1e-12)
            for r_ in out:
                if r_["subset"].startswith("matched"):
                    r_["p_tightest_subset"] = round(tighter / len(subs), 4)
                    r_["p_permutation_grouping"] = round(ge / len(labellings), 4)
            print(f"  multiplicity: ours is {tighter} of {len(subs)} subsets of size {len(ours)} "
                  f"this tight (p={tighter/len(subs):.3f}); exact permutation over the "
                  f"{len(labellings)} labellings p={ge/len(labellings):.3f}")
        if out:
            with open(os.path.join(a.out, "matched_context.csv"), "w", newline="") as f:
                keys = list(dict.fromkeys(k for r_ in out for k in r_))
                w = csv.DictWriter(f, fieldnames=keys, restval="")
                w.writeheader(); w.writerows(out)
            print(f"  wrote {a.out}/matched_context.csv")

    print(f"\nwrote {a.out}/prediction_scores.csv")


if __name__ == "__main__":
    sys.exit(main())
