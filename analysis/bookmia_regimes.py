"""Does the vacuity statement survive a 13x larger corpus?

Proposition 1's empirical content in the paper is one number: at the authors' own budget k = 3 the
certificate is vacuous for 100% of the 758 protected passages. Those passages are sixteen English
genre novels, and Limitations says so. This asks whether the statement is a property of divergence
budgets or of our sixteen books, on 9,870 passages across 100 BookMIA books -- and it costs nothing
but anchor forward passes, because s(x) and the verdict K >= S(x) need no decoding, no attack and
no memoriser.

Vacuity is `S(x) <= K` with `K = k * T_max`, T_max the decoder's configured generation cap --
exactly what analysis/certificate_cap.py computes and what the paper's 100% means. A deployment
publishes the budget for the sequence it is allowed to emit, not for the length of the span an
adversary happens to target, and the protected span is shorter than the cap.

The FIRST version of this script used `k >= s_tok` instead, which is Proposition 1's asymptotic
statement -- both K and S(x) are rays in T, so for a long enough work the ratio k/s(x) decides --
and is a different quantity. The CopyBench arm is a positive control for exactly this: it read
0.326 where the paper's own `certificate_cap_summary.csv` says 1.000, so the criterion was wrong
and not the corpus. It is kept below as a clearly labelled secondary, because it is the criterion
Figure 1(a) draws, and it is never the V1 number.

The unseen half is a control the paper has never had. The anchor saw neither half -- "seen" and
"unseen" label the target models BookMIA was built to audit, not our safe model -- so their s(x)
should be indistinguishable. If they are not, the label is confounded with text properties.

Bands committed in results/onset_prediction_bookmia_regimes.md before the run.

Usage: .venv/bin/python analysis/bookmia_regimes.py --out results
"""
import argparse
import csv
import os
import statistics as st

K_PUBLISHED = 3.0        # the authors' own budget, nats per token
T_MAX = 200              # the decoder's configured cap; K = k * T_max, as certificate_cap.py has it
ARMS = [("CopyBench (16 novels)", "results/regimes_copybench.csv"),
        ("BookMIA seen (50 books)", "results/regimes_bookmia100.csv"),
        ("BookMIA unseen (50 books)", "results/regimes_bookmia100unseen.csv")]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=float, default=K_PUBLISHED)
    ap.add_argument("--t-max", type=int, default=T_MAX,
                    help="the decoder's generation cap; the published budget is K = k * t_max")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    rows, got = [], {}
    for label, path in ARMS:
        r = load(path)
        if r is None:
            print(f"  [bookmia] missing {path}, skipping {label}")
            continue
        S = [float(x["total_nats"]) for x in r]
        s_tok = [float(x["nats_per_tok"]) for x in r]
        s_char = [float(x["s_rate"]) for x in r]
        width = [float(x["uncertified_width"]) for x in r]
        got[label] = s_tok
        rows.append(dict(
            arm=label, n_passages=len(r), n_books=len({x["novel"] for x in r}),
            k=a.k, K=a.k * a.t_max, t_max=a.t_max,
            frac_vacuous=round(sum(1 for v in S if v <= a.k * a.t_max) / len(S), 4),
            frac_rate_below_k=round(sum(1 for v in s_tok if v <= a.k) / len(s_tok), 4),
            S_median=round(st.median(S), 2),
            s_tok_median=round(st.median(s_tok), 4),
            s_tok_p10=round(sorted(s_tok)[len(s_tok) // 10], 4),
            s_tok_p90=round(sorted(s_tok)[9 * len(s_tok) // 10], 4),
            s_char_median=round(st.median(s_char), 4),
            kcrit_over_s_median=round(st.median(width), 4),
            frac_width_gt2=round(sum(1 for v in width if v > 2) / len(width), 4)))
    if not rows:
        raise SystemExit("[bookmia] nothing to aggregate")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "bookmia_regimes.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    for r in rows:
        print(f"  {r['arm']:28s} n={r['n_passages']:>5d} books={r['n_books']:>3d}  "
              f"vacuous at K={r['K']:g}: {r['frac_vacuous']:.4f}  S median {r['S_median']:.1f}  "
              f"s_tok median {r['s_tok_median']:.3f} [{r['s_tok_p10']:.3f}, {r['s_tok_p90']:.3f}]  "
              f"rate below k: {r['frac_rate_below_k']:.4f}")

    def by(name):
        return next((r for r in rows if r["arm"].startswith(name)), None)
    cb, seen, unseen = by("CopyBench"), by("BookMIA seen"), by("BookMIA unseen")

    if seen:
        f = seen["frac_vacuous"]
        v1 = "REPLICATES" if f >= 0.99 else "WEAKER" if f >= 0.90 else "DIFFERENT CORPUS"
        print(f"\n  V1 vacuous at K = {a.k:g}*{a.t_max} = {a.k * a.t_max:g} nats on "
              f"{seen['n_passages']} BookMIA-seen passages: {f:.4f}  ->  {v1}")
        if cb:
            print(f"     positive control, the same criterion on the paper's own 758: "
                  f"{cb['frac_vacuous']:.4f} against the 1.000 in certificate_cap_summary.csv")
        if v1 == "DIFFERENT CORPUS":
            print("     the 100% is then a property of the sixteen novels and the paper must "
                  "say so.")
    if cb and seen:
        d = abs(seen["s_tok_median"] / cb["s_tok_median"] - 1)
        print(f"  V2 median s(x) per token {seen['s_tok_median']:.3f} against CopyBench's "
              f"{cb['s_tok_median']:.3f}, {100 * d:.1f}% apart  ->  "
              f"{'SAME REGIME' if d <= 0.15 else 'SHIFTED'}")
    if seen and unseen:
        d = abs(seen["s_tok_median"] / unseen["s_tok_median"] - 1)
        print(f"  V3 seen {seen['s_tok_median']:.3f} against unseen "
              f"{unseen['s_tok_median']:.3f} under an anchor that saw neither, "
              f"{100 * d:.1f}% apart  ->  "
              f"{'INDISTINGUISHABLE' if d <= 0.05 else 'CONFOUNDED'}")
        if d > 0.05:
            print("     no arm on this corpus may treat the halves as exchangeable.")
    for r in rows:
        print(f"  V4 {r['arm']:28s} k_crit/s median x{r['kcrit_over_s_median']:.2f}, "
              f"{100 * r['frac_width_gt2']:.1f}% of passages above x2")
    print("\n  secondary, and NOT V1: the asymptotic criterion Figure 1(a) draws, k >= s(x) per "
          "token,\n  which decides vacuity for a work long enough that K and S(x) are both rays:")
    for r in rows:
        print(f"    {r['arm']:28s} {r['frac_rate_below_k']:.4f} of passages have "
              f"s(x) <= k = {a.k:g}")


if __name__ == "__main__":
    main()
