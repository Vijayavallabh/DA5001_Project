"""Can a deployer set the budget without knowing the protected work? (plan v5 / feat-071)

Section 3 tells a deployer to publish k/s(x), and then concedes the problem with that advice: only
a rate quoted "as a fraction of what ordinary traffic costs" is available to someone "who does not
know the protected works in advance". A rights-holder's work is exactly what a deployer has not
seen. So the paper's own prescription is, as it stands, not executable.

There is an obvious candidate the repository already carries the inputs for. The anchor's surprisal
rate on *public-domain* prose of the same kind -- 50 Gutenberg texts, scored on identical spans in
analysis/anchor_scaling.py -- needs no protected text at all, and a deployer can compute it for
their own anchor in one pass. The question is whether it transfers:

    s_protected  =~  c * s_proxy         with one c that holds across anchors?

This scores that the way the paper scores every other predictor: leave one anchor out, fit on the
rest, predict the held-out anchor's protected rate, and compare against the null a referee reaches
for -- quoting a constant number of nats and not rescaling at all. Ten anchors across three corpus
families, so the leave-one-out is over genuinely different models.

Writes <out>/proxy_budget.csv (per held-out anchor) and <out>/proxy_budget_summary.csv. No GPU.

  .venv/bin/python analysis/proxy_budget.py --out results
"""
from __future__ import annotations

import argparse, csv, os, statistics as st


def loo(rows, predict):
    """Leave-one-anchor-out absolute error of `predict(train_rows, held_out_row)`."""
    errs = []
    for i, held in enumerate(rows):
        train = rows[:i] + rows[i + 1:]
        errs.append(abs(predict(train, held) - held["s_prot"]))
    return errs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scaling", default="results/anchor_scaling_summary.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    rows = []
    for r in csv.DictReader(open(a.scaling)):
        rows.append(dict(model=r["model"], corpus=r["corpus"],
                         s_prot=float(r["s_passage"]),
                         s_proxy=float(r["s_gutenberg_span"]),
                         c_use=float(r["c_use"])))

    # three rules, each fitted only on the training anchors
    rescale = lambda tr, h: st.mean(x["s_prot"] / x["s_proxy"] for x in tr) * h["s_proxy"]
    constant = lambda tr, h: st.mean(x["s_prot"] for x in tr)
    by_cuse = lambda tr, h: st.mean(x["s_prot"] / x["c_use"] for x in tr) * h["c_use"]

    e_rescale, e_constant, e_cuse = loo(rows, rescale), loo(rows, constant), loo(rows, by_cuse)

    per = []
    for r, a1, a2, a3 in zip(rows, e_rescale, e_constant, e_cuse):
        per.append(dict(model=r["model"], corpus=r["corpus"],
                        s_protected=round(r["s_prot"], 4), s_proxy=round(r["s_proxy"], 4),
                        ratio=round(r["s_prot"] / r["s_proxy"], 4),
                        err_proxy=round(a1, 4), err_constant=round(a2, 4), err_cuse=round(a3, 4)))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "proxy_budget.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per[0])); w.writeheader(); w.writerows(per)

    ratios = [r["s_prot"] / r["s_proxy"] for r in rows]
    prot = [r["s_prot"] for r in rows]
    summary = dict(
        n_anchors=len(rows),
        s_protected_span=round(max(prot) / min(prot), 3),
        ratio_mean=round(st.mean(ratios), 4), ratio_sd=round(st.stdev(ratios), 4),
        ratio_cv_pct=round(100 * st.stdev(ratios) / st.mean(ratios), 2),
        ratio_lo=round(min(ratios), 4), ratio_hi=round(max(ratios), 4),
        loo_proxy=round(st.mean(e_rescale), 4),
        loo_constant=round(st.mean(e_constant), 4),
        loo_cuse=round(st.mean(e_cuse), 4),
        improvement_over_constant=round(st.mean(e_constant) / st.mean(e_rescale), 2),
        loo_proxy_pct=round(100 * st.mean(e_rescale) / st.mean(prot), 2),
    )
    with open(os.path.join(a.out, "proxy_budget_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary)); w.writeheader(); w.writerow(summary)

    print(f"{len(rows)} anchors, protected rate spanning {summary['s_protected_span']}x\n")
    print(f"{'held-out anchor':32s}{'s_prot':>9s}{'s_proxy':>9s}{'ratio':>8s}"
          f"{'|err| proxy':>13s}{'constant':>11s}")
    for p in per:
        print(f"{p['model'].split('/')[-1][:31]:32s}{p['s_protected']:>9.4f}{p['s_proxy']:>9.4f}"
              f"{p['ratio']:>8.4f}{p['err_proxy']:>13.4f}{p['err_constant']:>11.4f}")
    print(f"\nleave-one-anchor-out mean absolute error, nats per character:")
    print(f"  public-domain proxy, rescaled   {summary['loo_proxy']:.4f}   "
          f"({summary['loo_proxy_pct']:.1f}% of the protected rate)")
    print(f"  c_use, rescaled                 {summary['loo_cuse']:.4f}")
    print(f"  a constant, no rescaling        {summary['loo_constant']:.4f}")
    print(f"  proxy beats the constant by     {summary['improvement_over_constant']}x")
    print(f"\nratio s_protected/s_proxy: {summary['ratio_mean']} +- {summary['ratio_sd']} "
          f"(cv {summary['ratio_cv_pct']}%, range {summary['ratio_lo']}-{summary['ratio_hi']})")


if __name__ == "__main__":
    main()
