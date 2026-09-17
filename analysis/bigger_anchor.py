"""feat-128: at selection's price, should a deployer just serve a bigger safe model?

A reviewer's objection, and it appears nowhere in the paper: q(y) <= n p_s(y) makes the anchor's
capability the ceiling, so if the same judged quality is reachable by serving a LARGER safe model
once -- perfectly certified, since a safe model alone needs no certificate -- then selection's 35x
buys nothing a bigger checkpoint would not.

No new generation is needed. The Comma-7B selection arm's RANK-0 DRAW is Comma-7B served alone, so
one order_averaged_h2h.py pass at --sel-dir output/phase5/sel_comma7b_64 puts

    sel_n1     = Comma-7B alone   (the larger safe model)
    anchor_k0  = TinyComma alone  (the control every judged gain in the paper is taken over)

in ONE pass, against ONE fixed opponent, with every item shown in both orders. This reads the
per-prompt file that pass writes and forms the statistic the pre-registration committed:

    G_A = mean( u[Comma-7B alone] - u[TinyComma alone] ),  paired over the 500 prompts

against G_B = +0.1045, TinyComma's gain at n=64 over the same control. Both are GAINS over a shared
control, each computed inside its own pass -- never levels across passes (caution (m)).

Bands are results/onset_prediction_bigger_anchor.md, committed before the run.

No GPU.

Usage:
  .venv/bin/python analysis/bigger_anchor.py --out results
"""
import argparse
import csv
import os
import random

N_BOOT = 10000
G_B = 0.1045            # results/order_averaged_h2h.csv, TinyComma selection n=64 over the same control


def paired_boot(diffs, rng, n_boot=N_BOOT):
    n = len(diffs)
    xs = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return xs[int(0.025 * n_boot)], xs[int(0.975 * n_boot)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-prompt", default="results/order_averaged_h2h_per_prompt_comma7b_alone.csv")
    ap.add_argument("--canonical-per-prompt",
                    default="results/order_averaged_h2h_per_prompt.csv",
                    help="the canonical pass, for the cross-pass paired difference")
    ap.add_argument("--seed", type=int, default=7717)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    rows = list(csv.DictReader(open(a.per_prompt, encoding="utf-8")))
    assert rows, f"empty {a.per_prompt}"
    cols = rows[0].keys()
    big = next(c for c in cols if c.endswith("sel_n1"))          # Comma-7B served alone
    ctrl = next(c for c in cols if c.endswith("anchor_k0"))      # TinyComma served alone
    sel = next(c for c in cols if c.endswith("sel_n64"))         # Comma-7B at n=64

    dA = [float(r[big]) - float(r[ctrl]) for r in rows]          # the bigger anchor's own gain
    dSel = [float(r[sel]) - float(r[ctrl]) for r in rows]        # Comma-7B n=64 over the same control
    gA = sum(dA) / len(dA)
    gSel = sum(dSel) / len(dSel)
    loA, hiA = paired_boot(dA, rng)
    loS, hiS = paired_boot(dSel, rng)

    # The committed band is read on the DIFFERENCE of the two gains, and the two gains come from
    # two passes with two different "anchor sampled once" controls -- this pass's `anchor_k0` (the
    # sweep_plain draw) and the canonical pass's `sel_n1` (the selection run's own rank-0 draw).
    # They differ by about 0.010 in mean, so differencing the two means is NOT the paired
    # difference. The direct paired difference of the two SERVED arms needs no control at all and
    # is immune to the mismatch. Pairing across passes is sound only because judging is
    # deterministic, which is asserted here rather than left in a log.
    canon = {r["prompt_id"]: r for r in csv.DictReader(open(a.canonical_per_prompt,
                                                           encoding="utf-8"))}
    shared = [r for r in rows if r["prompt_id"] in canon]
    assert len(shared) == len(rows), (len(shared), len(rows), "the two passes disagree on prompts")
    same = sum(float(r[ctrl]) == float(canon[r["prompt_id"]]["u_anchor_k0"]) for r in shared)
    assert same == len(shared), (
        same, len(shared), "judging is not deterministic across these passes, so the cross-pass "
        "pairing below is invalid -- do not quote the difference")
    dDiff = [float(r[big]) - float(canon[r["prompt_id"]]["u_sel_n64"]) for r in shared]
    gDiff = sum(dDiff) / len(dDiff)
    loD, hiD = paired_boot(dDiff, rng)

    # the committed reading: does serving the bigger anchor once match TinyComma at n=64?
    verdict = ("BUY THE BIGGER ANCHOR" if gA >= G_B or hiA >= G_B else
               "SELECTION EARNS ITS PRICE")
    out = [
        dict(quantity="G_A  Comma-7B alone, over TinyComma alone", value=round(gA, 4),
             lo95=round(loA, 4), hi95=round(hiA, 4), n=len(rows), note="the larger safe model, n=1"),
        dict(quantity="G_B  TinyComma n=64, over the same control", value=G_B, lo95=0.082,
             hi95=0.128, n=len(rows), note="on record, results/order_averaged_h2h.csv"),
        dict(quantity="Comma-7B n=64, over TinyComma alone", value=round(gSel, 4),
             lo95=round(loS, 4), hi95=round(hiS, 4), n=len(rows),
             note="context: selection ON the bigger anchor"),
        dict(quantity="G_A - G_B  direct paired difference of the served arms",
             value=round(gDiff, 4), lo95=round(loD, 4), hi95=round(hiD, 4), n=len(shared),
             note="Comma-7B alone minus TinyComma n=64; no control, cross-pass, judging determin"
                  "istic on %d/%d" % (same, len(shared))),
        dict(quantity="verdict", value=verdict, lo95="", hi95="", n=len(rows),
             note="BUY THE BIGGER ANCHOR if G_A >= G_B or its interval reaches G_B"),
    ]
    path = os.path.join(a.out, "bigger_anchor.csv")
    os.makedirs(a.out, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    for r in out:
        print(f"  {r['quantity']:44s} {str(r['value']):>9s} [{r['lo95']}, {r['hi95']}]  {r['note']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
