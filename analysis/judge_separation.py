"""Plan v5 item D: how far is each decoder from the safe model a judge could have served instead?

The manuscript's headline is a separation in standard errors -- "a judge cannot separate the decoder
from serving its own safe model at k=1 and can at k=3" -- and until now that number was computed by
hand from the summary CSV. This script produces it, so the finer budget grid and the second judge
land as reproducible rows rather than arithmetic in a commit message.

Two statistics per arm, both against the `anchor only` arm (the safe model served alone) and against
the `risky only (null)` arm (the unconstrained model judged against itself at another seed):

  loss rate   the fraction of judged pairs the arm lost, which is what the published numbers use.
              z = (arm - reference) / SE of the difference of two independent proportions, so a
              NEGATIVE z means the arm loses less often than the reference -- it is better.
  utility U   win = 1, tie = 0.5, loss = 0, the same three-point score analysis/utility_price.py
              prices. Reported because a decoder can move ties into wins without moving losses.
              Its z carries the opposite sign convention by construction, so it is negated to match:
              negative is better here too.

Ties are not dropped: they enter U at 0.5 and are part of the denominator of the loss rate.

Each budget also carries the fraction of protected works for which the certificate is already
vacuous there, joined from the per-passage surprisals in results/certificate_caps.csv (Prop. 1: the
certificate says nothing once K = k*T_max reaches S(x)). That is the comparison the whole section
is for -- whether the budget where a judge can finally tell the decoder from its own safe model is
above or below the budget where the guarantee has gone silent.

Also reports the crossover -- the smallest budget at which the KL decoder separates from the anchor
by more than `--crossing-sigma`, interpolated inside its bracket. That is the number the finer grid
exists to pin: the published grid jumped k=1 to k=3 straight across it.

Writes <out>/judge_separation.csv. No GPU.

Usage:
  .venv/bin/python analysis/judge_separation.py --summary results/utility_v5_summary.csv --out results
"""
import argparse, csv, math, os

def vacuous_pct(caps_path, t_max):
    """k -> fraction of protected works whose certificate is vacuous at that budget, as a callable.
    Reads the committed per-passage anchor surprisals rather than recomputing them on a GPU."""
    import csv as _csv
    if not caps_path or not os.path.exists(caps_path):
        return None
    S = [float(r["S_safe"]) for r in _csv.DictReader(open(caps_path))]
    if not S:
        return None
    return lambda k: 100.0 * sum(1 for s in S if k * t_max >= s) / len(S)


ANCHOR = ("anchor only", 0.0)
NULL = ("risky only (null)", -1.0)


def moments(win, tie, loss, n):
    """Mean and per-pair variance of the three-point utility score, and of the loss indicator."""
    w, t, l = win / 100.0, tie / 100.0, loss / 100.0
    u = w + 0.5 * t
    var_u = (w + 0.25 * t) - u * u            # E[U^2] - E[U]^2, exact for a three-point law
    return u, max(var_u, 0.0) / max(n, 1), l, l * (1 - l) / max(n, 1)


def z(a, va, b, vb):
    """(a - b) / SE of the difference. None when neither side has any judged pairs."""
    se = math.sqrt(va + vb)
    return (a - b) / se if se > 0 else None


def crossing(points, thresh):
    """Smallest k whose z is at or below `thresh`, interpolated inside its bracket.
    None when no budget reaches it, and also when the smallest budget probed is already past it:
    there is no bracket then, and extrapolating backwards would invent a crossover."""
    prev = None
    for k, zk in sorted(points):
        if zk is not None and zk <= thresh:
            if prev is None:
                return None
            (k0, z0) = prev
            return k0 + (k - k0) * (z0 - thresh) / (z0 - zk) if z0 != zk else k
        if zk is not None:
            prev = (k, zk)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default="results/utility_v5_summary.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--crossing-sigma", type=float, default=-2.0)
    ap.add_argument("--crossing-arm", default="KL")
    ap.add_argument("--caps", default="results/certificate_caps.csv",
                    help="per-passage anchor surprisals, for the vacuity column ('' to skip)")
    ap.add_argument("--t-max", type=int, default=200, help="the budget is K = k * t_max")
    a = ap.parse_args()

    arms = {}
    for r in csv.DictReader(open(a.summary)):
        nj = int(r["n_judged"] or 0)
        if not nj:
            continue
        win, loss = float(r["win_pct"]), float(r["loss_pct"])
        arms[(r["decoder"], float(r["k"]))] = dict(
            row=r, n=nj, moments=moments(win, 100.0 - win - loss, loss, nj))

    refs = {name: arms[key]["moments"] for name, key in (("anchor", ANCHOR), ("null", NULL))
            if key in arms}
    if "anchor" not in refs:
        raise SystemExit(f"[sep] {a.summary} has no judged 'anchor only' arm to compare against")

    vac = vacuous_pct(a.caps, a.t_max)
    rows = []
    for (decoder, k), v in sorted(arms.items(), key=lambda t: (t[0][0], t[0][1])):
        u, vu, l, vl = v["moments"]
        row = dict(decoder=decoder, k=k, judge=v["row"].get("judge", ""), n_judged=v["n"],
                   loss_pct=round(100 * l, 1), utility=round(u, 4),
                   vacuous_pct=(round(vac(k), 1) if vac and k > 0 else ""))
        for name, (ru, rvu, rl, rvl) in refs.items():
            zl = z(l, vl, rl, rvl)
            zu = z(u, vu, ru, rvu)
            row[f"z_loss_vs_{name}"] = round(zl, 2) if zl is not None else ""
            # U rises when the arm is better, loss falls; negate so both columns read "negative = better"
            row[f"z_utility_vs_{name}"] = round(-zu, 2) if zu is not None else ""
        rows.append(row)

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "judge_separation.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    hdr = f"{'decoder':18s}{'k':>6s}{'n':>7s}{'loss%':>8s}{'U':>8s}{'vacuous%':>10s}"
    for name in refs:
        hdr += f"{'z loss/' + name:>16s}{'z U/' + name:>13s}"
    print(hdr)
    for r in rows:
        line = f"{r['decoder']:18s}{r['k']:>6g}{r['n_judged']:>7d}{r['loss_pct']:>8.1f}{r['utility']:>8.3f}{r['vacuous_pct']:>10}"
        for name in refs:
            line += f"{r[f'z_loss_vs_{name}']:>16}{r[f'z_utility_vs_{name}']:>13}"
        print(line)

    pts = [(r["k"], r[f"z_loss_vs_anchor"]) for r in rows
           if r["decoder"] == a.crossing_arm and r["k"] > 0 and r["z_loss_vs_anchor"] != ""]
    x = crossing(pts, a.crossing_sigma)
    print(f"\n{a.crossing_arm} decoder separates from the anchor by {a.crossing_sigma:g} sigma at "
          + (f"k = {x:.2f}" if x is not None else "no budget on this grid")
          + (f", where the certificate is vacuous for {vac(x):.1f}% of protected works"
             if x is not None and vac else ""))
    print(f"  grid: " + ", ".join(f"k={k:g}:{zk:+.2f}" for k, zk in sorted(pts)))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
