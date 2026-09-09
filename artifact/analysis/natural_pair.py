"""Plan v5 / feat-053: does the onset law hold for a memoriser nobody fine-tuned?

Every pair in Section 4 is self-paired against a LoRA memoriser we built, so the obvious objection
is that the law describes our fine-tuning recipe rather than the mechanism. Phase 2 already
measured the one pair where memorisation happened in pretraining: Llama-3.1-70B base, which
reproduces *Harry Potter and the Sorcerer's Stone* from its opening words (Cooper et al. 2025),
fused against the same TinyComma anchor at He et al.'s book decoding settings. It qualifies under
the same admissibility rule -- unconstrained single-query recall 0.314, well above 0.10.

Two comparisons, and only the first is informative:

  collapse   The natural pair's budget grid (0, 1, 1.5, 3, 5, 10, 20) overlaps the fine grids of
             the three built pairs in rescaled units k/s(x). Where they overlap we can ask the
             question the collapse claims to answer -- at the same fraction of the vacuity
             threshold, do they leak at the same rate? -- with no interpolation of the natural
             pair at all, only of the others.

  onset      Interpolating a crossing inside a bracket 1.5 nats wide is not a measurement. We
             report it anyway, together with what the SAME coarse treatment does to the three
             pairs whose fine grids we do have, which is the only way to read it.

s(x) is the TinyComma anchor's per-token surprisal on those 50 passages
(results/budget_path_hp_test.csv, produced by analysis/budget_path.py --split test
--novel harry_potter). No GPU.

Usage:
  .venv/bin/python analysis/natural_pair.py --out results
"""
import argparse, bisect, collections, csv, os, random, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.onset import curve, load_pairs  # noqa: E402
from analysis.onset_ci import crossing  # noqa: E402

THRESH = 0.01


def per_passage(path, mode="single"):
    """{k: [recall per passage]} for k > 0. k <= 0 are the mandatory baselines, not budget points."""
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["mode"] == mode and float(r["k"]) > 0:
            by[float(r["k"])].append(float(r["nv_recall"]))
    return dict(sorted(by.items()))


def interp(c, scale, x):
    ks = [k / scale for k in sorted(c)]
    vs = [c[k] for k in sorted(c)]
    if x < ks[0] or x > ks[-1]:
        return None
    i = max(1, bisect.bisect_left(ks, x))
    if ks[i] == ks[i - 1]:
        return vs[i]
    t = (x - ks[i - 1]) / (ks[i] - ks[i - 1])
    return vs[i - 1] + t * (vs[i] - vs[i - 1])


def boot_mean(vals, n=2000, seed=0, lo=2.5, hi=97.5):
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choices(vals, k=len(vals))) for _ in range(n))
    q = lambda p: ms[min(len(ms) - 1, int(p / 100 * len(ms)))]
    return q(lo), q(hi)


def coarse_onset(c, bracket_lo, bracket_hi, thresh=THRESH):
    """Re-estimate an onset keeping only the two grid points nearest a wide bracket, which is what
    the natural pair's grid leaves us with."""
    ks = sorted(c)
    lo = min(ks, key=lambda k: abs(k - bracket_lo))
    hi = min((k for k in ks if k > lo and c[k] >= thresh), default=None, key=lambda k: abs(k - bracket_hi))
    if hi is None:
        return None
    return crossing({lo: c[lo], hi: c[hi]}, thresh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nat", default="output/phase2/nm/hp1_B/composition.csv",
                    help="per-passage recall for the natural pair at the authors' book settings")
    ap.add_argument("--nat-alt", default="output/phase2/nm/hp1_A/composition.csv",
                    help="same pair at temperature 1, the released-log setting")
    ap.add_argument("--sx", default="results/budget_path_hp_test.csv")
    ap.add_argument("--pairs-file", default="results/onset_pairs.tsv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    s_nat = st.median(float(r["s_mean"]) for r in csv.DictReader(open(a.sx)))
    built = []
    for name, comp, bp in load_pairs(a.pairs_file):
        if not os.path.exists(comp):
            print(f"[natural] missing {comp}, skipping {name}", file=sys.stderr)
            continue
        built.append((name, curve(comp, "single", 0),
                      st.median(float(r["s_mean"]) for r in csv.DictReader(open(bp)))))

    rows = []
    for setting, path in (("authors' book settings", a.nat), ("temperature 1", a.nat_alt)):
        if not os.path.exists(path):
            print(f"[natural] missing {path}", file=sys.stderr)
            continue
        pp = per_passage(path)
        for k, vals in pp.items():
            x = k / s_nat
            others = {n: interp(c, s, x) for n, c, s in built}
            if all(v is None for v in others.values()):
                continue                      # no overlap with any fine grid: nothing to compare
            lo, hi = boot_mean(vals)
            comparable = [v for v in others.values() if v is not None]
            rows.append({"block": "collapse", "setting": setting, "k": k, "k_over_s": round(x, 4),
                         "natural_recall": round(st.mean(vals), 4),
                         "natural_ci_lo": round(lo, 4), "natural_ci_hi": round(hi, 4),
                         "n_passages": len(vals), "n_built_pairs_covering": len(comparable),
                         "built_recall_min": round(min(comparable), 4),
                         "built_recall_max": round(max(comparable), 4),
                         "built_recall_mean": round(st.mean(comparable), 4),
                         **{f"recall_{n}": (round(v, 4) if v is not None else "")
                            for n, v in others.items()}})

    # the onset, and what the same coarse grid does to the pairs we can check
    nat_c = {k: st.mean(v) for k, v in per_passage(a.nat).items()}
    nat_ks = sorted(nat_c)
    br_hi = next((k for k in nat_ks if nat_c[k] >= THRESH), None)
    br_lo = max((k for k in nat_ks if k < br_hi), default=None) if br_hi else None
    nat_onset = crossing(nat_c, THRESH)
    rows.append({"block": "onset", "setting": "natural pair, coarse grid", "k": "",
                 "k_over_s": round(nat_onset / s_nat, 4) if nat_onset else "",
                 "natural_recall": "", "natural_ci_lo": "", "natural_ci_hi": "",
                 "n_passages": len(next(iter(per_passage(a.nat).values()))),
                 "n_built_pairs_covering": "", "built_recall_min": "", "built_recall_max": "",
                 "built_recall_mean": ""})
    for name, c, s in built:
        fine = crossing(c, THRESH)
        coarse = coarse_onset(c, br_lo, br_hi) if br_lo is not None else None
        rows.append({"block": "grid_bias", "setting": name, "k": "",
                     "k_over_s": round(coarse / s, 4) if coarse else "",
                     "natural_recall": "", "natural_ci_lo": "", "natural_ci_hi": "",
                     "n_passages": "", "n_built_pairs_covering": "",
                     "built_recall_min": round(fine / s, 4) if fine else "",
                     "built_recall_max": round((coarse - fine) / s, 4) if (coarse and fine) else "",
                     "built_recall_mean": ""})

    os.makedirs(a.out, exist_ok=True)
    keys = list({k for r in rows for k in r})
    order = [k for k in rows[0] if k in keys] + [k for k in keys if k not in rows[0]]
    with open(os.path.join(a.out, "natural_pair.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=order, restval="")
        w.writeheader()
        w.writerows(rows)

    print(f"natural pair: TinyComma-1.8B + Llama-3.1-70B base (no fine-tuning), 50 Harry Potter "
          f"passages, s(x) = {s_nat:.3f} nats/token\n")
    print(f"{'setting':24s}{'k':>6s}{'k/s(x)':>8s}{'natural':>10s}{'95% CI':>16s}"
          f"{'built pairs':>14s}{'n':>3s}")
    for r in rows:
        if r["block"] != "collapse":
            continue
        ci = f"[{r['natural_ci_lo']:.3f},{r['natural_ci_hi']:.3f}]"
        rng = f"{r['built_recall_min']:.3f}-{r['built_recall_max']:.3f}"
        print(f"{r['setting']:24s}{r['k']:6g}{r['k_over_s']:8.3f}{r['natural_recall']:10.4f}"
              f"{ci:>16s}{rng:>14s}{r['n_built_pairs_covering']:3d}")
    print("\nonset, and the same coarse grid applied to the pairs we can check:")
    for r in rows:
        if r["block"] == "onset":
            print(f"  natural pair, bracket ({br_lo:g}, {br_hi:g}]: onset/s(x) = {r['k_over_s']}")
        elif r["block"] == "grid_bias":
            print(f"  {r['setting'][:38]:40s} fine {r['built_recall_min']}  "
                  f"coarse {r['k_over_s']}  shift {r['built_recall_max']}")
    print(f"\nwrote {a.out}/natural_pair.csv")


if __name__ == "__main__":
    main()
