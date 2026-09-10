"""Does anything a deployer can compute predict what a higher Renyi order is worth? (feat-075)

feat-073 measured the order comparison at matched utility and feat-074 found the advantage spans
orders of magnitude between pairs with no explanation. Two hypotheses were pre-registered and
refuted, but on four pairs, where the best candidate scored Spearman +0.80 at an exact two-sided
p = 0.33 -- a power at which a real predictor and a coincidence are the same observation. This runs
the same six candidates over all seven onset pairs, where rho = 1 is p = 2/5040.

The six candidates are fixed in results/onset_prediction_orders_matched.md before the runs:

    risky   the memoriser's own log-probability per token of the protected tokens
    anchor  the anchor's, i.e. s(x) measured on the same tokens
    gap     anchor - risky
    F       the fraction of the fidelity ceiling the audited decoder captures at k = 1
    params  the anchor's parameter count, summed from its safetensors headers
    ntok    the number of protected tokens scored

p-values are exact over all 7! orderings, not asymptotic: at n = 7 the normal approximation is
wrong in the direction that flatters a weak correlation.

Writes <out>/order_predictors.csv (per pair) and <out>/order_predictors_summary.csv. No GPU.

  .venv/bin/python analysis/order_predictors.py --glob 'results/order_frontier_*_bf16.csv' --out results
"""
from __future__ import annotations

import argparse, csv, glob, itertools, json, math, os, random, re, statistics as st, struct, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.order_law import LABEL  # noqa: E402

# The pairs are not independent draws: several share a model family, and a rank test over
# correlated points overstates its own power. Every candidate is therefore also scored on family
# means, where n = 5 and the permutation test is exact again.
FAMILY = {"KL3M-170M": "KL3M", "KL3M-520M": "KL3M", "KL3M-1.7B": "KL3M", "KL3M-3.7B": "KL3M",
          "Pleias-350M": "Pleias", "Pleias-1.2B": "Pleias", "Pleias-3B": "Pleias",
          "Phi-3.5-mini": "Phi", "Comma-7B": "Comma", "TinyComma-1.8B": "TinyComma"}

# where each pair's anchor lives, so its parameter count is read rather than quoted from a name
ANCHOR = {
    "kl3m": "output/phase5/anchor_kl3m-002-520m",
    "kl3m17b": "alea-institute/kl3m-003-1.7b",
    "pleias": "PleIAs/Pleias-1.2b-Preview",
    "pleias350": "PleIAs/Pleias-350m-Preview",
    "phi": "output/phase5/anchor_phi35mini",
    "comma": "common-pile/comma-v0.1-2t",
    "tinycomma": "jacquelinehe/tinycomma-1.8b-llama3-tokenizer",
    "kl3m170m": "alea-institute/kl3m-002-170m",
    "kl3m37b": "alea-institute/kl3m-003-3.7b",
    "pleias3b": "PleIAs/Pleias-3b-Preview",
}


def _shards(model):
    """Every .safetensors file of a model, whether it is a local directory or an hf_cache id."""
    if os.path.isdir(model):
        root = model
    else:
        cand = glob.glob(os.path.join("hf_cache", "models--" + model.replace("/", "--"),
                                      "snapshots", "*"))
        if not cand:
            return []
        root = cand[0]
    return sorted(glob.glob(os.path.join(root, "*.safetensors")))


def n_params(model):
    """Parameter count from the safetensors headers alone -- no weights are read into memory."""
    total = 0
    for path in _shards(model):
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            head = json.loads(fh.read(n))
        for k, v in head.items():
            if k == "__metadata__":
                continue
            sh = v.get("shape", [])
            c = 1
            for d in sh:
                c *= d
            total += c
    return total


def ranks(v):
    r = [0] * len(v)
    for i, j in enumerate(sorted(range(len(v)), key=lambda i: v[i])):
        r[j] = i
    return r


def spearman(a, b):
    n = len(a)
    RA, RB = ranks(a), ranks(b)
    return 1 - 6 * sum((RA[i] - RB[i]) ** 2 for i in range(n)) / (n * (n * n - 1))


def exact_p(a, b, cap=9, draws=2000000, seed=0):
    """Two-sided exact permutation p over all n! orderings. Above `cap` pairs the enumeration is
    not finishable, so a fixed-seed random permutation estimate is returned instead and
    `exact_p.exact` says which was used. At the sizes this paper reports (n <= 7) it is always
    exact, and that matters: the normal approximation flatters a weak correlation at small n."""
    n = len(b)
    RA, RB = ranks(a), ranks(b)
    norm = n * (n * n - 1) / 6.0
    # |rho| >= obs is the same event as sum of squared rank differences <= d2max (or >= its mirror)
    d2 = sum((RA[i] - RB[i]) ** 2 for i in range(n))
    obs = abs(1 - d2 / norm)
    lo, hi = norm * (1 - obs) + 1e-9, norm * (1 + obs) - 1e-9
    if n <= cap:
        exact_p.exact = True
        hit = sum(1 for perm in itertools.permutations(RB)
                  if (lambda s_: s_ <= lo or s_ >= hi)(
                      sum((RA[i] - perm[i]) ** 2 for i in range(n))))
        return hit / math.factorial(n)
    exact_p.exact = False
    rng = random.Random(seed)
    perm = list(RB)
    hit = 0
    for _ in range(draws):
        rng.shuffle(perm)
        s_ = sum((RA[i] - perm[i]) ** 2 for i in range(n))
        if s_ <= lo or s_ >= hi:
            hit += 1
    return hit / draws


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="results/order_frontier_*_bf16.csv")
    ap.add_argument("--orders", type=float, nargs="+", default=[2.0, 4.0, 8.0])
    ap.add_argument("--published-k", type=float, default=1.0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--prefix", default="order_predictors",
                    help="so the float32 control set does not overwrite the bfloat16 set the "
                         "rank test is pre-registered on")
    a = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(a.glob)):
        # A bfloat16 run is the same pair at another precision, never another pair: mixing the two
        # sets would double every pair and make the permutation test meaningless (and, at 13
        # "pairs", uncomputable). The caller picks one precision by naming it in the glob.
        if path.endswith("_matched.csv") or ("_bf16" in path and "bf16" not in a.glob):
            continue
        tag = re.sub(r"^order_frontier_|_bf16|\.csv$", "", os.path.basename(path))
        cells = list(csv.DictReader(open(path)))
        first, ntok = cells[0], float(cells[0]["n_tokens"])
        F = [float(c["price_frac_ceiling"]) for c in cells
             if float(c["alpha"]) == 1.0 and float(c["k"]) == a.published_k][0]
        m = {(float(x["alpha"]), float(x["published_k"])): x
             for x in csv.DictReader(open(path.replace(".csv", "_matched.csv")))}
        r = dict(pair=LABEL.get(tag, tag), tag=tag,
                 risky=round(float(first["logp_target_risky"]) / ntok, 5),
                 anchor=round(float(first["logp_target_safe"]) / ntok, 5),
                 F=round(F, 4), ntok=int(ntok), params=n_params(ANCHOR.get(tag, "")))
        r["gap"] = round(r["anchor"] - r["risky"], 5)
        for o in a.orders:
            key = (o, a.published_k)
            r[f"adv_a{int(o)}"] = (round(float(m[key]["nats_per_window"]), 4)
                                   if key in m and m[key]["nats_per_window"] != "" else None)
        rows.append(r)
    if len(rows) < 3:
        raise SystemExit(f"only {len(rows)} pairs matched {a.glob}; nothing to correlate")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{a.prefix}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print(f"{len(rows)} pairs, matched-utility advantage at published k = {a.published_k}, "
          f"nats per 50-token window\n")
    hdr = f"{'pair':16s}{'risky/tok':>11s}{'s(x)/tok':>10s}{'gap':>8s}{'F':>7s}" \
          f"{'params':>12s}{'ntok':>7s}"
    print(hdr + "".join(f"{'a=%d' % o:>9s}" for o in a.orders))
    for r in sorted(rows, key=lambda r: -(r[f"adv_a{int(a.orders[0])}"] or 0)):
        print(f"{r['pair'][:15]:16s}{r['risky']:>11.5f}{r['anchor']:>10.4f}{r['gap']:>8.3f}"
              f"{r['F']:>7.3f}{r['params']:>12,d}{r['ntok']:>7d}"
              + "".join(f"{r[f'adv_a{int(o)}']:>9.2f}" for o in a.orders))

    fams = sorted({FAMILY.get(r["pair"], r["pair"]) for r in rows})

    def by_family(key):
        """Mean of `key` within each family, in a fixed family order."""
        return [st.mean([r[key] for r in rows if FAMILY.get(r["pair"], r["pair"]) == f])
                for f in fams]

    CANDS = (("risky", "memoriser log p / token"), ("anchor", "anchor rate s(x)"),
             ("gap", "s(x) - memoriser rate"), ("F", "fraction of ceiling at k"),
             ("params", "anchor parameter count"), ("ntok", "protected tokens scored"))

    summary = []
    kind = "exact" if len(rows) <= 9 else "Monte Carlo, 2e6 draws,"
    print(f"\nSpearman against the advantage, {kind} two-sided p over {len(rows)}! orderings:\n")
    print(f"{'candidate':28s}" + "".join(f"{'alpha=%d' % o:>20s}" for o in a.orders))
    for cand, name in CANDS:
        x = [r[cand] for r in rows]
        line = f"{name:28s}"
        for o in a.orders:
            y = [r[f"adv_a{int(o)}"] for r in rows]
            rho, p = spearman(x, y), exact_p(x, y)
            xf, yf = by_family(cand), by_family(f"adv_a{int(o)}")
            rho_f, p_f = spearman(xf, yf), exact_p(xf, yf)
            summary.append(dict(candidate=cand, alpha=o, n_pairs=len(rows),
                                spearman=round(rho, 4), exact_two_sided_p=round(p, 5),
                                exact=exact_p.exact, n_families=len(fams),
                                spearman_family=round(rho_f, 4),
                                exact_two_sided_p_family=round(p_f, 5)))
            line += f"{rho:>+11.2f} (p={p:.3f})"
        print(line)

    print(f"\nsame candidates on {len(fams)} family means ({', '.join(fams)}), exact over all "
          f"{len(fams)}! orderings -- the conservative reading, since the pairs are not "
          f"independent draws:\n")
    print(f"{'candidate':28s}" + "".join(f"{'alpha=%d' % o:>20s}" for o in a.orders))
    for cand, name in CANDS:
        line = f"{name:28s}"
        for o in a.orders:
            s_ = next(x for x in summary if x["candidate"] == cand and x["alpha"] == o)
            line += f"{s_['spearman_family']:>+11.2f} (p={s_['exact_two_sided_p_family']:.3f})"
        print(line)
    with open(os.path.join(a.out, f"{a.prefix}_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)

    best = max(summary, key=lambda s: abs(s["spearman"]))
    print(f"\nbest candidate: {best['candidate']} at alpha = {best['alpha']:.0f}, "
          f"rho = {best['spearman']:+.2f}, exact p = {best['exact_two_sided_p']:.3f}")
    print("committed bands: |rho| >= 0.86 (p <= 0.024) is a predictor; below 0.7 is an earned")
    print("negative; between is inconclusive at this many pairs.")
    print(f"wrote {os.path.join(a.out, a.prefix)}.csv and _summary.csv")


if __name__ == "__main__":
    main()
