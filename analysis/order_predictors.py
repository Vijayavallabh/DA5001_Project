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

import argparse, csv, glob, itertools, json, os, re, struct, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.order_law import LABEL  # noqa: E402

# where each pair's anchor lives, so its parameter count is read rather than quoted from a name
ANCHOR = {
    "kl3m": "output/phase5/anchor_kl3m-002-520m",
    "kl3m17b": "alea-institute/kl3m-003-1.7b",
    "pleias": "PleIAs/Pleias-1.2b-Preview",
    "pleias350": "PleIAs/Pleias-350m-Preview",
    "phi": "output/phase5/anchor_phi35mini",
    "comma": "common-pile/comma-v0.1-2t",
    "tinycomma": "jacquelinehe/tinycomma-1.8b-llama3-tokenizer",
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


def spearman(a, b):
    n = len(a)
    ra = sorted(range(n), key=lambda i: a[i])
    rb = sorted(range(n), key=lambda i: b[i])
    RA, RB = [0] * n, [0] * n
    for i, j in enumerate(ra):
        RA[j] = i
    for i, j in enumerate(rb):
        RB[j] = i
    return 1 - 6 * sum((RA[i] - RB[i]) ** 2 for i in range(n)) / (n * (n * n - 1))


def exact_p(a, b):
    """Two-sided exact permutation p, over all n! orderings."""
    obs, hit, tot = abs(spearman(a, b)), 0, 0
    for perm in itertools.permutations(range(len(b))):
        tot += 1
        if abs(spearman(a, [b[i] for i in perm])) >= obs - 1e-12:
            hit += 1
    return hit / tot


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="results/order_frontier_*_bf16.csv")
    ap.add_argument("--orders", type=float, nargs="+", default=[2.0, 4.0, 8.0])
    ap.add_argument("--published-k", type=float, default=1.0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(a.glob)):
        if path.endswith("_matched.csv"):
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
    with open(os.path.join(a.out, "order_predictors.csv"), "w", newline="") as fh:
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

    summary = []
    print(f"\nSpearman against the advantage, exact two-sided p over all {len(rows)}! orderings:\n")
    print(f"{'candidate':28s}" + "".join(f"{'alpha=%d' % o:>20s}" for o in a.orders))
    for cand, name in (("risky", "memoriser log p / token"), ("anchor", "anchor rate s(x)"),
                       ("gap", "s(x) - memoriser rate"), ("F", "fraction of ceiling at k"),
                       ("params", "anchor parameter count"), ("ntok", "protected tokens scored")):
        x = [r[cand] for r in rows]
        line = f"{name:28s}"
        for o in a.orders:
            y = [r[f"adv_a{int(o)}"] for r in rows]
            rho, p = spearman(x, y), exact_p(x, y)
            summary.append(dict(candidate=cand, alpha=o, n_pairs=len(rows),
                                spearman=round(rho, 4), exact_two_sided_p=round(p, 5)))
            line += f"{rho:>+11.2f} (p={p:.3f})"
        print(line)
    with open(os.path.join(a.out, "order_predictors_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)

    best = max(summary, key=lambda s: abs(s["spearman"]))
    print(f"\nbest candidate: {best['candidate']} at alpha = {best['alpha']:.0f}, "
          f"rho = {best['spearman']:+.2f}, exact p = {best['exact_two_sided_p']:.3f}")
    print("committed bands: |rho| >= 0.86 (p <= 0.024) is a predictor; below 0.7 is an earned")
    print("negative; between is inconclusive at this many pairs.")
    print(f"wrote {os.path.join(a.out, 'order_predictors.csv')} and _summary.csv")


if __name__ == "__main__":
    main()
