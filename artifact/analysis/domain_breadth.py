"""Is the constructive claim anchor-bound? Selection's gain, split by domain.

Proposition 4 gives q(y) <= n p_s(y): a served string must be one the anchor would have drawn, so
selection can only REORDER what the anchor already produces. Limitations asserts the consequence --
"selection anchoring cannot exceed its anchor's support" -- and the paper has never measured it.

The standard-benchmark arms are weaker than the in-house prompt set (AlpacaEval +0.031/+0.067 at
n=8 against +0.054, MT-Bench nothing in either judge). If the ceiling is the reason, the gain must
be larger where the anchor can already do the task, and the aggregate averages over domains that
differ in exactly that. If it is flat across domains of very different anchor competence, the
Limitations diagnosis is wrong and the weakening is unexplained.

Seven cells, all fixed in results/onset_prediction_domain_breadth.md before any of them was
computed: AlpacaEval's five shipped sources, and MT-Bench in two committed families of four
categories (ten prompts per category resolve nothing). Both judges; the D1 reading requires the
sign in BOTH, so neither can be picked afterwards.

No GPU and no re-judging: this reads the per-prompt verdicts the scaling arms already wrote.

Usage:
  .venv/bin/python analysis/domain_breadth.py --out results
"""
import argparse
import csv
import itertools
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_decoding import boot_mean  # noqa: E402

# MT-Bench's eight categories in two families. Committed before the eight cells were computed;
# `analysis/domain_breadth.py --regroup` does not exist, on purpose.
MT_FAMILY = {"writing": "open-ended", "roleplay": "open-ended", "humanities": "open-ended",
             "stem": "open-ended", "reasoning": "constrained", "math": "constrained",
             "coding": "constrained", "extraction": "constrained"}


def domains(corpus_path, family=None):
    """prompt_id -> cell. The benchmark's own field, never a grouping of ours except MT_FAMILY."""
    out = {}
    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            cat = r["source_novel"]
            out[r["prompt_id"]] = family[cat] if family else cat
    return out


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):                      # average ties, as elsewhere in the repo
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for m in range(i, j + 1):
                r[order[m]] = (i + j) / 2.0 + 1
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def exact_p(xs, ys):
    """Two-sided exact permutation p over all orderings. Seven cells is 5040, so no sampling."""
    obs = spearman(xs, ys)
    perms = list(itertools.permutations(ys))
    hits = sum(1 for q in perms if abs(spearman(xs, list(q))) >= abs(obs) - 1e-12)
    return obs, hits / len(perms)


def cells(per_prompt_csv, dmap, rng, n_col="u_n8"):
    """(judge, cell) -> control level, gain, paired CI, over the prompts in that cell."""
    rows = list(csv.DictReader(open(per_prompt_csv, encoding="utf-8")))
    out = {}
    for judge in sorted({r["judge"] for r in rows}):
        by = {}
        for r in rows:
            if r["judge"] != judge or r["prompt_id"] not in dmap:
                continue
            by.setdefault(dmap[r["prompt_id"]], []).append(
                (float(r["u_n1"]), float(r[n_col])))
        for cell, vals in by.items():
            ctrl = [a for a, _ in vals]
            diff = [b - a for a, b in vals]
            lo, hi = boot_mean(diff, rng)
            clo, chi = boot_mean(ctrl, rng)
            out[(judge, cell)] = dict(
                judge=judge, cell=cell, n_prompts=len(vals),
                control=round(sum(ctrl) / len(ctrl), 4),
                control_lo95=round(clo, 4), control_hi95=round(chi, 4),
                gain=round(sum(diff) / len(diff), 4),
                gain_lo95=round(lo, 4), gain_hi95=round(hi, 4))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpaca-per-prompt", default="results/selection_scaling_per_prompt_alpaca.csv")
    ap.add_argument("--mtbench-per-prompt",
                    default="results/selection_scaling_per_prompt_mtbench.csv")
    ap.add_argument("--alpaca-corpus", default="data/bench/alpaca_factual.jsonl")
    ap.add_argument("--mtbench-corpus", default="data/bench/mtbench_factual.jsonl")
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--null-reps", type=int, default=20000,
                    help="draws of the within-prompt exchangeability null for D1")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    got = cells(a.alpaca_per_prompt, domains(a.alpaca_corpus), rng)
    for k, v in cells(a.mtbench_per_prompt, domains(a.mtbench_corpus, MT_FAMILY), rng).items():
        got[k] = v
    for v in got.values():
        v["benchmark"] = "MT-Bench" if v["cell"] in ("open-ended", "constrained") else "AlpacaEval"

    rows = sorted(got.values(), key=lambda r: (r["judge"], r["benchmark"], -r["n_prompts"]))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "domain_breadth.csv"), "w", newline="") as fh:
        f = ["judge", "benchmark", "cell", "n_prompts", "control", "control_lo95", "control_hi95",
             "gain", "gain_lo95", "gain_hi95"]
        w = csv.DictWriter(fh, fieldnames=f)
        w.writeheader()
        w.writerows([{k: r[k] for k in f} for r in rows])

    judges = sorted({r["judge"] for r in rows})
    for j in judges:
        print(f"\n  {j}")
        for r in [x for x in rows if x["judge"] == j]:
            print(f"    {r['cell']:<14s} n={r['n_prompts']:>3d}  control {r['control']:.3f} "
                  f"[{r['control_lo95']:.3f}, {r['control_hi95']:.3f}]  gain {r['gain']:+.4f} "
                  f"[{r['gain_lo95']:+.4f}, {r['gain_hi95']:+.4f}]")

    # ---- D1, against the band committed before any cell above existed ----
    rhos = {}
    for j in judges:
        sub = [x for x in rows if x["judge"] == j]
        rho, p = exact_p([x["control"] for x in sub], [x["gain"] for x in sub])
        rhos[j] = rho
        print(f"\n  D1 {j}: Spearman(control, gain) over {len(sub)} cells = {rho:+.3f}, "
              f"exact p = {p:.4f}")
    d1 = ("ANCHOR-BOUND" if all(r >= 0.5 for r in rhos.values()) else
          "ANTI" if all(r <= -0.5 for r in rhos.values()) else "UNRELATED")
    print(f"  D1 -> {d1}   (the band needs the sign in BOTH judges; shared noise biases it "
          f"negative, so a positive reading is the conservative one)")

    # ---- D3, the better-powered half: the family contrast is on the CONTROL, not the gain ----
    print()
    for j in judges:
        o = got.get((j, "open-ended"))
        c = got.get((j, "constrained"))
        if not (o and c):
            continue
        d = o["control"] - c["control"]
        print(f"  D3 {j}: MT-Bench control open-ended {o['control']:.3f} vs constrained "
              f"{c['control']:.3f}, difference {d:+.3f}  ->  "
              f"{'CEILING VISIBLE' if d > 0.10 else 'NO FAMILY EFFECT'}")
    print("\n  D2 is descriptive by pre-registration: no single cell above is quoted as a "
          "positive or a null, because none of them can resolve a gain of 0.05.")

    # ---- the artefact the pre-registration named but did not size (added after the run) ----
    paths = [a.alpaca_per_prompt, a.mtbench_per_prompt]
    dmaps = [domains(a.alpaca_corpus), domains(a.mtbench_corpus, MT_FAMILY)]
    nrows = []
    print()
    for j in judges:
        m, sd, draws = exchange_null(paths, dmaps, j, reps=a.null_reps, seed=a.seed % 10000)
        tail = sum(1 for d in draws if d <= rhos[j]) / len(draws)
        nrows.append(dict(judge=j, observed_rho=round(rhos[j], 4), null_mean=round(m, 4),
                          null_sd=round(sd, 4), p_vs_null=round(tail, 4), reps=a.null_reps))
        print(f"  D1 null {j}: observed {rhos[j]:+.3f} | no-effect null mean {m:+.3f} "
              f"sd {sd:.3f} | P(null <= observed) = {tail:.4f}")
    with open(os.path.join(a.out, "domain_breadth_null.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(nrows[0])); w.writeheader(); w.writerows(nrows)
    if d1 == "ANTI" and any(r["p_vs_null"] > 0.05 for r in nrows):
        print("\n  The band reads ANTI, which is the direction the pre-registration named as the\n"
              "  mechanically cheap one, and the no-effect null already produces most of it. The\n"
              "  correlation is NOT separable from the artefact: seven cells cannot resolve this,\n"
              "  and neither direction may be reported as a finding.")

# ---------------------------------------------------------------------------------------------
# Added after the run, and labelled as such: the pre-registration NAMED the shared-noise bias
# ("a positive rho is conservative and a negative one is the cheap direction") but committed no
# way of sizing it, and the reading came back ANTI -- the cheap direction. So the artefact has to
# be measured before the reading is used for anything.
#
# The null is exchangeability WITHIN a prompt: if selection does nothing, u_n1 and u_n8 are two
# draws of the same thing, so swapping them per prompt with probability 1/2 generates a world with
# no effect that keeps every cell's size, every prompt's pair of values, and -- crucially -- the
# same mechanical coupling between a cell's control and its gain. The distribution of Spearman
# under that null is the artefact. Anything the observed rho has beyond it is real.
def exchange_null(per_prompt_paths, dmaps, judge, reps=20000, seed=7):
    import statistics
    rng = random.Random(seed)
    pairs = []                                    # (cell, u_n1, u_n8)
    for path, dmap in zip(per_prompt_paths, dmaps):
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if r["judge"] == judge and r["prompt_id"] in dmap:
                pairs.append((dmap[r["prompt_id"]], float(r["u_n1"]), float(r["u_n8"])))
    cells_ = sorted({c for c, _, _ in pairs})
    out = []
    for _ in range(reps):
        acc = {c: [0.0, 0.0, 0] for c in cells_}
        for c, a, b in pairs:
            if rng.random() < 0.5:
                a, b = b, a
            acc[c][0] += a
            acc[c][1] += b - a
            acc[c][2] += 1
        xs = [acc[c][0] / acc[c][2] for c in cells_]
        ys = [acc[c][1] / acc[c][2] for c in cells_]
        out.append(spearman(xs, ys))
    return statistics.mean(out), statistics.pstdev(out), out

if __name__ == "__main__":
    main()
