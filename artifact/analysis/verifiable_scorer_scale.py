"""feat-118: does scorer-scale saturation survive without a judge?

feat-117 found that a 1.5B reward beats a 0.5B one at n=64 and that 3B and 7.6B buy nothing more.
Every number in that result came from a pairwise LLM judge whose order consistency this paper
measures at 0.24-0.35 and calls UNUSABLE. GSM8K exact match has no judge: the answer is a number
and the grader is `==`.

This scores the bands in results/onset_prediction_verifiable_scorer_scale.md against the four
per-scorer runs of analysis/selection_verifiable.py. It recomputes per-problem correctness from the
cached generations and the cached reward CSVs, because the committed per-arm CSVs hold aggregates
and every band here is a PAIRED comparison between two scorers on the same problems.

H0 is an internal gate with no GPU: the accuracy this script recomputes must equal what
selection_verifiable.py committed, for every scorer and every n. If it does not, the two are
reading the caches differently and no band may be quoted.

Shape is deliberately not the test. On this axis the 7.6B reward is already non-monotone
(Spearman 0.9286 against log n) and majority vote itself falls from 0.546 at n=32 to 0.542 at
n=64, so only levels at n=64 are read.

Writes <out>/verifiable_scorer_scale{,_bands}.csv. No GPU: cached JSONL and CSVs only.

Usage:
  .venv/bin/python analysis/verifiable_scorer_scale.py --out results
"""
import argparse
import csv
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_verifiable import (correct_tqa, extract, extract_tqa,  # noqa: E402
                                           load_gsm8k, load_triviaqa, majority, read_gen, spearman)

N_GRID = (1, 2, 4, 8, 16, 32, 64)
N_BOOT = 10000
# tag, parameters in billions counted off the checkpoint, reward-cache suffix
SCORERS = [("0.5B", 0.4940, "_qwen05b"), ("1.5B", 1.5437, "_qwen15b"),
           ("3B", 3.0859, "_qwen3b"), ("7.6B", 7.6156, "")]
TASKS = [("gsm8k", "_comma7b"), ("triviaqa", "_tqa_comma7b")]


def load_rewards(path):
    by = {}
    for r in csv.DictReader(open(path)):
        by.setdefault(r["prompt_id"], []).append((int(r["rank"]), float(r["reward"])))
    return {q: [s for _, s in sorted(v)] for q, v in by.items()}


def paired_boot(diffs, rng, n_boot=N_BOOT):
    n = len(diffs)
    xs = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return xs[int(0.025 * n_boot)], xs[int(0.975 * n_boot)]


def task_correct(task, tag, out_dir, gen_dir, limit, max_n):
    """(scorer tag -> {n: [0/1 per problem]}) plus majority vote, on one task."""
    if task == "gsm8k":
        _, items = load_gsm8k(limit, 8)
        pick, ok = extract, lambda p, g: p == g
    else:
        _, items = load_triviaqa(limit, 5)
        pick, ok = extract_tqa, correct_tqa
    gens = read_gen(os.path.join(gen_dir, f"anchor{tag}_n{max_n}.jsonl"))
    assert all(len(gens[it["qid"]]) == max_n for it in items), "ragged generation file"
    ans = {q: [pick(t) for t in v] for q, v in gens.items()}

    out = {}
    for name, _, sfx in SCORERS:
        rw = load_rewards(os.path.join(out_dir, f"selection_verifiable_rewards{tag}{sfx}.csv"))
        out[name] = {n: [1.0 if ok(ans[it["qid"]][max(range(n), key=lambda j: rw[it["qid"]][j])],
                                   it["gold"]) else 0.0 for it in items] for n in N_GRID}
    out["majority vote"] = {n: [1.0 if ok(ans[it["qid"]][majority(ans[it["qid"]][:n])], it["gold"])
                                else 0.0 for it in items] for n in N_GRID}
    return out, len(items)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-dir", default="output/phase5/verifiable")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--max-n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=13109)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    rows, bands, gate = [], [], []
    C = {}
    for task, tag in TASKS:
        C[task], n_items = task_correct(task, tag, a.out, a.gen_dir, a.limit, a.max_n)
        print(f"[vss] {task}: {n_items} problems x {a.max_n} candidates x {len(SCORERS)} scorers",
              flush=True)

        # H0: what we recompute must equal what selection_verifiable.py committed.
        for name, pf, sfx in SCORERS:
            committed = {int(r["n"]): float(r["acc"])
                         for r in csv.DictReader(
                             open(os.path.join(a.out, f"selection_verifiable{tag}{sfx}.csv")))
                         if r["arm"].startswith("pointwise")}
            for n in N_GRID:
                mine = sum(C[task][name][n]) / n_items
                if abs(mine - committed[n]) > 5e-4:
                    gate.append(f"{task} {name} n={n}: {mine:.4f} vs {committed[n]:.4f}")

        for name, pf, _ in SCORERS + [("majority vote", "", "")]:
            accs = [sum(C[task][name][n]) / n_items for n in N_GRID]
            rho = spearman([math.log(n) for n in N_GRID], accs)
            for n, acc in zip(N_GRID, accs):
                rows.append(dict(task=task, scorer=name, params_b=pf, n=n,
                                 nats_certified=round(math.log(n), 4), acc=round(acc, 4),
                                 n_problems=n_items, spearman_acc_logn=round(rho, 4)))

    bands.append(dict(band="H0 recomputation matches the committed per-arm CSVs", quantity="",
                      value="", lo95="", hi95="",
                      reading="MATCHES" if not gate else "FAILED " + "; ".join(gate[:4])))
    if gate:
        print("  H0 FAILED -- no band below may be quoted:", *gate[:4], sep="\n    ")

    tags = [s[0] for s in SCORERS]
    readings = {}
    for task, _ in TASKS:
        d = [x - y for x, y in zip(C[task]["7.6B"][64], C[task]["0.5B"][64])]
        m, (lo, hi) = sum(d) / len(d), paired_boot(d, rng)
        bands.append(dict(band=f"H1 scorer scale matters at all ({task})",
                          quantity="7.6B - 0.5B at n=64", value=round(m, 4), lo95=round(lo, 4),
                          hi95=round(hi, 4), reading="SEPARATES" if not (lo <= 0 <= hi) else "FLAT"))
        readings[(task, "H1")] = bands[-1]["reading"]
        sep = []
        for x, y in zip(tags, tags[1:]):
            d = [p - q for p, q in zip(C[task][y][64], C[task][x][64])]
            m, (lo, hi) = sum(d) / len(d), paired_boot(d, rng)
            r = "SEPARATES" if not (lo <= 0 <= hi) else "-"
            sep.append((f"{y} over {x}", r))
            bands.append(dict(band=f"H2 adjacent step, {y} over {x} ({task})",
                              quantity=f"{y} - {x} at n=64", value=round(m, 4), lo95=round(lo, 4),
                              hi95=round(hi, 4), reading=r))
        got = [k for k, r in sep if r == "SEPARATES"]
        if got == ["1.5B over 0.5B"]:
            h3 = "AGREES"
        elif not got and readings[(task, "H1")] == "FLAT":
            h3 = "DISAGREES, NO SCORER EFFECT"
        elif any(k in got for k in ("3B over 1.5B", "7.6B over 3B")):
            h3 = "DISAGREES, LATER SATURATION"
        else:
            h3 = "DISAGREES, OTHER"
        bands.append(dict(band=f"H3 the two axes agree ({task})",
                          quantity="separating adjacent steps: " + (", ".join(got) or "none"),
                          value="", lo95="", hi95="", reading=h3))

    for task, _ in TASKS:
        for name, _, _ in SCORERS:
            r = [x for x in rows if x["task"] == task and x["scorer"] == name][0]
            bands.append(dict(band=f"H5 shape, {name} ({task})", quantity="Spearman(acc, log n)",
                              value=r["spearman_acc_logn"], lo95="", hi95="",
                              reading="MONOTONE" if r["spearman_acc_logn"] == 1.0
                                      else "NOT MONOTONE (the 7.6B reference is 0.9286)"))

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "verifiable_scorer_scale.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(a.out, "verifiable_scorer_scale_bands.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(bands[0])); w.writeheader(); w.writerows(bands)

    print()
    for task, _ in TASKS:
        for name in tags + ["majority vote"]:
            acc = [x for x in rows if x["task"] == task and x["scorer"] == name]
            print(f"  {task:9s} {name:14s} " +
                  " ".join(f"n{r['n']}={r['acc']:.3f}" for r in acc) +
                  f"   rho {acc[0]['spearman_acc_logn']:+.4f}")
    print()
    for b in bands:
        print(f"  {b['band']:44s} {str(b['value']):>8} [{b['lo95']}, {b['hi95']}]  {b['reading']}")
    print(f"wrote {os.path.join(a.out, 'verifiable_scorer_scale.csv')}")


if __name__ == "__main__":
    main()
