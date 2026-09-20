"""feat-106: the judge-free head-to-head, on a task with a checkable answer.

Section 6's comparison of the two mechanisms is a judged preference at three pairs. A judge-free one
did not exist, and `results/onset_prediction_verifiable.md` had to say why: it needs an anchor that
both shares the risky model's tokenizer -- only TinyComma does, among openly licensed safe models --
and can do the task. TinyComma scores `0.04` on GSM8K and cannot. It scores `0.07` on TriviaQA and
can, so this runs both mechanisms on the same 500 questions and scores them by exact-alias
containment instead of by a judge.

Both arms come from `h1.py` on `data/bench/triviaqa`, so neither mechanism gets a code path the
other does not:

  metered   --k-values -1 0 0.5 1 3 20 --trajectories-per-prompt 1
  selection --k-values 0              --trajectories-per-prompt 64   (the anchor alone, n draws)

The certificate each buys is `k * T_max` against `log n`, and the realised spend of the metered arm
is read off its own aggregate rather than assumed.

Usage:
  .venv/bin/python analysis/verifiable_metered.py --metered-dir output/phase5/tqa_metered \
    --selection-dir output/phase5/tqa_sel64 --corpus data/bench/triviaqa_factual.jsonl \
    --tag _tqa --out results
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_verifiable import (  # noqa: E402
    N_GRID, boot, boot_gain, correct_tqa, extract_lambada, extract_mmlu, extract_tqa,
    majority, spearman)

CLASSES = ("factual",)


def gold_map(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out[r["prompt_id"]] = {a for a in
                               (x.strip() for x in r["reference"].split("|||")) if a}
    return out


def arms(gen_dir):
    """k token (as h1 wrote it into the filename) -> {prompt_id: [generation, ...]}."""
    out = {}
    for path in sorted(glob.glob(os.path.join(gen_dir, "trajectories_k*.jsonl"))):
        base = os.path.basename(path)
        if not any(f"_{c}." in base for c in CLASSES):
            continue
        k = base.split("_")[1][1:]          # caution (o): the CLI string, verbatim
        by = out.setdefault(k, {})
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by.setdefault(r["metadata"]["prompt_id"], []).append(
                (r["aggregate"].get("generation") or "",
                 float(r["aggregate"].get("total_spend") or 0.0)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metered-dir", required=True)
    ap.add_argument("--selection-dir", required=True)
    ap.add_argument("--corpus", default="data/bench/triviaqa_factual.jsonl")
    ap.add_argument("--task", choices=("triviaqa", "mmlu", "lambada"), default="triviaqa",
                    help="selects the extractor and the correctness rule; the gold\n                         set comes from --corpus either way")
    ap.add_argument("--t-max", type=int, default=24)
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=8801)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    extract_fn, ok_fn = ((extract_mmlu, lambda p, g: p in g) if a.task == "mmlu"
                         else (extract_lambada, lambda p, g: p in g) if a.task == "lambada"
                         else (extract_tqa, correct_tqa))

    gold = gold_map(a.corpus)
    met, sel = arms(a.metered_dir), arms(a.selection_dir)
    assert sel, f"no selection arm in {a.selection_dir}"
    sel_k = sorted(sel, key=lambda s: abs(float(s)))[0]
    draws = sel[sel_k]
    pids = sorted(p for p in draws if p in gold)
    assert pids, "no prompt ids join the corpus"
    n_max = min(len(draws[p]) for p in pids)
    print(f"[vm] {len(pids)} questions, {n_max} draws each, metered arms {sorted(met)}", flush=True)

    def score(pred_by_pid):
        return [1.0 if ok_fn(pred_by_pid[p], gold[p]) else 0.0 for p in pids]

    rows = []
    # ---- selection: majority vote over the anchor's own draws, nested -------------------------
    ans = {p: [extract_fn(g) for g, _ in draws[p][:n_max]] for p in pids}
    base = None
    for n in [x for x in N_GRID if x <= n_max]:
        picked = {p: ans[p][majority(ans[p][:n])] for p in pids}
        c = score(picked)
        if n == 1:
            base = c
        acc, lo, hi = boot(c, a.reps, a.seed + n)
        g, glo, ghi = boot_gain(c, base, a.reps, a.seed + n)
        # Three numbers, because H2 of the pre-registration reads on "the same axes as Figure
        # 1(b)" and those axes are Table 1's: the metered decoder's realised KL against
        # selection's `log n - (n-1)/n`, the sharper KL bound for best-of-n (Beirami et al.),
        # which is what selection_decoding.py has always written and what 1.204 and 3.175 are.
        # `certificate_nats` is the PATHWISE `log n`, a stronger order and a different quantity;
        # `realised_nats` is 0.0 for selection because every served token is an anchor draw, so
        # the per-token meter reads zero by construction and cannot be the comparison axis.
        rows.append(dict(mechanism="selection (majority vote)", arm=f"n={n}",
                         certificate_nats=round(math.log(n), 4),
                         kl_nats=round(math.log(n) - (n - 1) / n, 4) if n > 1 else 0.0,
                         realised_nats=0.0,
                         n_questions=len(pids), acc=round(acc, 4), acc_lo95=round(lo, 4),
                         acc_hi95=round(hi, 4), gain=round(g, 4), gain_lo95=round(glo, 4),
                         gain_hi95=round(ghi, 4)))
    # ---- metered: one trajectory per question at each budget ---------------------------------
    zero = met.get("0") or met.get("0.0")
    ctrl = None
    if zero:
        ctrl = score({p: extract_fn(zero[p][0][0]) for p in pids if p in zero})
    for k in sorted(met, key=lambda s: float(s)):
        by = met[k]
        if any(p not in by for p in pids):
            continue
        c = score({p: extract_fn(by[p][0][0]) for p in pids})
        spend = sum(by[p][0][1] for p in pids) / len(pids)
        acc, lo, hi = boot(c, a.reps, a.seed)
        rec = dict(mechanism="metered decoder", arm=f"k={k}",
                   certificate_nats=(round(float(k) * a.t_max, 4) if float(k) > 0 else 0.0),
                   kl_nats=round(spend, 4),
                   realised_nats=round(spend, 4), n_questions=len(pids), acc=round(acc, 4),
                   acc_lo95=round(lo, 4), acc_hi95=round(hi, 4))
        if ctrl is not None:
            g, glo, ghi = boot_gain(c, ctrl, a.reps, a.seed)
            rec.update(gain=round(g, 4), gain_lo95=round(glo, 4), gain_hi95=round(ghi, 4))
        else:
            rec.update(gain="", gain_lo95="", gain_hi95="")
        rows.append(rec)

    for m in ("selection (majority vote)", "metered decoder"):
        sub = [r for r in rows if r["mechanism"] == m and r["certificate_nats"]]
        rho = spearman([math.log(r["certificate_nats"]) for r in sub],
                       [r["acc"] for r in sub]) if len(sub) > 2 else ""
        for r in rows:
            if r["mechanism"] == m:
                r["spearman_acc_logcert"] = round(rho, 4) if rho != "" else ""

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"verifiable_metered{a.tag}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[vm] wrote {path}")
    for r in rows:
        print(f"  {r['mechanism']:26s} {r['arm']:8s} cert={r['certificate_nats']:>9} "
              f"spend={r['realised_nats']:>9} acc={r['acc']:.4f} "
              f"[{r['acc_lo95']:.4f}, {r['acc_hi95']:.4f}] gain={r['gain']}")


if __name__ == "__main__":
    main()
