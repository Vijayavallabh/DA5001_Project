"""feat-150: an interval on the LIFT RATIOS, which the paper quotes as bare point estimates.

Two referee reports ask for uncertainty on a ratio of two estimates. The paper says majority vote
lifts GSM8K "3.4x the pointwise reward" and lifts TriviaQA "4.2x less", and quotes neither with an
interval -- each is a quotient of two bootstrapped gains, so the reader cannot tell a 3.4x that
excludes 1 from one that does not.

WHY A PAIRED BOOTSTRAP AND NOT FIELLER. Fieller's theorem is the textbook interval for a ratio of
means, and it needs two things we would have to invent: approximate bivariate normality of the two
gains, and their correlation. Both arms here select from the SAME n draws on the SAME questions, so
that correlation is large and unknown. Resampling questions and recomputing both gains on each
resample carries the correlation exactly, assumes no distribution, and costs nothing. Fieller is
reported alongside, computed from the same resamples' moments, only to show the two agree where the
denominator is comfortably away from zero -- and to make the point that it is the weaker tool here.

Reuses the scorer's own extract/majority/gain code so these numbers are the same quantity
results/selection_verifiable_*.csv reports, not merely a similar one (caution (v)).

Reads:  output/phase5/verifiable/anchor_{,tqa_}comma7b_n64.jsonl
        results/selection_verifiable_rewards{,_tqa}_comma7b.csv
Writes: <out>/lift_ratio_interval.csv

No GPU, no generation, no judge.

Usage:
  .venv/bin/python analysis/lift_ratio_interval.py --out results
"""
import argparse
import csv
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_verifiable import (  # noqa: E402
    correct_tqa, extract, extract_tqa, majority, norm_answer, read_gen)

TASKS = {
    "gsm8k": dict(gen="output/phase5/verifiable/anchor_comma7b_n64.jsonl",
                  rewards="results/selection_verifiable_rewards_comma7b.csv",
                  vote_n=32, reward_n=64),
    "triviaqa": dict(gen="output/phase5/verifiable/anchor_tqa_comma7b_n64.jsonl",
                     rewards="results/selection_verifiable_rewards_tqa_comma7b.csv",
                     vote_n=64, reward_n=64),
}
T = 1.96


def gold_gsm8k(limit, n_shot):
    from analysis.selection_verifiable import load_gsm8k
    return {it["qid"]: it["gold"] for it in load_gsm8k(limit, n_shot)[1]}


def gold_tqa(limit, n_shot):
    from analysis.selection_verifiable import load_triviaqa
    return {it["qid"]: it["gold"] for it in load_triviaqa(limit, n_shot)[1]}


def load_rewards(path):
    by = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        by.setdefault(r["prompt_id"], {})[int(r["rank"])] = float(r["reward"])
    return {k: [v[i] for i in sorted(v)] for k, v in by.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=8801)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    rows = []
    draws = {}
    for task, cfg in TASKS.items():
        gens = read_gen(cfg["gen"])
        rewards = load_rewards(cfg["rewards"])
        if task == "gsm8k":
            gold, pick, ok = gold_gsm8k(a.limit, 8), extract, lambda p, g: p == g
        else:
            gold, pick, ok = gold_tqa(a.limit, 8), extract_tqa, correct_tqa
        qids = [q for q in sorted(gens) if q in gold and q in rewards]
        assert qids, task

        def score(sel):
            """sel(qid) -> index of the served draw; returns the 0/1 vector over qids."""
            out = []
            for q in qids:
                ans = [pick(t) for t in gens[q]]
                out.append(1.0 if ok(ans[sel(q, ans)], gold[q]) else 0.0)
            return out

        base = score(lambda q, ans: 0)
        vote = score(lambda q, ans: majority(ans[:cfg["vote_n"]]))
        rew = score(lambda q, ans: max(range(cfg["reward_n"]),
                                       key=lambda i: rewards[q][i]))

        m = len(qids)
        g_vote = (sum(vote) - sum(base)) / m
        g_rew = (sum(rew) - sum(base)) / m

        rng = random.Random(a.seed)
        rv, rr, ratios = [], [], []
        for _ in range(a.reps):
            idx = [rng.randrange(m) for _ in range(m)]
            dv = sum(vote[i] - base[i] for i in idx) / m
            dr = sum(rew[i] - base[i] for i in idx) / m
            rv.append(dv)
            rr.append(dr)
            if dr > 0:                      # a ratio is undefined where the denominator is not
                ratios.append(dv / dr)
        ratios.sort()
        undef = 1.0 - len(ratios) / a.reps
        lo, hi = ratios[int(0.025 * len(ratios))], ratios[int(0.975 * len(ratios))]
        # Caution (g): an interval computed over only the resamples where the quantity is DEFINED
        # is conditioned on exactly the draws that behaved, and at this rate it is not an interval
        # at all. TriviaQA's reward gain is negative, so its vote-over-reward ratio is undefined on
        # most resamples; the interval is suppressed rather than printed with a footnote.
        defined = undef <= 0.5

        # Fieller from the same resamples' moments, for comparison only.
        def msd(x):
            mu = sum(x) / len(x)
            return mu, math.sqrt(sum((v - mu) ** 2 for v in x) / (len(x) - 1))
        mv, sv = msd(rv)
        mr, sr = msd(rr)
        cov = sum((x - mv) * (y - mr) for x, y in zip(rv, rr)) / (len(rv) - 1)
        t = 1.96
        disc = (g_vote * g_rew - t * t * cov) ** 2 - \
               (g_rew ** 2 - t * t * sr ** 2) * (g_vote ** 2 - t * t * sv ** 2)
        den = g_rew ** 2 - t * t * sr ** 2
        if disc > 0 and den > 0:
            root = math.sqrt(disc)
            f_lo = (g_vote * g_rew - t * t * cov - root) / den
            f_hi = (g_vote * g_rew - t * t * cov + root) / den
        else:
            f_lo = f_hi = float("nan")     # Fieller is unbounded: the denominator is not resolved

        draws[task] = dict(vote=rv, reward=rr, g_vote=g_vote, g_rew=g_rew, m=m)
        rows.append(dict(
            task=task, n_questions=m,
            vote_n=cfg["vote_n"], reward_n=cfg["reward_n"],
            gain_vote=round(g_vote, 4), gain_reward=round(g_rew, 4),
            ratio=(round(g_vote / g_rew, 3) if g_rew and defined else ""),
            ratio_lo95=round(lo, 3) if defined else "",
            ratio_hi95=round(hi, 3) if defined else "",
            undefined_frac=round(undef, 4),
            corr_of_gains=round(cov / (sv * sr), 3),
            fieller_lo95="" if math.isnan(f_lo) else round(f_lo, 3),
            fieller_hi95="" if math.isnan(f_hi) else round(f_hi, 3),
            excludes_one=("UNDEFINED -- denominator not resolved" if not defined
                          else "yes" if lo > 1 or hi < 1 else "no"),
        ))
        if defined:
            print(f"[lr] {task:9s} n={m}  vote {g_vote:+.4f}  reward {g_rew:+.4f}  "
                  f"ratio {g_vote/g_rew:.3f} [{lo:.3f}, {hi:.3f}]  "
                  f"(undefined on {100*undef:.2f}% of resamples, gain corr {cov/(sv*sr):+.3f}; "
                  f"Fieller [{f_lo:.3f}, {f_hi:.3f}])", flush=True)
        else:
            print(f"[lr] {task:9s} n={m}  vote {g_vote:+.4f}  reward {g_rew:+.4f}  "
                  f"RATIO NOT QUOTABLE: the denominator is negative and the quotient is undefined "
                  f"on {100*undef:.1f}% of resamples", flush=True)

    # ACROSS-TASK: the vote lift on GSM8K against the vote lift on TriviaQA. Disjoint question
    # sets, so the two bootstrap series are independent by construction and are simply paired
    # INDEX-WISE here to form the ratio -- which is valid precisely because they are independent.
    gv, tv = draws["gsm8k"]["vote"], draws["triviaqa"]["vote"]
    xr = sorted(x / y for x, y in zip(gv, tv) if y > 0)
    undef = 1.0 - len(xr) / len(gv)
    g_ratio = draws["gsm8k"]["g_vote"] / draws["triviaqa"]["g_vote"]
    rows.append(dict(
        task="gsm8k_over_triviaqa (vote lift, ACROSS tasks)",
        n_questions=draws["gsm8k"]["m"] + draws["triviaqa"]["m"],
        vote_n="", reward_n="",
        gain_vote=round(draws["gsm8k"]["g_vote"], 4),
        gain_reward=round(draws["triviaqa"]["g_vote"], 4),
        ratio=round(g_ratio, 3),
        ratio_lo95=round(xr[int(0.025 * len(xr))], 3),
        ratio_hi95=round(xr[int(0.975 * len(xr))], 3),
        undefined_frac=round(undef, 4), corr_of_gains=0.0,
        fieller_lo95="", fieller_hi95="",
        excludes_one="yes" if xr[int(0.025 * len(xr))] > 1 else "no"))
    print(f"[lr] ACROSS   GSM8K vote {draws['gsm8k']['g_vote']:+.4f} / TriviaQA vote "
          f"{draws['triviaqa']['g_vote']:+.4f} = {g_ratio:.3f} "
          f"[{xr[int(0.025*len(xr))]:.3f}, {xr[int(0.975*len(xr))]:.3f}]  "
          f"(undefined on {100*undef:.2f}% of resamples)", flush=True)

    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "lift_ratio_interval.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[lr] wrote {p}")


if __name__ == "__main__":
    main()
