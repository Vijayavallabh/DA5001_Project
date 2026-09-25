"""Review 2 Q6 (post hoc, descriptive, no band): does selection's utility live where its certificate is vacuous?

Proposition 1 certifies q(E) <= n p_s(E) for every event E, which excludes something about E only when
S(E) = -log p_s(E) > log n. The review asks whether the gains concentrate on events with S <= log n,
where the certificate says nothing.

Verifiable tasks (GSM8K, TriviaQA; the Comma-7B anchor's 64 cached draws per question): the event is
"the extracted answer is correct". The vote at n = 32 reads draws 0-31; the anchor's probability of the
event is estimated on the HELD-OUT draws 32-63 (pi_ho), so the stratifier never sees the draws the vote
used. Strata: pi_ho = 0 (no correct answer in 32 held-out draws, consistent with S > log 32) and
pi_ho >= 1/32 (S <= log 32 on the plug-in, the certificate vacuous for the answer), the latter split
at 1/4 and 1/2. The committed accuracies (0.320 -> 0.546 GSM8K, 0.280 -> 0.328 TriviaQA) are asserted.

Judged workload (the headline's 500 prompts): there is no answer event, so the event is the served
completion itself; S(y) = -sum log p_s over its tokens through the first end-of-text, read off the
pool's per-step log (output/phase5/sel_anchor64, the anchor's own probabilities at temperature 1.0).
The per-prompt judged gain is Table 2's judge-B pass (results/matched_h2h_per_prompt_matched_plain_B.csv,
u_sel_n64 - u_sel_n1), stratified by the served completion's S against log 64.

Writes results/utility_surprisal.csv.  Usage: .venv/bin/python analysis/utility_surprisal.py
"""
import argparse
import csv
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_scaling import load_rewards  # noqa: E402
from analysis.selection_verifiable import (correct_tqa, extract, extract_tqa, load_gsm8k,  # noqa: E402
                                           load_triviaqa, majority, read_gen)

CLASSES = ("neutral", "factual", "creative")
EOS = {128001, 128009}      # the Llama-3 vocabulary's end-of-text ids, shared by the anchor


def boot(xs, seed, reps=10000):
    rng = random.Random(seed)
    m = len(xs)
    d = sorted(sum(xs[rng.randrange(m)] for _ in range(m)) / m for _ in range(reps))
    return sum(xs) / m, d[int(0.025 * reps)], d[int(0.975 * reps)]


def strata_rows(task, items, ans, ok, n_vote, seed):
    rows, per = [], []
    for it in items:
        a = ans[it["qid"]]
        assert len(a) == 64, (it["qid"], len(a))
        c1 = float(ok(a[0], it["gold"]))
        cv = float(ok(a[majority(a[:n_vote])], it["gold"]))
        pi = sum(ok(x, it["gold"]) for x in a[n_vote:]) / (64 - n_vote)
        per.append((pi, c1, cv))
    acc1, accv = sum(x[1] for x in per) / len(per), sum(x[2] for x in per) / len(per)
    total_gain = accv - acc1
    bins = [("pi_ho = 0 (S > log n)", lambda p: p == 0),
            ("pi_ho in [1/32, 1/4) (S <= log n)", lambda p: 0 < p < 0.25),
            ("pi_ho in [1/4, 1/2)", lambda p: 0.25 <= p < 0.5),
            ("pi_ho in [1/2, 1]", lambda p: p >= 0.5),
            ("pi_ho >= 1/32, all (S <= log n)", lambda p: p > 0)]
    for name, f in bins:
        sub = [x for x in per if f(x[0])]
        if not sub:
            continue
        g, lo, hi = boot([x[2] - x[1] for x in sub], seed)
        rows.append(dict(task=task, n=n_vote, log_n=round(math.log(n_vote), 4), stratum=name,
                         questions=len(sub), acc_n1=round(sum(x[1] for x in sub) / len(sub), 4),
                         acc_vote=round(sum(x[2] for x in sub) / len(sub), 4), gain=round(g, 4),
                         gain_lo95=round(lo, 4), gain_hi95=round(hi, 4),
                         share_of_total_gain=round(sum(x[2] - x[1] for x in sub) / len(per) / total_gain, 4)
                         if total_gain else ""))
    rows.append(dict(task=task, n=n_vote, log_n=round(math.log(n_vote), 4), stratum="all", questions=len(per),
                     acc_n1=round(acc1, 4), acc_vote=round(accv, 4), gain=round(total_gain, 4)))
    return rows, acc1, accv


def served_surprisal(pool_dir, rewards, n):
    """prompt_id -> (S of the rank-0 draw, S of the argmax over the first n rewards), nats."""
    s = {}
    for cls in CLASSES:
        for line in open(os.path.join(pool_dir, f"trajectories_k0_{cls}.jsonl")):
            r = json.loads(line)
            m = r["metadata"]
            tot = 0.0
            for st in r["per_step_log"]:
                tot -= math.log(st["p_s_prob"])
                if st.get("sampled_token_id") in EOS:
                    break
            s.setdefault(m["prompt_id"], []).append((m["seed"], tot))
    out = {}
    for p, v in s.items():
        v.sort()                                  # seed order is load_candidates' order, which the rewards index
        rw = rewards[p][:n]
        j = max(range(len(rw)), key=lambda i: rw[i])
        out[p] = (v[0][1], v[j][1])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-dir", default="output/phase5/verifiable")
    ap.add_argument("--pool-dir", default="output/phase5/sel_anchor64")
    ap.add_argument("--rewards", default="results/selection_rewards64.csv")
    ap.add_argument("--judged", default="results/matched_h2h_per_prompt_matched_plain_B.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    rows = []
    _, gsm = load_gsm8k(500, 8)
    g = read_gen(os.path.join(a.gen_dir, "anchor_comma7b_n64.jsonl"))
    r, acc1, accv = strata_rows("gsm8k", gsm, {q: [extract(t) for t in v] for q, v in g.items()},
                                lambda p, gd: p == gd, 32, 61)
    assert (round(acc1, 3), round(accv, 3)) == (0.32, 0.546), (acc1, accv)   # the committed 0.320 -> 0.546
    rows += r
    _, tqa = load_triviaqa(500, 8)
    g = read_gen(os.path.join(a.gen_dir, "anchor_tqa_comma7b_n64.jsonl"))
    r, acc1, accv = strata_rows("triviaqa", tqa, {q: [extract_tqa(t) for t in v] for q, v in g.items()},
                                correct_tqa, 32, 62)
    assert (round(acc1, 3), round(accv, 3)) == (0.28, 0.328), (acc1, accv)   # the committed 0.280 -> 0.328
    rows += r

    rw = load_rewards(a.rewards)
    S = served_surprisal(a.pool_dir, rw, 64)
    u = {r_["prompt_id"]: (float(r_["u_sel_n64"]), float(r_["u_sel_n1"])) for r_ in csv.DictReader(open(a.judged))}
    pids = sorted(u)
    assert len(pids) == 500 and set(pids) <= set(S)
    ln = math.log(64)
    for name, f in (("served S > log 64", lambda s: s > ln), ("served S <= log 64", lambda s: s <= ln)):
        sub = [p for p in pids if f(S[p][1])]
        if not sub:
            rows.append(dict(task="judged headline (judge B)", n=64, log_n=round(ln, 4), stratum=name, questions=0))
            continue
        gm, lo, hi = boot([u[p][0] - u[p][1] for p in sub], 63)
        rows.append(dict(task="judged headline (judge B)", n=64, log_n=round(ln, 4), stratum=name,
                         questions=len(sub), acc_n1=round(sum(u[p][1] for p in sub) / len(sub), 4),
                         acc_vote=round(sum(u[p][0] for p in sub) / len(sub), 4), gain=round(gm, 4),
                         gain_lo95=round(lo, 4), gain_hi95=round(hi, 4),
                         share_of_total_gain=round(sum(u[p][0] - u[p][1] for p in sub)
                                                   / sum(u[p][0] - u[p][1] for p in pids), 4)))
    srt = sorted(S[p][1] for p in pids)
    srt1 = sorted(S[p][0] for p in pids)
    rows.append(dict(task="judged headline (judge B)", n=64, log_n=round(ln, 4), stratum="served S, nats",
                     questions=500, s_median_n1=round(srt1[250], 2), s_median=round(srt[250], 2),
                     s_p5=round(srt[25], 2), s_min=round(srt[0], 2)))
    os.makedirs(a.out, exist_ok=True)
    cols = ["task", "n", "log_n", "stratum", "questions", "acc_n1", "acc_vote", "gain", "gain_lo95", "gain_hi95",
            "share_of_total_gain", "s_median_n1", "s_median", "s_p5", "s_min"]
    with open(os.path.join(a.out, "utility_surprisal.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)
    for r_ in rows:
        print(r_)


if __name__ == "__main__":
    main()
