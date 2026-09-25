"""Seventh review round: four checks the referees asked for that need no GPU and no judge call.

1. onset_window_threshold.csv (review 3, W12). Figure 2(a)'s sweeps decode every passage for T tokens,
   the longest target in the sweep (analysis/composition_attack.py, single mode), so a budget of k per
   token certifies K = kT and a 50-token window at the passage's own rate s is exposed from k = 50 s / T,
   a fraction 50/T of s(x). The caption said "about 0.8 s(x)"; 0.8 is the ABSOLUTE k at which a median
   window is exposed at T_max = 200 (results/window_vacuity.csv), a different quantity in other units.
2. opponent_degeneracy.csv (+ _examples) (review 3, Q11). Is the 70B base at temperature 1.0, judged no
   better than the anchor, degenerate? Length, empties, repetition and script mixing for the 70B base,
   the 8B-Instruct and the anchor, the same 500 ordinary prompts, lowest seed, de-echoed text.
3. audit_cost.csv (review 3, W9a/Q7). What a rights-holder pays to certify near-verbatim reproduction
   per WORK rather than for the average passage: 3n/eps anchor draws each, priced at the measured
   anchor-only serving time (results/anchor_only_cost.csv).
5. cpk_certificate.csv (review 1, Q1). CP-k run against the risky model accepts a complete output y only if
   its realised log-ratio log p_r(y)/p_s(y) is at most kappa, and certifies kappa + log(1/acceptance).
   Read off the risky model's own 1,500 ordinary trajectories (output/sweep_plain, k=-1), pooled over
   prompts: the best certificate over kappa, and where it sits against the window's 159.8 nats.
4. frontier_ratio.csv (review 3, W4/Q10; review 1, Q5). Proposition 4's ratio gamma*T / log(1/pi) read on
   the SAME order-averaged instrument as Figure 1(d)'s dotted frontier (the anchor's per-prompt levels of
   results/frontier_levels_per_prompt_*.csv), pooled and per prompt class, and Lambda* at the utility each
   meter arm actually reaches, not only at the maximum.

Usage: .venv/bin/python analysis/review7_checks.py --out results
"""
import argparse
import csv
import glob
import json
import math
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.utility import degeneracy, load_arm  # noqa: E402
from analysis.order_averaged_h2h import lowest_seed  # noqa: E402

PAIRS = "results/onset_theory_pairs.tsv"


def write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


def window_thresholds(out):
    onset = {r["pair"]: r for r in csv.DictReader(open("results/onset_table.csv"))}
    rows = []
    for line in open(PAIRS):
        pair, _, bp, *_ = line.rstrip("\n").split("\t")
        if pair not in onset:
            continue                                        # ladder rungs are not in Figure 2(a)
        n = [int(r["n_target"]) for r in csv.DictReader(open(bp))]
        T = max(n)
        rows.append(dict(pair=pair, passages=len(n), T_decoded=T, n_target_median=st.median(n),
                         window_threshold_over_s=round(50 / T, 4),
                         median_passage_threshold_over_s=round(st.median(n) / T, 4),
                         onset_over_s=onset[pair]["ratio"]))
    write(os.path.join(out, "onset_window_threshold.csv"), rows)
    return rows


def opponent_degeneracy(out):
    arms = {"70B base, k=-1": ("output/phase5/imit_llama70b", "-1"),
            "8B-Instruct, k=-1": ("output/sweep_plain", "-1"),
            "anchor alone, k=0": ("output/sweep_plain", "0")}
    rows, ex = [], []
    for name, (d, tok) in arms.items():
        got = lowest_seed(load_arm(d, tok, "kl", deecho=True))
        texts = [got[p][1] for p in sorted(got)]
        words = [len(t.split()) for t in texts]
        d3 = [degeneracy(t) for t in texts]
        nonascii = [sum(ord(c) > 127 for c in t) / max(1, len(t)) for t in texts]
        rows.append(dict(arm=name, prompts=len(texts), median_words=st.median(words),
                         empty_pct=round(100 * sum(w == 0 for w in words) / len(words), 1),
                         distinct3_median=round(st.median(x for x, _ in d3), 4),
                         top5gram_share_median=round(st.median(y for _, y in d3), 4),
                         looping_pct=round(100 * sum(y >= 0.2 for _, y in d3) / len(d3), 1),
                         nonascii_over_10pct=round(100 * sum(x > 0.10 for x in nonascii) / len(nonascii), 1)))
        if name.startswith("70B"):
            for p in sorted(got)[:3]:
                ex.append(dict(arm=name, prompt_id=p, first_40_words=" ".join(got[p][1].split()[:40])))
    write(os.path.join(out, "opponent_degeneracy.csv"), rows)
    # raw generations stay out of results/ (committed, public): the examples carry forum handles the
    # base model reproduces, so they go to the gitignored output/ tree beside the runs they come from
    exdir = os.path.join(ROOT, "output", "review_audit")
    os.makedirs(exdir, exist_ok=True)
    write(os.path.join(exdir, "opponent_degeneracy_examples.csv"), ex)


def audit_cost(out):
    cost = {int(r["width"]): r for r in csv.DictReader(open("results/anchor_only_cost.csv"))}
    sec_per_draw = float(cost[200]["C_plain_anchor_alone_s"]) / 200   # 200 prompts x one draw, one card
    rows = []
    for n in (8, 64):
        for eps in (0.01, 0.001):
            draws = math.ceil(3 * n / eps)
            for works in (1, 10**4, 10**5, 10**6):
                rows.append(dict(n=n, eps=eps, draws_per_work=draws, works=works,
                                 total_draws=draws * works,
                                 gpu_hours=round(draws * works * sec_per_draw / 3600, 2),
                                 sec_per_draw=round(sec_per_draw, 5)))
    write(os.path.join(out, "audit_cost.csv"), rows)


def frontier_ratio(out):
    per, cls = {}, {}
    for f in sorted(glob.glob("results/frontier_levels_per_prompt_*.csv")):
        for r in csv.DictReader(open(f)):
            per.setdefault(r["prompt_id"], {}).update({k: float(v) for k, v in r.items() if k.startswith("u_")})
    for (p, _), (c, _, _) in load_arm("output/sweep_plain", "0", "kl").items():
        cls[p] = c
    spend = {r["arm"]: float(r["x_nats"]) for r in csv.DictReader(open("results/frontier_levels.csv"))}

    def lam_star(law, u):
        vals, n = sorted(set(law)), len(law)
        probs = [law.count(v) / n for v in vals]
        best = 0.0
        for j in range(1, 20001):
            lam = j * 0.005
            best = max(best, lam * u - math.log(sum(q * math.exp(lam * v) for v, q in zip(vals, probs))))
        return best

    rows = []
    groups = {"all": list(per)} | {c: [p for p in per if cls.get(p) == c] for c in ("neutral", "factual", "creative")}
    for g, pids in groups.items():
        law = [per[p]["u_anchor_k0"] for p in pids]
        pi = sum(u == 1.0 for u in law) / len(law)
        # pi = 0 when the anchor never wins BOTH presentation orders in the group: Lambda*(u_max) is then
        # infinite and Proposition 4's conservative ratio says nothing, which is itself the answer.
        row = dict(group=g, prompts=len(pids), anchor_level=round(st.mean(law), 4), pi_wins_both_orders=round(pi, 4),
                   u_max_anchor=max(law), log_inv_pi=round(-math.log(pi), 4) if pi > 0 else "inf",
                   spend_k10=spend["met_k10"],
                   ratio_at_u_max=round(spend["met_k10"] / -math.log(pi), 1) if pi > 0 else 0.0)
        for k in ("0.5", "1", "3", "10"):
            u = st.mean(per[p][f"u_met_k{k}"] for p in pids)
            ls = lam_star(law, u)
            row[f"level_k{k}"] = round(u, 4)
            row[f"lambda_star_k{k}"] = round(ls, 5)
            row[f"ratio_k{k}"] = round(spend[f"met_k{k}"] / ls, 1) if ls > 0 else ""
        rows.append(row)
    write(os.path.join(out, "frontier_ratio.csv"), rows)


def cpk_certificate(out):
    from analysis.window_logratio import trajectory_steps
    tot = []
    for c in ("neutral", "factual", "creative"):
        for line in open(f"output/sweep_plain/trajectories_k-1_{c}.jsonl"):
            tot.append(sum(x for x, _ in trajectory_steps(json.loads(line))))
    tot.sort()
    n = len(tot)
    best = min(((tot[i] + math.log(n / (i + 1)), tot[i], (i + 1) / n) for i in range(n)), key=lambda t: t[0])
    rows = [dict(quantity="best certificate over kappa, nats", value=round(best[0], 2), kappa=round(best[1], 2),
                 acceptance=round(best[2], 4), trajectories=n),
            dict(quantity="median realised log-ratio of a whole output, nats", value=round(st.median(tot), 2),
                 kappa="", acceptance="", trajectories=n)]
    for acc in (0.01, 0.1, 0.5, 0.9):
        i = max(0, math.ceil(acc * n) - 1)
        rows.append(dict(quantity=f"certificate at acceptance {acc}", value=round(tot[i] + math.log(n / (i + 1)), 2),
                         kappa=round(tot[i], 2), acceptance=round((i + 1) / n, 4), trajectories=n))
    write(os.path.join(out, "cpk_certificate.csv"), rows)


def query_caps(out):
    """Review 1 Q7 / review 2 Q9 / review 3 W9(b): the per-user query cap an untrusted scorer forces,
    floor(S / log n), for the shortest events the paper protects -- a 10- and a 50-token window at their
    median anchor surprisal (results/window_vacuity.csv) and a TriviaQA answer (per-question anchor
    surprisal at temperature 0.7, results/tqa_vacuity_t07.csv), at n = 8 and 64."""
    rows = []
    wv = {r["window_tokens"]: float(r["S_median_nats"]) for r in csv.DictReader(open("results/window_vacuity.csv"))}
    for w in ("10", "50"):
        for n in (8, 64):
            rows.append(dict(event=f"{w}-token window, median", S_nats=round(wv[w], 2), n=n,
                             cap_median=math.floor(wv[w] / math.log(n)), frac_no_query=0.0))
    tq = [float(r["s_anchor_nats"]) for r in csv.DictReader(open("results/tqa_vacuity_t07.csv"))]
    for n in (8, 64):
        caps = [math.floor(s / math.log(n)) for s in tq]
        rows.append(dict(event="TriviaQA answer, median over questions", S_nats=round(st.median(tq), 2), n=n,
                         cap_median=st.median(caps), frac_no_query=round(sum(c < 1 for c in caps) / len(caps), 3)))
    write(os.path.join(out, "query_caps.csv"), rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    window_thresholds(a.out)
    opponent_degeneracy(a.out)
    audit_cost(a.out)
    frontier_ratio(a.out)
    cpk_certificate(a.out)
    query_caps(a.out)


if __name__ == "__main__":
    main()
