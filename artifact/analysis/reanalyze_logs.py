"""Reanalysis of the released Anchored-Decoding logs (feat-002).

Reads output/h1_outputs/trajectories_k{k}_{split}.jsonl and writes to --out:
  regime_table.csv              per (k, split): constraint activity, utilisation, invariant checks,
                                identical-generation fraction vs. the next larger k on the same split
  llr_tails.csv                 per (k, split): realised log-likelihood-ratio L(y) = sum_t log p*/p_s tails
  prefix_debt_forced_tokens.csv per (k, split): delta_init and leading tokens forced to the safe model
  surprisal.csv                 per (k, split): safe-model surprisal of the generated text and the cap (K+ln2)/S
  seed_collisions.csv           per held-out row in output/h2*_outputs/heldout_validation.jsonl: duplicated spends
  per_trajectory.csv            one row per trajectory (inputs for feat-007 / feat-013)

Usage: .venv/bin/python analysis/reanalyze_logs.py --logs output --out results
"""
import argparse, csv, glob, json, math, os, re, statistics as st

EPS = 1e-3  # tolerance for the per-trajectory invariant Z <= max(0, B) and per-step a_t <= k_t
LN2 = math.log(2)
FNAME = re.compile(r"trajectories_k(?P<k>-?[0-9.]+)_(?P<split>[a-z_]+)\.jsonl$")


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def load(path):
    """One dict per trajectory, keyed by (prompt_id, seed)."""
    recs = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        m, a, steps = r["metadata"], r["aggregate"], r["per_step_log"]
        Z, B, gl = a["total_spend"], a["final_budget"], a["generation_length_tokens"]
        d = dict(prompt_id=m["prompt_id"], seed=m["seed"], k=m["k"], K=m["K"], split=m["split"],
                 Z=Z, B=B, util=a["budget_utilization"], gen_len=gl,
                 debt=r["prefix_analysis"]["true_prefix_debt"], rouge=a["rouge_l"], mh=a["minhash_5gram"],
                 n_active=0, n_forced=0, n_free=0, nsteps=0, max_a=0.0, viol_step=0, max_step_over=0.0,
                 L=0.0, Lmax=-math.inf, L_risky=0.0, S_safe=0.0, S_star=0.0, lead_forced=0)
        lead = True
        for s in steps[:gl]:
            at, kt, bd = s["a_t"], s["k_t"], s["bd"]
            if kt is None or bd is None:
                continue
            d["nsteps"] += 1
            d["max_a"] = max(d["max_a"], at)
            if at > kt + EPS:
                d["viol_step"] += 1
                d["max_step_over"] = max(d["max_step_over"], at - kt)
            if bd <= 1e-6:
                d["n_forced"] += 1
                if lead:
                    d["lead_forced"] += 1
            else:
                lead = False
                d["n_free" if bd >= 1 - 1e-6 else "n_active"] += 1
            ps, pst, pr = s.get("p_s_prob"), s.get("p_star_prob"), s.get("p_risky_prob")
            if ps and pst and ps > 0 and pst > 0:
                lr = math.log(pst) - math.log(ps)
                d["L"] += lr
                d["Lmax"] = max(d["Lmax"], lr)
                d["S_safe"] += -math.log(ps)
                d["S_star"] += -math.log(pst)
            if ps and pr and ps > 0 and pr > 0:
                d["L_risky"] += math.log(pr) - math.log(ps)
        d["invariant_viol"] = int(Z > max(0.0, B) + EPS)
        d["gen"] = a["generation"]
        recs[(m["prompt_id"], m["seed"])] = d
    return recs


def regime_row(k, K, split, R, identical_next):
    n = len(R)
    tot = sum(r["nsteps"] for r in R)
    act = sum(r["n_active"] for r in R)
    forced = sum(r["n_forced"] for r in R)
    free = sum(r["n_free"] for r in R)
    util = [r["util"] for r in R]
    return dict(k=k, K=K, split=split, n_traj=n, n_steps=tot,
                active_pct=round(100 * act / tot, 3), forced_safe_pct=round(100 * forced / tot, 3),
                risky_unchanged_pct=round(100 * free / tot, 3),
                traj_with_active_step=sum(r["n_active"] > 0 for r in R),
                traj_with_forced_step=sum(r["n_forced"] > 0 for r in R),
                identical_gen_vs_next_k_pct=identical_next,
                Z_mean=round(st.mean([r["Z"] for r in R]), 2), Z_max=round(max(r["Z"] for r in R), 2),
                util_mean=round(st.mean(util), 4), util_p95=round(q(util, .95), 4), util_max=round(max(util), 4),
                util_gt_0p9=sum(u > 0.9 for u in util),
                invariant_violations=sum(r["invariant_viol"] for r in R),
                per_step_violations=sum(r["viol_step"] for r in R),
                max_step_overshoot=round(max(r["max_step_over"] for r in R), 6),
                gen_len_mean=round(st.mean([r["gen_len"] for r in R]), 1),
                early_eos_lt50=sum(r["gen_len"] < 50 for r in R),
                rougeL_mean=round(st.mean([r["rouge"] for r in R]), 4),
                jacc5_gt0=sum(r["mh"] > 0 for r in R))


def llr_row(k, K, split, R):
    L = [r["L"] for r in R]
    Lm = [r["Lmax"] for r in R if r["Lmax"] > -math.inf]
    Lr = [r["L_risky"] for r in R]
    return dict(k=k, K=K, split=split, n_traj=len(R),
                L_mean=round(st.mean(L), 2), L_sd=round(st.pstdev(L), 2), L_p50=round(q(L, .5), 2),
                L_p90=round(q(L, .9), 2), L_p95=round(q(L, .95), 2), L_p99=round(q(L, .99), 2), L_max=round(max(L), 2),
                n_L_gt_K=sum(l > K for l in L), frac_L_gt_K=round(sum(l > K for l in L) / len(L), 4),
                n_L_gt_half_K=sum(l > K / 2 for l in L),
                step_boost_mean=round(st.mean(Lm), 3), step_boost_p95=round(q(Lm, .95), 3), step_boost_max=round(max(Lm), 3),
                L_risky_mean=round(st.mean(Lr), 2), L_risky_max=round(max(Lr), 2))


def debt_row(k, K, split, R):
    d = [r["debt"] for r in R if r["debt"] is not None]
    lead = [r["lead_forced"] for r in R]
    pred = [math.floor(r["debt"] / k) if r["debt"] and r["debt"] > 0 else 0 for r in R]  # k_t=0 while (t+1)k <= delta
    return dict(k=k, K=K, split=split, n_traj=len(R),
                delta_init_mean=round(st.mean(d), 3), delta_init_max=round(max(d), 3),
                delta_gt_k=sum(x > k for x in d),
                lead_forced_mean=round(st.mean(lead), 3), lead_forced_max=max(lead),
                pred_floor_delta_over_k_mean=round(st.mean(pred), 3),
                lead_forced_eq_pred=sum(a == b for a, b in zip(lead, pred)))


def surprisal_row(k, K, split, R):
    S = [r["S_safe"] for r in R]
    cap = [min(1.0, (K + LN2) / s) if s > 0 else 1.0 for s in S]
    n = len(S)
    return dict(k=k, K=K, split=split, n_traj=n,
                S_safe_mean=round(st.mean(S), 1), S_safe_p10=round(q(S, .1), 1), S_safe_p50=round(q(S, .5), 1),
                S_safe_p90=round(q(S, .9), 1), S_star_mean=round(st.mean([r["S_star"] for r in R]), 1),
                cap_vacuous_n=sum(c >= 1.0 for c in cap), cap_vacuous_pct=round(100 * sum(c >= 1.0 for c in cap) / n, 2),
                cap_p50=round(q(cap, .5), 4), cap_mean=round(st.mean(cap), 4))


def seed_collision_rows(logs):
    rows = []
    for path in sorted(glob.glob(os.path.join(logs, "h2*_outputs", "heldout_validation.jsonl"))):
        for line in open(path):
            r = json.loads(line)
            sp = [round(x, 6) for x in r["spends"]]
            rows.append(dict(file=os.path.relpath(path, logs), candidate_id=r["candidate_id"], N=r["N"],
                             unique_spends=len(set(sp)), duplicated_spends=len(sp) - len(set(sp)),
                             candidate_valid=r.get("candidate_valid")))
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", default="output")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = {}  # (k, split) -> recs
    for path in sorted(glob.glob(os.path.join(args.logs, "h1_outputs", "trajectories_k*_*.jsonl"))):
        mm = FNAME.search(path)
        k, split = float(mm["k"]), mm["split"]
        data[(k, split)] = load(path)
        print(f"loaded {path}: {len(data[(k, split)])} trajectories")

    regime, llr, debt, surp, per_traj = [], [], [], [], []
    for (k, split), recs in sorted(data.items()):
        R = list(recs.values())
        K = R[0]["K"]
        larger = sorted(kk for (kk, ss) in data if ss == split and kk > k)
        ident = ""
        if larger:
            nxt = data[(larger[0], split)]
            shared = set(recs) & set(nxt)
            ident = round(100 * sum(recs[x]["gen"] == nxt[x]["gen"] for x in shared) / len(shared), 2)
        regime.append(regime_row(k, K, split, R, ident))
        llr.append(llr_row(k, K, split, R))
        debt.append(debt_row(k, K, split, R))
        surp.append(surprisal_row(k, K, split, R))
        for r in R:
            per_traj.append({kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in r.items() if kk != "gen"})

    write_csv(os.path.join(args.out, "regime_table.csv"), regime)
    write_csv(os.path.join(args.out, "llr_tails.csv"), llr)
    write_csv(os.path.join(args.out, "prefix_debt_forced_tokens.csv"), debt)
    write_csv(os.path.join(args.out, "surprisal.csv"), surp)
    write_csv(os.path.join(args.out, "seed_collisions.csv"), seed_collision_rows(args.logs))
    write_csv(os.path.join(args.out, "per_trajectory.csv"), per_traj)
    for row in regime:
        print(f"k={row['k']} {row['split']}: active={row['active_pct']}% identical_vs_next_k={row['identical_gen_vs_next_k_pct']} "
              f"util_max={row['util_max']} invariant_viol={row['invariant_violations']} step_viol={row['per_step_violations']}")
    for row in llr:
        print(f"k={row['k']} {row['split']}: L>K in {row['n_L_gt_K']}/{row['n_traj']} (max L={row['L_max']}, max step boost={row['step_boost_max']})")


if __name__ == "__main__":
    main()
