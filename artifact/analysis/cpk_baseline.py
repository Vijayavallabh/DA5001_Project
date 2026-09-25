"""feat-211, the utility half: CP-k's once-per-output rejection rule, run with the anchor as its safe model.

Vyas et al.'s CP-k rejection-samples complete outputs of the model it protects against safe models
trained on disjoint shards. No model pair here meets that premise, so this runs the same rule with the
one safe model the paper has: for each prompt the risky model's draws y_1..y_M are tested in seed
order, the first whose realised log-ratio
    R(y) = log p_r(y|x) - log p_s(y|x),
summed over the served tokens through the harness's own terminating token (served_steps, which reads
both probabilities off the per-step log), is at most kappa is served, and if none passes the anchor's own draw (the committed anchor arm) is served. Then
    q(y) <= M p_r(y) 1[R(y) <= kappa] + p_s(y) <= (M e^kappa + 1) p_s(y)
for every y, so D_inf(q || p_s) <= C = log(1 + M e^kappa) pathwise, whatever the acceptance rate.

Each certificate C on the grid becomes an arm directory of served records, judged by
analysis/matched_h2h.py exactly like every other arm (results/onset_prediction_cpk_baseline.md).

Writes <out-dir>/trajectories_kcpk<C>_<class>.jsonl, <results>/cpk_baseline.csv and
<results>/cpk_baseline_per_prompt.csv.
"""
import argparse
import csv
import glob
import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dap.shared import served_generation  # noqa: E402

CLASSES = ("neutral", "factual", "creative")
END_OF_TEXT = 128001
GRID = (4.1589, 33.271, 83.178, 159.83)


def kappa_for(c, m):
    """The acceptance threshold whose certificate log(1 + m e^kappa) is exactly c, in log space: e^c
    overflows a double past c = 709, and the leakage grid runs to 800 (the smoke caught it)."""
    return c + math.log(-math.expm1(-c)) - math.log(m)


def certificate(kappa, m):
    """log(1 + m e^kappa), as a stable softplus."""
    x = kappa + math.log(m)
    return x + math.log1p(math.exp(-x)) if x > 0 else math.log1p(math.exp(x))


def served_steps(rec):
    """log p_r - log p_s of every SERVED token, through the harness's own terminating token.

    A plain run (no chat template) stops only at <|end_of_text|>: dap/e1.py passes eos=tokenizer.eos_token_id,
    128001. An <|eot_id|> (128009) mid-generation is therefore not the end -- the model writes on and the served
    text keeps every token after it. analysis/window_logratio.trajectory_steps treats 128009 as an end, which
    cut R short on exactly the draws a small kappa then accepted: all three served at C = 33.27 had an eot_id
    at step 8 or 18 and ran on to 200 tokens (caught 2026-09-25 from the arm summary, before any verdict)."""
    assert not rec["metadata"].get("chat_template"), "a chat-template run stops at <|eot_id|> too"
    out = []
    for st in rec["per_step_log"]:
        ps, pr = st.get("p_s_prob"), st.get("p_risky_prob")
        if ps is None or pr is None or ps <= 0 or pr <= 0:
            raise ValueError(f"step without both probabilities: {st}")
        out.append(math.log(pr) - math.log(ps))
        if st.get("sampled_token_id") == END_OF_TEXT:
            break
    return out


def token(c):
    return f"cpk{c:g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--risky-dirs", nargs="+", required=True, help="h1.py k=-1 runs, any number of seeds")
    ap.add_argument("--anchor-dir", default="output/sweep_plain", help="the committed anchor arm, k=0")
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--grid", type=float, nargs="+", default=list(GRID))
    ap.add_argument("--out-dir", default="output/feat211/cpk_arms")
    ap.add_argument("--results", default="results")
    ap.add_argument("--prompts", type=int, default=500, help="G0: the prompt count every arm must cover")
    a = ap.parse_args()

    draws = {}                                   # prompt_id -> [(seed, cls, R, n_tokens, words, line)]
    for d in a.risky_dirs:
        for cls in CLASSES:
            for path in glob.glob(os.path.join(d, f"trajectories_k-1_{cls}.jsonl")):
                for line in open(path):
                    r = json.loads(line)
                    steps = served_steps(r)
                    gen = served_generation(r["aggregate"], r["prefix_analysis"]["prefix_text"])
                    draws.setdefault(r["metadata"]["prompt_id"], []).append(
                        (r["metadata"]["seed"], cls, sum(steps), len(steps), len(gen.split()), line))
    anchor = {}
    for cls in CLASSES:
        for line in open(os.path.join(a.anchor_dir, f"trajectories_k0_{cls}.jsonl")):
            r = json.loads(line)
            m = r["metadata"]
            if m.get("constraint", "kl") != "kl":
                continue
            if m["prompt_id"] not in anchor or m["seed"] < anchor[m["prompt_id"]][0]:
                anchor[m["prompt_id"]] = (m["seed"], cls, line)
    # G0: every prompt has exactly M draws at M distinct seeds, and an anchor draw to fall back on
    pids = sorted(draws)
    assert len(pids) == a.prompts, len(pids)
    for p in pids:
        seeds = [x[0] for x in draws[p]]
        assert len(seeds) == a.m == len(set(seeds)), (p, len(seeds))
        assert p in anchor, p
        draws[p].sort(key=lambda x: x[0])        # seed order is draw order

    os.makedirs(a.out_dir, exist_ok=True)
    rows, per = [], {p: dict(prompt_id=p, cls=draws[p][0][1], median_R=round(st.median(x[2] for x in draws[p]), 2),
                             min_R=round(min(x[2] for x in draws[p]), 2)) for p in pids}
    all_tok = [x[3] for p in pids for x in draws[p]]
    all_words = [x[4] for p in pids for x in draws[p]]
    for c in a.grid:
        k = kappa_for(c, a.m)
        files = {cls: open(os.path.join(a.out_dir, f"trajectories_k{token(c)}_{cls}.jsonl"), "w") for cls in CLASSES}
        acc_tok, acc_words, n_acc, pass_rate, empties = [], [], 0, [], 0
        for p in pids:
            j = next((i for i, x in enumerate(draws[p]) if x[2] <= k), None)
            pass_rate.append(sum(x[2] <= k for x in draws[p]) / a.m)
            if j is None:
                _, cls, line = anchor[p]
                per[p][f"served_C{c:g}"] = "anchor"
            else:
                seed, cls, R, n, w, line = draws[p][j]
                assert R <= k + 1e-9                       # G1: the served draw passed its own test
                n_acc += 1
                acc_tok.append(n)
                acc_words.append(w)
                empties += (w == 0)
                per[p][f"served_C{c:g}"] = f"risky#{j}"
            files[cls].write(line if line.endswith("\n") else line + "\n")
        for f in files.values():
            f.close()
        rows.append(dict(certificate_nats=c, kappa=round(k, 4), check_certificate=round(certificate(k, a.m), 4),
                         m=a.m, prompts=len(pids), served_risky_pct=round(100 * n_acc / len(pids), 1),
                         mean_pass_rate_per_draw=round(st.mean(pass_rate), 5),
                         accepted_median_tokens=(st.median(acc_tok) if acc_tok else ""),
                         accepted_median_words=(st.median(acc_words) if acc_words else ""),
                         all_draws_median_tokens=st.median(all_tok), all_draws_median_words=st.median(all_words),
                         accepted_empty=empties, token=token(c)))
    allR = [x[2] for p in pids for x in draws[p]]
    rows.append(dict(certificate_nats="median R of all risky draws, nats", kappa=round(st.median(allR), 2),
                     prompts=len(pids), m=a.m))
    with open(os.path.join(a.results, "cpk_baseline.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.results, "cpk_baseline_per_prompt.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per[pids[0]].keys()))
        w.writeheader()
        w.writerows(per[p] for p in pids)
    for r in rows:
        print(r, flush=True)


if __name__ == "__main__":
    main()
