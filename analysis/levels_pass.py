"""Order-averaged judged levels for arbitrary arms against the ONE committed opponent, de-echoed.

A generalisation of analysis/frontier_levels.py (same judge, same opponent, same construction:
every item in both presentation orders, the corpus prompt shown to the judge, all text through
dap.shared.served_generation) whose arms are named on the command line instead of hardcoded, so a
fresh draw or a new decoder can be judged without editing a script that feeds a figure.

Arm specs (`--arm name=spec`):
  sel:<gen_dir>:<reward_csv>:<n>[:<ktoken>]   argmax of the first n rewards (seed order) over the
                                               pool in gen_dir (files trajectories_k<ktoken>_*.jsonl,
                                               ktoken defaults to "0")
  pick:<gen_dir>:<csv>[:<ktoken>]              the draw at the rank the csv names per prompt
                                               (prompt_id,rank), e.g. an adaptive stopping rule
  traj:<run_dir>:<ktoken>[:<constraint>]       the lowest-seed trajectory of one arm (constraint
                                               defaults to "kl")

Judging a greedy judge in both orders is a deterministic function of the two texts, and each arm is
judged in its own batches in sorted-prompt order, so an arm's per-prompt level does not depend on
which other arms share the pass (feat-186 checked this on 500/500 prompts).

  judge:  levels_pass.py --arm a=... --arm b=... --tag T              -> results/levels_T{,_per_prompt}.csv
  score:  levels_pass.py --score T1,T2 --contrast D=+a-b+c-d ...       -> results/levels_<out-tag>_contrasts.csv
"""
import argparse
import csv
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.order_averaged_h2h import lowest_seed, paired_boot, true_prompts, u_of  # noqa: E402

OPPONENT = "output/sweep_plain"


def parse_contrast(expr):
    """'+a-b+c' -> [(+1,'a'),(-1,'b'),(+1,'c')]. A bare leading name counts as '+'."""
    expr = expr.replace(" ", "")
    if expr and expr[0] not in "+-":
        expr = "+" + expr
    terms = re.findall(r"([+-])([^+-]+)", expr)
    assert "".join(s + n for s, n in terms) == expr, f"unparseable contrast {expr!r}"
    return [(1 if s == "+" else -1, n) for s, n in terms]


def contrast_values(U, terms, pids):
    return [sum(c * U[(n, p)] for c, n in terms) for p in pids]


def load_texts(spec):
    """prompt_id -> de-echoed text for one arm spec, plus its first-draw empty indicator."""
    from analysis.selection_decoding import load_candidates
    from analysis.selection_scaling import load_rewards
    from analysis.utility import load_arm
    kind, *f = spec.split(":")
    if kind == "sel":
        gen_dir, rew, n = f[0], f[1], int(f[2])
        cands = load_candidates(gen_dir, k=(f[3] if len(f) > 3 else "0"), deecho=True)
        rewards = load_rewards(rew)
        out = {}
        for p, r in rewards.items():
            assert len(r) >= n and len(cands[p]) >= n, (p, len(r), len(cands[p]), n)
            r = r[:n]
            out[p] = cands[p][max(range(n), key=lambda i: r[i])][3]
        return out
    if kind == "pick":   # pick:<gen_dir>:<csv of prompt_id,rank>[:<ktoken>] -- a served rank per prompt
        gen_dir, path = f[0], f[1]
        cands = load_candidates(gen_dir, k=(f[2] if len(f) > 2 else "0"), deecho=True)
        return {r["prompt_id"]: cands[r["prompt_id"]][int(r["rank"])][3]
                for r in csv.DictReader(open(path, encoding="utf-8"))}
    if kind == "traj":
        run_dir, tok = f[0], f[1]
        cons = f[2] if len(f) > 2 else "kl"
        return {p: g for p, (_, g) in lowest_seed(load_arm(run_dir, tok, cons, deecho=True)).items()}
    raise SystemExit(f"unknown arm kind {kind!r} in {spec!r}")


def read_per_prompt(out, tags):
    U, pids = {}, None
    for t in tags:
        rows = list(csv.DictReader(open(os.path.join(out, f"levels_{t}_per_prompt.csv"))))
        ids = [r["prompt_id"] for r in rows]
        assert pids is None or ids == pids, f"tag {t} covers a different prompt list"
        pids = ids
        for r in rows:
            for c, v in r.items():
                if c != "prompt_id":
                    assert (c, r["prompt_id"]) not in U or U[(c, r["prompt_id"])] == float(v), \
                        f"arm {c} judged twice with different levels -- the pass is not deterministic"
                    U[(c, r["prompt_id"])] = float(v)
    return U, pids


def score(a):
    U, pids = read_per_prompt(a.out, a.score.split(","))
    rng = random.Random(a.seed)
    rows = []
    for spec in a.contrast:
        name, expr = spec.split("=", 1)
        terms = parse_contrast(expr)
        v = contrast_values(U, terms, pids)
        lo, hi = paired_boot(v, rng)
        m = sum(v) / len(v)
        reading = "ABOVE ZERO" if lo > 0 else "BELOW ZERO" if hi < 0 else "STRADDLES ZERO"
        rows.append(dict(contrast=name, expr=expr, value=round(m, 4), lo95=round(lo, 4),
                         hi95=round(hi, 4), n=len(pids), reading=reading))
        print(f"{name:28s} {expr:40s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  {reading}")
    path = os.path.join(a.out, f"levels_{a.out_tag}_contrasts.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", default=[], help="name=spec, repeatable")
    ap.add_argument("--tag", default="")
    ap.add_argument("--score", default="", help="comma list of judged tags to read")
    ap.add_argument("--contrast", action="append", default=[], help="NAME=+a-b..., repeatable")
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--judge", default="microsoft/Phi-3.5-mini-instruct")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=187)
    a = ap.parse_args()
    if a.score:
        return score(a)
    assert a.tag and a.arm, "--tag and at least one --arm are required to judge"

    from analysis.selection_decoding import load_baseline
    from analysis.utility import judge_batch
    arms = [s.split("=", 1) for s in a.arm]
    opp = load_baseline(OPPONENT, deecho=True)
    prompts = true_prompts("data")
    texts = {name: load_texts(spec) for name, spec in arms}
    pids = sorted(set(opp) & set(prompts) & set.intersection(*(set(t) for t in texts.values())))
    assert len(pids) == 500, f"{len(pids)} prompts shared, expected the headline's 500"
    print(f"[levels] {len(pids)} prompts, arms {[n for n, _ in arms]}", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.judge, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.judge, torch_dtype=torch.bfloat16).cuda().eval()

    rng = random.Random(a.seed)
    U, rows = {}, []
    for name, spec in arms:
        v = {}
        for order, items in (("first", [(prompts[p], texts[name][p], opp[p]) for p in pids]),
                             ("second", [(prompts[p], opp[p], texts[name][p]) for p in pids])):
            got = []
            for i in range(0, len(items), 200):
                got += judge_batch(model, tok, items[i:i + 200], "cuda")
            v[order] = got
        us = [0.5 * (u_of(f, True) + u_of(s, False)) for f, s in zip(v["first"], v["second"])]
        for p, u in zip(pids, us):
            U[(name, p)] = u
        lo, hi = paired_boot(us, rng)
        empty = sum(1 for p in pids if not texts[name][p].strip()) / len(pids)
        rows.append(dict(arm=name, spec=spec, level=round(sum(us) / len(us), 4), lo95=round(lo, 4),
                         hi95=round(hi, 4), empty_frac=round(empty, 4), n=len(pids)))
        print(f"[levels] {name:14s} {rows[-1]['level']:.4f} [{lo:.4f}, {hi:.4f}] "
              f"empty {empty:.3f}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"levels_{a.tag}_per_prompt.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["prompt_id"] + [n for n, _ in arms])
        for p in pids:
            w.writerow([p] + [U[(n, p)] for n, _ in arms])
    with open(os.path.join(a.out, f"levels_{a.tag}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
