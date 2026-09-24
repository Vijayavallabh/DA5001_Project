"""The headline head-to-head under a selection rule that never serves an empty candidate (post hoc).

results/empty_as_loss.csv shows the registered construction leans on the judge scoring an empty answer
near parity: count every empty served text as a loss and the k=10 difference falls from +0.0505 to
+0.0155 [-0.0245, +0.0555]. The reward prefers empty drafts (Appendix B), so the remedy belongs in the
RULE, not the judge: serve the highest-scoring NON-empty draw. That is still a selection rule over the
same n draws, so Proposition 1 certifies it at log n unchanged.

Only the prompts whose committed pick is empty change pick, so only those are judged. Every GPU on both
hosts was held by another project when this ran, so the judge (judge B, Phi-3.5-mini) runs on CPU in
float32. Because the committed pass ran in bfloat16 on a GPU, the same prompts' COMMITTED picks are
re-judged here too, and the CPU verdicts are compared with the committed levels before any new level is
used; the other prompts keep their committed levels, which are a deterministic function of the text.

Writes results/nonempty_rule.csv. Descriptive and post hoc: no band was registered.

Usage: OMP_NUM_THREADS=32 .venv/bin/python analysis/nonempty_rule.py --out results
"""
import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.order_averaged_h2h import lowest_seed, paired_boot, true_prompts, u_of  # noqa: E402
from analysis.selection_decoding import load_baseline, load_candidates  # noqa: E402
from analysis.selection_scaling import load_rewards  # noqa: E402
from analysis.utility import judge_batch, load_arm  # noqa: E402


def empty(t):
    return not (t or "").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-prompt", default="results/order_averaged_h2h_per_prompt_deecho.csv")
    ap.add_argument("--judge", default="microsoft/Phi-3.5-mini-instruct")
    ap.add_argument("--seed", type=int, default=7717)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    opp = load_baseline("output/sweep_plain", deecho=True)
    cands = load_candidates("output/phase5/sel_anchor64", deecho=True)
    rewards = load_rewards("results/selection_rewards64.csv")
    anchor = lowest_seed(load_arm("output/sweep_plain", 0.0, "kl", deecho=True))
    metered = lowest_seed(load_arm("output/phase2/conc_all", 10.0, "kl", deecho=True))
    prompts = true_prompts("data")
    rows = list(csv.DictReader(open(a.per_prompt)))
    by = {r["prompt_id"]: r for r in rows}

    pick, alt = {}, {}
    for p in by:
        r = rewards[p][:64]
        order = sorted(range(len(r)), key=lambda i: (-r[i], i))   # ties by index, as max() breaks them
        pick[p] = order[0]
        alt[p] = next((i for i in order if not empty(cands[p][i][3])), order[0])
    changed = sorted(p for p in by if alt[p] != pick[p])
    assert all(empty(cands[p][pick[p]][3]) for p in changed)
    print(f"[nonempty] {len(changed)} of {len(by)} prompts change pick", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "32")))
    tok = AutoTokenizer.from_pretrained(a.judge, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.judge, torch_dtype=torch.float32).eval()

    def both_orders(texts):
        fwd = judge_batch(model, tok, [(prompts[p], texts[p], opp[p]) for p in changed], "cpu")
        rev = judge_batch(model, tok, [(prompts[p], opp[p], texts[p]) for p in changed], "cpu")
        return {p: 0.5 * (u_of(x, True) + u_of(y, False)) for p, x, y in zip(changed, fwd, rev)}

    committed_cpu = both_orders({p: cands[p][pick[p]][3] for p in changed})
    agree = sum(abs(committed_cpu[p] - float(by[p]["u_sel_n64"])) < 1e-9 for p in changed)
    print(f"[nonempty] CPU float32 reproduces the committed level on {agree}/{len(changed)}", flush=True)
    new = both_orders({p: cands[p][alt[p]][3] for p in changed})

    def lvl(arm, p, rule):
        if arm == "sel_nonempty":
            u, t = (new[p] if p in new else float(by[p]["u_sel_n64"])), cands[p][alt[p]][3]
        else:
            u = float(by[p][f"u_{arm}"])
            t = {"sel_n64": cands[p][pick[p]][3], "sel_n1": cands[p][0][3],
                 "metered_k10": metered[p][1], "anchor_k0": anchor[p][1]}[arm]
        if rule == "empty is a loss" and empty(t):
            return 0.5 if empty(opp[p]) else 0.0
        return u

    out = [dict(rule="control", quantity="committed picks re-judged on CPU, level reproduced",
                value=agree, lo95="", hi95="", n=len(changed))]
    pids = [r["prompt_id"] for r in rows]
    for rule in ("as judged", "empty is a loss"):
        rng = random.Random(a.seed)
        dS = [lvl("sel_nonempty", p, rule) - lvl("sel_n1", p, rule) for p in pids]
        dM = [lvl("metered_k10", p, rule) - lvl("anchor_k0", p, rule) for p in pids]
        dD = [x - y for x, y in zip(dS, dM)]
        dL = [lvl("sel_nonempty", p, rule) - lvl("metered_k10", p, rule) for p in pids]
        for name, xs in (("D1 non-empty selection gain", dS), ("D2 metered gain", dM),
                         ("D3 difference", dD), ("served levels, selection minus meter", dL)):
            lo, hi = paired_boot(xs, rng)
            out.append(dict(rule=rule, quantity=name, value=round(sum(xs) / len(xs), 4),
                            lo95=round(lo, 4), hi95=round(hi, 4), n=len(xs)))
    path = os.path.join(a.out, "nonempty_rule.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(r)
    print("wrote", path)


if __name__ == "__main__":
    main()
