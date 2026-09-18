"""feat-113: does the headline reversal survive order-averaging on BOTH arms?

The paper compares selection anchoring's judged gain (+0.142 at n=64) against the metered decoder's
(+0.072 at k=10). Both were measured with every pair shown to the judge in ONE random order, which
makes each estimate unbiased but leaves the comparison resting on an instrument this paper's own
check calls UNUSABLE: order consistency 0.292, first-slot win rate 0.127, and one arm whose gain
fell from +0.081 to +0.0405 when position was removed by construction rather than in expectation
(results/judge_consistency.csv).

That check was never run on the two arms the abstract actually compares, and the DIFFERENCE of
their gains -- which is the claim -- has never been computed at all. This computes it.

Four arms, every one judged against the SAME fixed opponent (the unconstrained risky model at its
lowest seed, which is what selection_decoding.load_baseline returns and what utility.py judges the
metered arms against), and every item judged in BOTH presentation orders:

  sel_n64      selection at n=64, pointwise Qwen reward, picks replayed from the cached scores
  sel_n1       its control: the rank-0 draw, one anchor sample
  metered_k10  the metered decoder at k=10, output/phase2/conc_all
  anchor_k0    its control: the anchor served alone, output/sweep_plain

Bands are results/onset_prediction_order_averaged_h2h.md, committed before this ran. This script
computes them; it does not choose them.

No generation. A scoring pass over text already on disk.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/order_averaged_h2h.py --out results
"""
import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_decoding import boot_mean, load_baseline, load_candidates  # noqa: E402
from analysis.selection_scaling import load_rewards  # noqa: E402
from analysis.utility import judge_batch, load_arm  # noqa: E402
from dap.shared import load_prompt_corpus  # noqa: E402

N_BOOT = 10000
HEADER = "Complete the prefix:\n"


def true_prompts(data_dir):
    """prompt_id -> the prompt the models were actually given, header stripped.

    utility.py and selection_scaling.py both show the judge a prompt reconstructed as
    `full_text minus generation`, which is not the prompt: the split point depends on how much
    each arm generated, so for the same prompt_id the three arms here disagree on it in 455 of
    500 cases. Judging four arms under four different prompt strings would confound the
    comparison, so every arm is shown the one prompt from the corpus."""
    out = {}
    for r in load_prompt_corpus(data_dir, "text"):
        t = r.prompt_text
        out[r.prompt_id] = t[len(HEADER):] if t.startswith(HEADER) else t
    return out


def lowest_seed(arm):
    """(prompt_id, seed) -> (cls, prompt, gen) collapsed to prompt_id -> (prompt, gen) at the
    lowest seed, which is the convention load_baseline uses for the opponent."""
    out = {}
    for (pid, seed), (_, prompt, gen) in arm.items():
        if pid not in out or seed < out[pid][0]:
            out[pid] = (seed, prompt, gen)
    return {p: (v[1], v[2]) for p, v in out.items()}


def u_of(verdict, arm_is_first):
    if verdict == "Tie":
        return 0.5
    won = (verdict == "A") if arm_is_first else (verdict == "B")
    return 1.0 if won else 0.0


def paired_boot(diffs, rng, n_boot=N_BOOT):
    n = len(diffs)
    xs = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return xs[int(0.025 * n_boot)], xs[int(0.975 * n_boot)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sel-dir", default="output/phase5/sel_anchor64")
    ap.add_argument("--rewards", default="results/selection_rewards64.csv")
    ap.add_argument("--metered-dir", default="output/phase2/conc_all")
    ap.add_argument("--anchor-dir", default="output/sweep_plain")
    ap.add_argument("--baseline-dir", default="output/sweep_plain")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--judge", default="microsoft/Phi-3.5-mini-instruct")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--k", type=float, default=10.0)
    ap.add_argument("--metered-constraint", default="kl",
                    help="constraint tag of the --metered-dir arm; 'kl' is the deployed rule and "
                         "the default, so every number on record is unaffected. feat-126 judges a "
                         "'renyi:8' arm and feat-125 a sparse-causal 'kl' arm through the same path.")
    ap.add_argument("--seed", type=int, default=7717)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tag", default="",
                    help="suffix for the output CSVs. EMPTY writes the CANONICAL "
                         "results/order_averaged_h2h.csv, which holds the paper's headline -- pass "
                         "a tag for every exploratory arm so that file is never overwritten "
                         "(feat-123 had to restore it from a copy; feat-125/126/128 use tags).")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ---- assemble the four arms against one fixed opponent -----------------------------------
    opp = load_baseline(a.baseline_dir)
    cands = load_candidates(a.sel_dir)
    rewards = load_rewards(a.rewards)
    metered = lowest_seed(load_arm(a.metered_dir, a.k, a.metered_constraint))
    anchor = lowest_seed(load_arm(a.anchor_dir, 0.0, "kl"))
    prompts = true_prompts(a.data_dir)

    pids = sorted(set(opp) & set(cands) & set(rewards) & set(metered) & set(anchor) & set(prompts))
    assert pids, "no prompt is present in all four arms and the opponent"
    print(f"[h2h] {len(pids)} prompts shared by all four arms "
          f"(opp {len(opp)}, sel {len(cands)}, metered {len(metered)}, anchor {len(anchor)})",
          flush=True)

    texts = {}
    for p in pids:
        r = rewards[p][:a.n]
        best = max(range(len(r)), key=lambda i: r[i])
        texts[("sel_n%d" % a.n, p)] = cands[p][best][3]
        texts[("sel_n1", p)] = cands[p][0][3]
        texts[("metered_k%g" % a.k, p)] = metered[p][1]
        texts[("anchor_k0", p)] = anchor[p][1]

    arms = [f"sel_n{a.n}", "sel_n1", f"metered_k{a.k:g}", "anchor_k0"]

    # ---- judge every arm against the opponent, in both orders --------------------------------
    tok = AutoTokenizer.from_pretrained(a.judge, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.judge, torch_dtype=getattr(torch, a.dtype)).cuda().eval()

    U, CONS = {}, {}
    for arm in arms:
        fwd = [(prompts[p], texts[(arm, p)], opp[p]) for p in pids]   # arm first
        rev = [(prompts[p], opp[p], texts[(arm, p)]) for p in pids]   # arm second
        v = {}
        for tag, items in (("first", fwd), ("second", rev)):
            got = []
            for i in range(0, len(items), 200):
                got += judge_batch(model, tok, items[i:i + 200], "cuda")
                print(f"[h2h] {arm} arm-{tag} {len(got)}/{len(items)}", flush=True)
            v[tag] = got
        for p, x, y in zip(pids, v["first"], v["second"]):
            U[(arm, p)] = 0.5 * (u_of(x, True) + u_of(y, False))
            U[(arm, p, "single")] = u_of(x, True)       # the one-order estimate, for the contrast
            CONS[(arm, p)] = ((x == "A" and y == "B") or (x == "B" and y == "A")
                              or (x == "Tie" and y == "Tie"))

    # ---- the two gains, and the difference that is the paper's claim -------------------------
    sel, met = f"sel_n{a.n}", f"metered_k{a.k:g}"
    dS = [U[(sel, p)] - U[("sel_n1", p)] for p in pids]
    dM = [U[(met, p)] - U[("anchor_k0", p)] for p in pids]
    dD = [x - y for x, y in zip(dS, dM)]
    sS = [U[(sel, p, "single")] - U[("sel_n1", p, "single")] for p in pids]
    sM = [U[(met, p, "single")] - U[("anchor_k0", p, "single")] for p in pids]

    gS, gM, gD = (sum(z) / len(z) for z in (dS, dM, dD))
    loS, hiS = paired_boot(dS, rng)
    loM, hiM = paired_boot(dM, rng)
    loD, hiD = paired_boot(dD, rng)

    d1 = "SURVIVES" if not (loS <= 0 <= hiS) else "DISSOLVES"
    d2 = "SURVIVES" if not (loM <= 0 <= hiM) else "DISSOLVES"
    d3 = ("REVERSAL REFUTED" if gD <= 0 else
          "REVERSAL CONFIRMED" if not (loD <= 0 <= hiD) else "REVERSAL UNRESOLVED")

    cons = {arm: sum(CONS[(arm, p)] for p in pids) / len(pids) for arm in arms}

    rows = [dict(quantity="D1 selection gain, order-averaged", arm=sel, value=round(gS, 4),
                 lo95=round(loS, 4), hi95=round(hiS, 4), n=len(pids),
                 single_order=round(sum(sS) / len(sS), 4), reading=d1),
            dict(quantity="D2 metered gain, order-averaged", arm=met, value=round(gM, 4),
                 lo95=round(loM, 4), hi95=round(hiM, 4), n=len(pids),
                 single_order=round(sum(sM) / len(sM), 4), reading=d2),
            dict(quantity="D3 difference of gains, paired", arm=f"{sel} - {met}", value=round(gD, 4),
                 lo95=round(loD, 4), hi95=round(hiD, 4), n=len(pids),
                 single_order=round(sum(sS) / len(sS) - sum(sM) / len(sM), 4), reading=d3)]
    rows += [dict(quantity="order consistency", arm=arm, value=round(cons[arm], 4), lo95="",
                  hi95="", n=len(pids), single_order="",
                  reading="STABLE" if cons[arm] >= 0.70 else
                          "NOISY" if cons[arm] >= 0.50 else "UNUSABLE") for arm in arms]

    os.makedirs(a.out, exist_ok=True)
    suffix = f"_{a.tag}" if a.tag else ""
    with open(os.path.join(a.out, f"order_averaged_h2h{suffix}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(a.out, f"order_averaged_h2h_per_prompt{suffix}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["prompt_id"] + [f"u_{x}" for x in arms] + ["gain_sel", "gain_metered", "diff"])
        for p, x, y, z in zip(pids, dS, dM, dD):
            w.writerow([p] + [U[(arm, p)] for arm in arms] + [x, y, round(z, 4)])

    for r in rows:
        print(f"  {r['quantity']:38s} {r['arm']:22s} {r['value']:+.4f} "
              f"[{r['lo95']}, {r['hi95']}]  single-order {r['single_order']}  {r['reading']}")
    print(f"\n  D3 {d3}: selection {gS:+.4f} vs metered {gM:+.4f}, "
          f"difference {gD:+.4f} [{loD:+.4f}, {hiD:+.4f}] over {len(pids)} paired prompts")
    print(f"wrote {os.path.join(a.out, f'order_averaged_h2h{suffix}.csv')}")


if __name__ == "__main__":
    main()
