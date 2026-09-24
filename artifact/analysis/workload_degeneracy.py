"""How little the paper's own budget does, per workload -- as a committed artefact.

Three numbers have carried the vacuity argument across three workloads and NONE of them had a
producing script: the appendix's `794 of 805 served completions are identical to the opponent byte
for byte`, feat-170's `4.3%` and feat-173's `95.0%`. Each was computed once by hand into a scoring
log, which is caution (ai)'s shape (the claim about a set of numbers is checked against nothing)
crossed with caution (j)'s (a paper number must round from the CSV, once).

For each (workload, metered arm) this writes, from the run directories themselves:

  activity      steps_active / (active + forced_safe + risky_unchanged) AT THIS k, the fraction
                of decode steps at which the budget actually bound. A directory is not a budget:
                output/phase2/conc_all holds six of them and summing it reads 8.376% where the
                k=10 arm is 0.008%. The denominator is the three counters,
                NEVER aggregate.generation_length_tokens, which is the PADDED length (caution (ah)).
  byte_ident    the fraction of served completions EQUAL, byte for byte, to the unconstrained
                opponent's. This is the currency the other one cannot fake: a decoder that is
                producing the risky model's own text is not a metered decoder in any sense a
                reader would accept.

Both are over the prompts the judged pass actually shared, read off that pass's per-prompt file,
so the CSV and the head-to-head describe the same comparison.

Usage:
  .venv/bin/python analysis/workload_degeneracy.py --out results
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.order_averaged_h2h import lowest_seed  # noqa: E402
from analysis.selection_decoding import load_baseline  # noqa: E402
from analysis.utility import load_arm  # noqa: E402

# (workload, budget label, k, metered dir, opponent dir, per-prompt suffix)
ARMS = [
    # NOTE: output/mixpow/conc_all holds k=10 and output/mixpow/conc_k10 holds k=1. The directory
    # names are swapped with respect to their contents; the k is read off the trajectory files.
    # The committed metered arm, which is the one the paper's own 171.3 nats and 0.008% come from.
    ("ours (committed)", "vacuous", 10.0, "output/phase2/conc_all", "output/sweep_plain",  ""),
    ("ours",       "vacuous", 10.0, "output/wscope/a_conc",   "output/wscope/a_baseline", "__wscope_a"),
    ("ours",       "binding",  0.9, "output/wscope/c_conc",   "output/wscope/a_baseline", "__wscope_c"),
    ("AlpacaEval", "vacuous", 10.0, "output/mixpow/conc_all", "output/mixpow/baseline",   "__mixpow_judgeB"),
    # TWO AlpacaEval binding arms, and they are not interchangeable: feat-168's is the one whose
    # measured 8.008% every later workload rate-matched against, and feat-170 Arm B is its
    # disjoint re-draw. The appendix quotes the first; the second replicates it.
    ("AlpacaEval", "binding",  1.0, "output/mixpow/conc_k10", "output/mixpow/baseline",   "__mixpowk_judgeB"),
    ("AlpacaEval re-draw", "binding", 1.0, "output/wscope/b_conc", "output/wscope/b_baseline", "__wscope_b"),
    ("MT-Bench",   "vacuous", 10.0, "output/mtb/conc_k10",    "output/mtb/baseline",      "__mtb_conc_k10"),
    ("MT-Bench",   "binding",  1.0, "output/mtb/conc_bind",   "output/mtb/baseline",      "__mtb_conc_bind"),
    # feat-176: a public-domain COMPLETION corpus, the first workload of that kind we did not make.
    ("Gutenberg",  "vacuous", 10.0, "output/gutenberg/conc_k10",  "output/gutenberg/baseline", "__gutenberg_conc_k10"),
    ("Gutenberg",  "binding",  0.9, "output/gutenberg/conc_bind", "output/gutenberg/baseline", "__gutenberg_conc_bind"),
    # feat-174: reading comprehension (CoTaEval's NewsQA split), neither completion nor
    # instruction-following, and the arm the task-type axis was predicted against before it ran.
    ("CoTaEval-QA", "vacuous", 10.0, "output/cotaeval_qa/conc_k10",  "output/cotaeval_qa/baseline", "__cotaeval_qa_conc_k10"),
    ("CoTaEval-QA", "binding",  1.4, "output/cotaeval_qa/conc_bind", "output/cotaeval_qa/baseline", "__cotaeval_qa_conc_bind"),
    # feat-177: completion on books the anchor provably does not reproduce (G3) and BookMIA labels
    # unseen -- the arm that removes feat-176's familiarity confound.
    ("unseenbooks", "vacuous", 10.0, "output/unseenbooks/conc_k10",  "output/unseenbooks/baseline", "__unseenbooks_conc_k10"),
    ("unseenbooks", "binding",  1.0, "output/unseenbooks/conc_bind", "output/unseenbooks/baseline", "__unseenbooks_conc_bind"),
]


def activity(gen_dir, k, pids):
    """(active, total, n) over exactly the prompts the judged pass shared, at exactly this k.

    Two scopes have to match for the two columns of this table to describe one comparison, and
    neither is the directory's. A DIRECTORY IS NOT A BUDGET -- output/phase2/conc_all holds six of
    them, and summing it reads 8.376% where its k=10 arm is 0.008%, the 1046x error. A DIRECTORY
    IS NOT A PROMPT SET either: that arm's activity is 0.0080% over the three classes the judge
    sees and 0.0145% over all six, and quoting one for the other is caution (v)'s shape. So both
    are pinned to the judged intersection here.

    The denominator is the three step counters, never aggregate.generation_length_tokens, which
    is the PADDED length (caution (ah))."""
    act = tot = n = 0
    for f in sorted(glob.glob(os.path.join(gen_dir, f"trajectories_k{k:g}_*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["metadata"]["prompt_id"] not in pids:
                continue
            a = r["aggregate"]
            n += 1
            act += a.get("steps_active", 0)
            tot += (a.get("steps_active", 0) + a.get("steps_forced_safe", 0)
                    + a.get("steps_risky_unchanged", 0))
    assert n, f"{gen_dir}: no trajectory at k={k} over the {len(pids)} judged prompts"
    assert tot, f"{gen_dir}: no decode steps counted"
    return act, tot, n


INDEPENDENT_ACT = 0.001      # below this the decoder is the risky model at >99.9% of steps


def interpretable(act, ident):
    """Is this row's byte-identity a statement about the MECHANISM or about the SAMPLING?

    Only where the metered arm and the opponent are drawn from one seed stream. That is not
    recorded anywhere in the runs, but it is implied by the two columns already here: at an
    activity of 0.008% the decoder is the risky model at better than 99.99% of steps, so if the
    served text still differs it can only be a different SAMPLE of the same distribution. A low
    identity at a near-zero activity is therefore the signature of independent sampling, and the
    appendix says the same thing in words -- the committed protocol "cannot detect" the degeneracy
    because its two arms are sampled independently. Same workload, same k=10: the committed pass
    reads 2.8% and wscope's reads 99.5%.

    A reader of this CSV must not take that 2.8% for evidence that the budget did something."""
    if act < INDEPENDENT_ACT and ident < 0.5:
        return "no: arms sampled independently"
    return "yes"


def judged_pids(sfx):
    p = f"results/order_averaged_h2h_per_prompt{sfx}.csv"
    if not os.path.exists(p):
        return None
    return {r["prompt_id"] for r in csv.DictReader(open(p, encoding="utf-8"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    rows = []
    for workload, label, k, met_dir, opp_dir, sfx in ARMS:
        if not (os.path.isdir(met_dir) and os.path.isdir(opp_dir)):
            print(f"[skip] {workload}/{label}: {met_dir} or {opp_dir} is not here")
            continue
        pids = judged_pids(sfx)
        met = lowest_seed(load_arm(met_dir, k, "kl"))
        opp = load_baseline(opp_dir)
        shared = sorted(set(met) & set(opp) & (pids if pids is not None else set(met)))
        assert shared, f"{workload}/{label}: no prompt shared by the arm and the opponent"
        same = sum(1 for p in shared if met[p][1] == opp[p])

        act, tot, n = activity(met_dir, k, set(shared))
        rows.append(dict(workload=workload, budget=label, k=k,
                         steps_active=act, steps_total=tot, n_trajectories=n,
                         activity=round(act / tot, 8),
                         n_prompts=len(shared), n_byte_identical=same,
                         byte_ident=round(same / len(shared), 6),
                         byte_ident_interpretable=interpretable(act / tot, same / len(shared)),
                         pids_from=("judged pass" if pids is not None else "arm and opponent")))

    assert rows, "no workload directory was found; run this where the generations are"
    out = os.path.join(a.out, "workload_degeneracy.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{'workload':<18} {'budget':<8} {'k':>5} {'activity':>10} {'== opp':>10}  n")
    for r in rows:
        print(f"{r['workload']:<18} {r['budget']:<8} {r['k']:>5} {r['activity']:>9.5%} "
              f"{r['byte_ident']:>9.1%}  {r['n_byte_identical']:>4}/{r['n_prompts']:<5} "
              f"{r['byte_ident_interpretable']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
