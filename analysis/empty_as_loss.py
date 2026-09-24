"""The headline head-to-head with every empty served text scored as a loss (post hoc, no GPU).

Two referee reports (2026-09-24, sixth round) note that the judge scores an empty completion near parity
(0.45-0.49 against the risky model's full text, Appendix B), while the reward prefers empties and the
anchor controls are empty on 12.8% of prompts, and ask whether the headline survives a rule that counts
an empty answer as a loss. The judged verdicts are deterministic given the text, so this needs no
judge call: it re-reads the committed per-prompt levels of the repaired-text pass
(results/order_averaged_h2h_per_prompt_deecho.csv, the D3 = +0.0505 pass) and replaces the level of
every arm whose served text is empty with 0 (0.5 if the opponent's is empty too). The texts are
reassembled exactly as analysis/order_averaged_h2h.py assembled them, and the arm set is asserted to
match the CSV's prompts.

Usage: .venv/bin/python analysis/empty_as_loss.py --out results
"""
import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.order_averaged_h2h import lowest_seed, paired_boot  # noqa: E402
from analysis.selection_decoding import load_baseline, load_candidates  # noqa: E402
from analysis.selection_scaling import load_rewards  # noqa: E402
from analysis.utility import load_arm  # noqa: E402


def empty(t):
    return not (t or "").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-prompt", default="results/order_averaged_h2h_per_prompt_deecho.csv")
    ap.add_argument("--seed", type=int, default=7717)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    opp = load_baseline("output/sweep_plain", deecho=True)
    cands = load_candidates("output/phase5/sel_anchor64", deecho=True)
    rewards = load_rewards("results/selection_rewards64.csv")
    metered = lowest_seed(load_arm("output/phase2/conc_all", 10.0, "kl", deecho=True))
    anchor = lowest_seed(load_arm("output/sweep_plain", 0.0, "kl", deecho=True))
    rows = list(csv.DictReader(open(a.per_prompt)))
    pids = [r["prompt_id"] for r in rows]
    text = {}
    for p in pids:
        r = rewards[p][:64]
        best = max(range(len(r)), key=lambda i: r[i])
        text[("sel_n64", p)] = cands[p][best][3]
        text[("sel_n1", p)] = cands[p][0][3]
        text[("metered_k10", p)] = metered[p][1]
        text[("anchor_k0", p)] = anchor[p][1]
    arms = ["sel_n64", "sel_n1", "metered_k10", "anchor_k0"]

    def level(r, arm, rule):
        u = float(r[f"u_{arm}"])
        if rule == "empty is a loss" and empty(text[(arm, r["prompt_id"])]):
            return 0.5 if empty(opp[r["prompt_id"]]) else 0.0
        return u

    out = []
    for rule in ("as judged", "empty is a loss"):
        rng = random.Random(a.seed)
        dS = [level(r, "sel_n64", rule) - level(r, "sel_n1", rule) for r in rows]
        dM = [level(r, "metered_k10", rule) - level(r, "anchor_k0", rule) for r in rows]
        dD = [x - y for x, y in zip(dS, dM)]
        for name, xs in (("D1 selection gain", dS), ("D2 metered gain", dM), ("D3 difference", dD)):
            lo, hi = paired_boot(xs, rng)
            out.append(dict(rule=rule, quantity=name, value=round(sum(xs) / len(xs), 4),
                            lo95=round(lo, 4), hi95=round(hi, 4), n=len(xs)))
    as_judged = {r["quantity"]: r for r in out if r["rule"] == "as judged"}
    committed = next(r for r in csv.DictReader(open(os.path.join(a.out, "order_averaged_h2h_deecho.csv")))
                     if r["quantity"].startswith("D3"))
    assert abs(as_judged["D3 difference"]["value"] - float(committed["value"])) < 1e-4, (
        "the per-prompt CSV does not reproduce the committed D3; the texts or rows are not the pass's")
    for arm in arms + ["opponent"]:
        n = sum(empty(opp[p] if arm == "opponent" else text[(arm, p)]) for p in pids)
        out.append(dict(rule="empty served texts", quantity=arm, value=n, lo95="", hi95="", n=len(pids)))
    path = os.path.join(a.out, "empty_as_loss.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(r)
    print("wrote", path)


if __name__ == "__main__":
    main()
