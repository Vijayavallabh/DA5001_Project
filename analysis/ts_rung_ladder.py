"""feat-167 band B3: the judged utility ladder across TokenSwap auxiliaries.

Every rung is read against ONE shared rule-off control (`norule`), which the registration fixes as
an exclusion -- the control does not depend on the auxiliary, feat-165 generates it once, and
comparing rungs against different controls is excluded in advance.

Pairing across judging passes is only legitimate because the construction is order-averaged, which
under a greedy judge is a deterministic function of the text and so draws nothing (caution (ap)).
This script does not take that on trust: it asserts the anchor's per-prompt level is identical in
every pass before it pairs anything. A judged LEVEL is never compared across passes; what is paired
here is one arm against the control on the same prompt, which is the construction feat-165's H3
already used.
"""
import argparse
import csv
import json
import os
import pathlib
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.order_averaged_h2h import paired_boot  # noqa: E402

# (tag, auxiliary parameter count in millions, the arm's generation directory). TinyComma is
# feat-165's own rung, re-read here so the ladder is one series rather than three rungs beside a
# fourth quoted from another log.
RUNGS = [("ts_distilgpt2", 82, "output/tokenswap/util_distilgpt2"),
         ("ts_kl3m170m", 170, "output/tokenswap/util_kl3m170m"),
         ("ts_pleias350m", 350, "output/tokenswap/util_pleias350m"),
         ("tokenswap", 1800, "output/tokenswap/utility")]


def bind_rate(gen_dir, arm="tokenswap"):
    """|G| and the fraction of decode steps the rule changed, recomputed from the trajectories.

    The launcher's own log prints both, but a number scraped from a log is a comment rather than
    data (caution (ag)), and the shape claim this script makes is only legible beside them: a rung
    whose G barely survives its auxiliary's tokenizer is running a materially weaker rule, not a
    more permissive method. Definition matches tokenswap_decode.py exactly -- the per-trajectory
    ratio, meaned over trajectories. `generation_length_tokens` is the true step count here (the
    swap loop steps one token at a time and never pads), so caution (ah) does not bite.
    """
    fracs, gsize = [], set()
    for path in sorted(pathlib.Path(gen_dir).glob(f"trajectories_k{arm}_*.jsonl")):
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            agg = r["aggregate"]
            fracs.append(agg["changed_steps"] / max(agg["generation_length_tokens"], 1))
            gsize.add(r["metadata"]["tokenswap_G_tokens"])
    assert fracs, gen_dir
    assert len(gsize) == 1, f"{gen_dir}: G size is not constant across the arm: {sorted(gsize)}"
    return gsize.pop(), sum(fracs) / len(fracs), len(fracs)


def load(tag, out):
    path = os.path.join(out, f"order_averaged_h2h_per_prompt__{tag}.csv")
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[r["prompt_id"]] = r
    assert rows, path
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results")
    ap.add_argument("--control", default="norule")
    ap.add_argument("--seed", type=int, default=7717)
    a = ap.parse_args()

    ctrl = load(a.control, a.out)
    rng = random.Random(a.seed)
    rows = []

    for tag, params, gen_dir in RUNGS:
        cur = load(tag, a.out)
        pids = sorted(set(cur) & set(ctrl))
        assert len(pids) == len(ctrl) == len(cur), \
            f"{tag}: {len(cur)} prompts against control {len(ctrl)}, overlap {len(pids)}"

        # The pairing is legitimate only if the two passes judged the same anchor text to the same
        # level. Order-averaging makes that exact, so assert it exactly rather than to a tolerance.
        bad = [p for p in pids if cur[p]["u_anchor_k0"] != ctrl[p]["u_anchor_k0"]]
        assert not bad, f"{tag}: anchor level differs from the control pass on {len(bad)} prompts"

        col = "u_" + tag if tag != "tokenswap" else "u_tokenswap"
        gain = sum(float(cur[p][col]) - float(cur[p]["u_anchor_k0"]) for p in pids) / len(pids)
        diffs = [float(cur[p][col]) - float(ctrl[p]["u_" + a.control]) for p in pids]
        cost = sum(diffs) / len(diffs)
        lo, hi = paired_boot(diffs, rng)
        gtok, bind, ntraj = bind_rate(gen_dir)
        rows.append((tag, params, gain, cost, lo, hi, gtok, bind, ntraj))

    ctrl_gain = sum(float(ctrl[p]["u_" + a.control]) - float(ctrl[p]["u_anchor_k0"])
                    for p in ctrl) / len(ctrl)
    print(f"shared rule-off control ({a.control}) gain over the anchor: {ctrl_gain:+.4f}  "
          f"n={len(ctrl)}")
    print(f"{'rung':<16}{'params/M':>9}{'|G| ids':>9}{'binds':>8}"
          f"{'gain':>10}{'cost vs control':>18}{'95% CI':>24}")
    for tag, params, gain, cost, lo, hi, gtok, bind, ntraj in rows:
        print(f"{tag:<16}{params:>9}{gtok:>9}{bind:>8.4f}{gain:>+10.4f}{cost:>+18.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>24}")

    gains = [r[2] for r in rows]
    up = all(b > a_ for a_, b in zip(gains, gains[1:]))
    down = all(b < a_ for a_, b in zip(gains, gains[1:]))
    shape = "MONOTONE RISING" if up else "MONOTONE FALLING" if down else "NOT MONOTONE"
    peak = max(rows, key=lambda r: r[2])
    print(f"\nshape in auxiliary size, ALL rungs: {shape}; "
          f"peak at {peak[0]} ({peak[1]}M, {peak[2]:+.4f})")

    # A rung whose rule barely binds is not a point on a utility-vs-size ladder, it is a different
    # rule. Report the shape over the rungs that bind comparably, and say which rung that excludes
    # and why -- the exclusion is mechanical (a bind rate an order of magnitude off the others),
    # not a judgement about whether the number flatters anyone.
    binds = [r[7] for r in rows]
    med = sorted(binds)[len(binds) // 2]
    comparable = [r for r in rows if med / 4 <= r[7] <= med * 4]
    dropped = [r for r in rows if r not in comparable]
    g2 = [r[2] for r in comparable]
    up2 = all(b > a_ for a_, b in zip(g2, g2[1:]))
    down2 = all(b < a_ for a_, b in zip(g2, g2[1:]))
    shape2 = "MONOTONE RISING" if up2 else "MONOTONE FALLING" if down2 else "NOT MONOTONE"
    print(f"shape over the {len(comparable)} rungs that bind within 4x of the median "
          f"({med:.4f}): {shape2}")
    for r in dropped:
        print(f"  off the ladder: {r[0]} binds {r[7]:.4f} on {r[6]} of "
              f"{max(x[6] for x in rows)} G ids -- a weaker rule, not a cheaper one")


if __name__ == "__main__":
    main()
