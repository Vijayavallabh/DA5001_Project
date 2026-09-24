"""Every TokenSwap arm's gate quantities, recomputed from its own trajectories.

feat-165's committed gate table quoted `gamma = 0.3733` and a bind rate of `25.62%`. Both are real
numbers out of `output/logs/tokenswap_leakage.log` and neither is the quantity the table claims:
that launcher appends, so the log holds three runs, and those two values are from the 17:19 run --
the one whose pipeline did not match MemFree's and which feat-165 itself declared INVALID and
relaunched. The corrected 18:04 run, which produced every band the arm reports, reads `0.3745` and
`25.27%`. `0.3733` is additionally the rule-OFF arm's mass rather than the swap arm's.

Both gates pass either way and no band moves, which is exactly why nothing caught it: a gate whose
verdict is right can carry a number from a superseded run indefinitely. So the gate quantities get
the treatment every other paper number gets -- computed once into a CSV, and pinned there
(caution (j)) -- rather than being read off a log by a human (caution (ag)).

Writes results/tokenswap_gates.csv: one row per (directory, arm).
"""
import argparse
import csv
import json
import os
import pathlib

# The swap arm and the rule-off control. `-1` is what tokenswap_decode.py names the control.
ARMS = ("tokenswap", "-1")


def scan(gen_dir, arm):
    """(|G| in token ids, mean per-trajectory bind rate, mean mass on G, n) or None if absent."""
    fracs, masses, gsize = [], [], set()
    for path in sorted(pathlib.Path(gen_dir).glob(f"trajectories_k{arm}_*.jsonl")):
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            agg = r["aggregate"]
            # generation_length_tokens is the true step count here: the swap loop steps one token
            # at a time and never pads, so caution (ah) does not bite.
            fracs.append(agg["changed_steps"] / max(agg["generation_length_tokens"], 1))
            masses.append(agg["mass_on_G"])
            gsize.add(r["metadata"]["tokenswap_G_tokens"])
    if not fracs:
        return None
    assert len(gsize) == 1, f"{gen_dir}/{arm}: |G| is not constant: {sorted(gsize)}"
    return gsize.pop(), sum(fracs) / len(fracs), sum(masses) / len(masses), len(fracs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="output/tokenswap")
    ap.add_argument("--out", default="results/tokenswap_gates.csv")
    a = ap.parse_args()

    rows = []
    for d in sorted(pathlib.Path(a.root).iterdir()):
        if not d.is_dir():
            continue
        for arm in ARMS:
            got = scan(d, arm)
            if got is None:
                continue
            gtok, bind, mass, n = got
            rows.append({"arm_dir": d.name, "arm": arm, "g_token_ids": gtok,
                         "bind_rate": round(bind, 6), "mass_on_g": round(mass, 6), "n": n})
    assert rows, f"no tokenswap trajectories under {a.root}"

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out} ({len(rows)} rows)")
    for r in rows:
        print(f"  {r['arm_dir']:<18}{r['arm']:<11}|G|={r['g_token_ids']:>4}  "
              f"binds={r['bind_rate']:.4f}  mass={r['mass_on_g']:.4f}  n={r['n']}")


if __name__ == "__main__":
    main()
