"""The joint frontier over certificate, judged level and per-request latency (results/pareto_frontier_note.md).

Every input is already a committed CSV; this only sets them side by side and marks, mechanically,
which configurations another one beats on all three axes at once. No GPU.
  level    judge B, order-averaged, against the committed opponent, de-echoed text:
           results/frontier_levels.csv (released 8B pair) and the per-prompt files of the He-config
           pass (results/order_averaged_h2h_per_prompt_he70b_k{05,20}.csv, the authors' 70B pair)
  cert     selection: log n, pathwise (D_inf); meter: its KL budget K = k T, T = 200; anchor 0;
           a risky model alone has none (inf)
  latency  results/batched_latency.csv, per-request seconds at W = 1 and W = 8. The meter is timed
           at k = 10; the decoder forwards both models at every step whatever k is (caution (ay)),
           so every k of one pair is given that pair's measured cell. The anchor alone is a single
           draw without the reward pass (the SEL n = 1 cell's generation seconds).
Usage: .venv/bin/python analysis/pareto_frontier.py --out results
"""
import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

T, S_W = 200, 159.83   # decode length; a 50-token window's surprisal under the audited anchor


def rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def dominates(a, b):
    ge = (a["cert"] <= b["cert"], a["level"] >= b["level"], a["lat"] <= b["lat"])
    gt = (a["cert"] < b["cert"], a["level"] > b["level"], a["lat"] < b["lat"])
    return all(ge) and any(gt)


def per_prompt(res):
    """arm -> {prompt: order-averaged level}. Every arm is judged against the same opponent text by
    a greedy judge in both orders, so a per-prompt level is a function of the arm's text alone and
    arms from different passes pair up prompt by prompt."""
    u = {}
    for x in "abc":
        for r in rows(os.path.join(res, f"frontier_levels_per_prompt_{x}.csv")):
            for c, v in r.items():
                if c.startswith("u_"):
                    u.setdefault(c[2:], {})[r["prompt_id"]] = float(v)
    for f, cols in (("k05", {"u_metered_k0.5": "met70_k0.5", "u_risky70b": "risky70b"}),
                    ("k20", {"u_metered_k20": "met70_k20", "u_met70b_k1": "met70_k1"})):
        for r in rows(os.path.join(res, f"order_averaged_h2h_per_prompt_he70b_{f}.csv")):
            for c, arm in cols.items():
                u.setdefault(arm, {})[r["prompt_id"]] = float(r[c])
    u["anchor"] = u.pop("anchor_k0")
    return u


def margin(ua, ub, seed=190):
    """Paired level difference a - b over shared prompts, with a 95% bootstrap interval."""
    import random
    from analysis.selection_decoding import boot_mean
    d = [ua[p] - ub[p] for p in sorted(ua) if p in ub]
    lo, hi = boot_mean(d, random.Random(seed))
    return sum(d) / len(d), lo, hi


def build(res):
    u = per_prompt(res)
    lv = {a: sum(v.values()) / len(v) for a, v in u.items()}
    lat = {}
    for r in rows(os.path.join(res, "batched_latency.csv")):
        pair = "70b" if "70B" in r["risky"] else "8b" if r["risky"] != "-" else ""
        lat[(r["arm"], pair, int(r["W"]), int(r["n"]))] = (float(r["per_request_s"]),
                                                          float(r["gen_s"]) / int(r["W"]))
    out = []
    for W in (1, 8):
        def add(arm, pair, cert, kind, level, key, gen_only=False):
            if key not in lat:
                return
            out.append(dict(W=W, arm=arm, pair=pair, cert=cert, cert_kind=kind,
                            cert_over_Sw=round(cert / S_W, 4) if math.isfinite(cert) else "inf",
                            level=round(level, 4), lat=lat[key][1 if gen_only else 0]))
        for n in (1, 8, 64):
            add(f"sel_n{n}", "anchor only", math.log(n), "pathwise", lv[f"sel_n{n}"], ("SEL", "", W, n))
        add("anchor", "anchor only", 0.0, "none needed", lv["anchor"], ("SEL", "", W, 1), True)
        for k in ("0.5", "1", "3", "5", "10", "20"):
            add(f"met_k{k}", "8B", float(k) * T, "KL budget", lv[f"met_k{k}"], ("MET", "8b", W, 1))
        for k in ("0.5", "1", "20"):
            add(f"met70_k{k}", "70B", float(k) * T, "KL budget", lv[f"met70_k{k}"], ("MET", "70b", W, 1))
        add("risky8b", "8B", math.inf, "none", 0.5, ("RISKY", "8b", W, 1))
        add("risky70b", "70B", math.inf, "none", lv["risky70b"], ("RISKY", "70b", W, 1))
    for r in out:
        same = [o for o in out if o["W"] == r["W"] and o is not r]
        by = [o["arm"] for o in same if dominates(o, r)]
        r["dominated_by"] = " ".join(by)
        r["pareto"] = "YES" if not by else "no"
        # the level margin of the dominator whose paired interval sits highest; the risky 8B arm
        # IS the opponent, level 0.5 by symmetry and no per-prompt draw, so it gets no interval
        ms = [(margin(u[b], u[r["arm"]]), b) for b in by if b in u and r["arm"] in u]
        if ms:
            (m, lo, hi), b = max(ms, key=lambda x: (x[0][0], x[0][1]))   # largest margin
            r.update(best_dominator=b, level_margin=round(m, 4), margin_lo95=round(lo, 4),
                     margin_hi95=round(hi, 4))
        r["lat"] = round(r["lat"], 4)
        r["cert"] = round(r["cert"], 4) if math.isfinite(r["cert"]) else "inf"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = build(a.out)
    path = os.path.join(a.out, "pareto_frontier.csv")
    keys = []
    for r in out:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(r)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
