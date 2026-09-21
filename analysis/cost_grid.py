"""feat-162: the compute-matched cell, measured on the clock instead of proxied by parameter counts.

The manuscript concedes that "held to the metered decoder's own compute the mechanism loses: at
n=4 and 0.92x the cost it gains -0.0395". That 0.92x is analysis/compute_matched.py's cost_ratio,
which is n(P_anchor + P_scorer)/(P_anchor + P_risky) -- parameter counts. The paper's OWN
measurement (results/serving_latency.csv, caution (ae)) says the scorer is 9.3% of selection's wall
clock and the draws are the rest, so a cell selected by shrinking the scorer is selected on an axis
the measurement rejects. This measures the cells instead.

Bands and gates: results/onset_prediction_cost_matched_measured.md, committed before any timing
existed. This script computes them; it does not choose them.

Two modes, because one needs a GPU and the other must never touch one:

  --time-reward  (GPU) score the first n candidates of a generation directory and report the model
                 LOAD and the SCORING separately. A server loads once and then serves, so a loader
                 inside a serving cost is not a serving cost.
  --report       (CPU) parse the timing log, fit the loader out of the generation cells, apply the
                 gates in order and then the bands.

Usage:
  .venv/bin/python analysis/cost_grid.py --time-reward --gen-dir <dir> --n 64 \
      --model Qwen/Qwen2.5-7B-Instruct --rep 1
  .venv/bin/python analysis/cost_grid.py --report --out results
"""
import argparse
import csv
import os
import re
import statistics as st
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.serving_cost import P_ANCHOR, P_RISKY  # noqa: E402
from analysis.serving_latency import served_tokens  # noqa: E402

GRID = (1, 2, 4, 8, 16, 64)
PER_PROMPT = "results/compute_matched_per_prompt.csv"
CM_GRID = (2, 4, 8, 16, 32, 64)   # the grid compute_matched.py judged, in the order it judged it
CM_SEED = 9163                    # its --seed, and therefore its bootstrap stream
F4 = (-0.0395, -0.0720, -0.0065)  # the committed band this replay has to land on exactly
G_MET = 0.0400                    # metered_k10's committed gain, for the n=1 cell
WIDTHS = (8, 16, 32, 64, 128, 200)
B162 = 6.99                       # feat-162's b, seconds per completion at width 40
G0W_TOL = 0.25                    # G0: c_a(64) within this of B162
G2W_TOL = 0.10                    # G2: served tokens per request across widths
FALLS, RISES = 0.80, 1.25         # B1's three-way rule
SCORERS = {"Qwen/Qwen2.5-7B-Instruct": 7.6156, "Qwen/Qwen2.5-0.5B-Instruct": 0.4940}
PROXY_N4 = 0.92          # what the manuscript prints for sel05b_n4
WINDOW = (0.70, 1.45)    # B1's matched window
G0 = (15.0, 70.0)        # instrument band on ratio(64, 7.6B)
G1_MAX = 0.25            # reward share of selection's wall clock
G2_TOL = 0.05            # served-token agreement
G3_R2 = 0.98             # the linear model is allowed only this well fitted


def fit_line(xs, ys):
    """Least squares y = a + b x, with R^2. Plain arithmetic: the grid is twelve points."""
    n = len(xs)
    assert n >= 3 and len(ys) == n, "a two-point fit has no residual and no R^2"
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    assert sxx > 0, "every x is the same; there is no slope to fit"
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return a, b, r2


def paired_at(arm, per_prompt=PER_PROMPT):
    """The paired difference `arm - metered_k10`, from the utilities compute_matched.py committed.

    The registration said to recompute this by re-running compute_matched.py under its F5 gate.
    That is a GPU re-judge, and it is strictly worse than what the repository already holds: the
    per-prompt gains of every cell are committed, so the same judged utilities can be re-paired on
    a CPU with no second judging pass and therefore no cross-pass drift at all (caution (ap)).

    What replaces F5 is stronger. `paired_boot` draws from one `Random(seed)` stream that
    compute_matched.py advances once per arm and then once per band, so reproducing a committed
    band requires replaying that whole sequence -- and F4 then lands on its committed value to four
    decimals at both interval ends. The new cell is drawn next from the same stream.
    """
    import random
    from analysis.order_averaged_h2h import paired_boot
    rows = list(csv.DictReader(open(per_prompt, encoding="utf-8")))
    g = {k[len("gain_"):]: [float(r[k]) for r in rows] for k in rows[0] if k.startswith("gain_")}
    rng = random.Random(CM_SEED)
    for s_ in ("05b", "7b"):                      # the rows loop, in insertion order
        for n in CM_GRID:
            paired_boot(g[f"sel{s_}_n{n}"], rng)
    paired_boot(g["metered_k10"], rng)
    paired_boot([x - y for x, y in zip(g["sel05b_n64"], g["sel7b_n64"])], rng)   # F2
    d4 = [x - y for x, y in zip(g["sel05b_n4"], g["metered_k10"])]               # F4
    m4, (l4, h4) = sum(d4) / len(d4), paired_boot(d4, rng)
    replicates = (round(m4, 4), round(l4, 4), round(h4, 4)) == F4
    if arm not in g:            # n=1 selects the only draw there is: it IS the control, gain 0
        assert arm.endswith("_n1"), f"{arm} is not a judged cell and is not the n=1 control"
        return 0.0 - G_MET, None, None, replicates
    d = [x - y for x, y in zip(g[arm], g["metered_k10"])]
    m, (lo, hi) = sum(d) / len(d), paired_boot(d, rng)
    return m, lo, hi, replicates


def parse(log):
    """The launcher writes one line per timed cell. Nothing else in the log is read."""
    draws, met, rew = [], [], []
    for line in open(log, encoding="utf-8"):
        m = re.search(r"\[cost\] DRAWS rep=(\d+) n=(\d+) seconds=([\d.]+) dir=(\S+)", line)
        if m:
            draws.append((int(m.group(1)), int(m.group(2)), float(m.group(3)), m.group(4)))
        m = re.search(r"\[cost\] MET rep=(\d+) tpp=(\d+) seconds=([\d.]+) dir=(\S+)", line)
        if m:
            met.append((int(m.group(1)), int(m.group(2)), float(m.group(3)), m.group(4)))
        m = re.search(r"\[cost\] REWARD rep=(\d+) n=(\d+) model=(\S+) load_s=([\d.]+) "
                      r"score_s=([\d.]+)", line)
        if m:
            rew.append((int(m.group(1)), int(m.group(2)), m.group(3),
                        float(m.group(4)), float(m.group(5))))
    return draws, met, rew


def time_reward(gen_dir, n, model, dtype, batch_size):
    """Load the scorer, then score the first n candidates of every prompt. Times them apart."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from analysis.selection_decoding import load_candidates
    from analysis.selection_scaling import score_rewards

    cands = load_candidates(gen_dir)
    pids = sorted(p for p in cands if len(cands[p]) >= n)
    assert pids, f"{gen_dir} holds no prompt with {n} candidates"
    items = [(cands[p][j][2], cands[p][j][3]) for p in pids for j in range(n)]

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rm = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=getattr(torch, dtype)).cuda().eval()
    torch.cuda.synchronize()
    t1 = time.time()
    score_rewards(rm, tok, items, rm.device, batch_size=batch_size)
    torch.cuda.synchronize()
    t2 = time.time()
    del rm
    torch.cuda.empty_cache()
    return len(items), len(pids), t1 - t0, t2 - t1


def parse_width(log):
    """One line per timed cell. ANCHOR is the k=0 draw path, MET the k=10 metered decoder."""
    out = []
    for line in open(log, encoding="utf-8"):
        m = re.search(r"\[width\] (ANCHOR|MET) rep=(\d+) W=(\d+) tpp=(\d+) seconds=([\d.]+) "
                      r"dir=(\S+)", line)
        if m:
            out.append((m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)),
                        float(m.group(5)), m.group(6)))
    return out


def report_width(log, out):
    """feat-163. The per-batch cost of each path at each width, loader removed by differencing
    one against two completions -- the same construction feat-162 used, measured and not fitted."""
    cells = parse_width(log)
    assert cells, f"no width cells in {log}"
    have = sorted({c[2] for c in cells})
    assert have == sorted(WIDTHS), f"widths are {have}, registered {sorted(WIDTHS)}"

    def mean(path, w, tpp):
        v = [c[4] for c in cells if c[0] == path and c[2] == w and c[3] == tpp]
        assert v, f"no {path} cell at W={w} tpp={tpp}"
        return st.mean(v)

    c_a = {w: mean("ANCHOR", w, 2) - mean("ANCHOR", w, 1) for w in WIDTHS}
    c_m = {w: mean("MET", w, 2) - mean("MET", w, 1) for w in WIDTHS}
    ratio = {w: c_a[w] / c_m[w] for w in WIDTHS}

    # ---- gates -----------------------------------------------------------------------------
    gates = []
    g0 = abs(c_a[64] - B162) / B162
    gates.append((f"G0 instrument, c_a(64) within {G0W_TOL:.0%} of feat-162's {B162}s",
                  round(c_a[64], 3), "PASS" if g0 <= G0W_TOL else "FAIL"))
    bad = []
    per_req = {}
    for c in cells:
        tok, n_p = served_tokens(c[5])
        if n_p != c[2]:
            bad.append((c[0], c[2], c[3], n_p))
        per_req[(c[0], c[2], c[3])] = tok / n_p
    gates.append(("G1 one batch per seed group, served requests == W",
                  len(bad), "PASS" if not bad else f"FAIL {bad[:3]}"))
    lo, hi = min(per_req.values()), max(per_req.values())
    spread = (hi - lo) / lo
    gates.append((f"G2 same work served across widths, within {G2W_TOL:.0%}",
                  round(spread, 4), "PASS" if spread <= G2W_TOL else "FAIL"))
    g = {x[0].split()[0]: x[2] for x in gates}
    scored = g["G0"] == "PASS"

    rows = [dict(width=w, anchor_batch_s=round(c_a[w], 3), metered_batch_s=round(c_m[w], 3),
                 c_a_over_c_m=round(ratio[w], 4),
                 anchor_per_request_s=round(c_a[w] / w, 5),
                 metered_per_request_s=round(c_m[w] / w, 5),
                 implied_sel64_vs_metered=round(64 * ratio[w], 2)) for w in WIDTHS]

    bands = [dict(band=n, quantity="gate", value=v, reading=r) for n, v, r in gates]
    if not scored:
        bands.append(dict(band="B1 does c_a/c_m fall with width?", quantity="--", value="--",
                          reading="NOT SCORED -- G0 failed"))
    else:
        rel = ratio[200] / ratio[8]
        bands.append(dict(
            band="B1 does c_a/c_m fall with width?",
            quantity=f"ratio(200) {ratio[200]:.4f} over ratio(8) {ratio[8]:.4f}",
            value=round(rel, 4),
            reading="FALLS" if rel < FALLS else "RISES" if rel > RISES else "FLAT"))
        widest = max(WIDTHS)
        bands.append(dict(
            band="B2 implied selection price at n=64, properly batched",
            quantity=f"64 * c_a/c_m at W={widest}, against feat-162's 67.24x at W=40",
            value=round(64 * ratio[widest], 2),
            reading="A FLOOR ON OUR MEASUREMENT'S PESSIMISM, NOT A PRICE ANY ARM HERE PAYS"))
        bands.append(dict(
            band="B3 does batching lower the bill?",
            quantity="anchor seconds per request, W=8 over W=200",
            value=round((c_a[8] / 8) / (c_a[200] / 200), 2),
            reading="the number the naive objection is about, and the wrong one to look at"))

    os.makedirs(out, exist_ok=True)
    for path, data in ((os.path.join(out, "batch_width.csv"), rows),
                       (os.path.join(out, "batch_width_bands.csv"), bands)):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w_ = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
            w_.writeheader(); w_.writerows(data)
        print(f"wrote {path}")
    for r in rows:
        print(f"  W={r['width']:<4d} anchor {r['anchor_batch_s']:>7.3f}s  metered "
              f"{r['metered_batch_s']:>7.3f}s  c_a/c_m {r['c_a_over_c_m']:>7.4f}  "
              f"implied n=64 {r['implied_sel64_vs_metered']:>7.2f}x")
    for x in bands:
        print(f"  {x['band']:52s} {str(x['value']):>9s}  {x['reading']}")


def report(log, out, per_prompt=PER_PROMPT):
    draws, met, rew = parse(log)
    assert draws and met and rew, f"{log} is missing a whole class of cell"
    have = sorted({d[1] for d in draws})
    assert have == sorted(GRID), f"grid is {have}, registered {sorted(GRID)}"
    assert sorted({m[1] for m in met}) == [1, 2], "the metered path needs one and two completions"

    d_mean = {n: st.mean(d[2] for d in draws if d[1] == n) for n in GRID}
    m_mean = {t: st.mean(m[2] for m in met if m[1] == t) for t in (1, 2)}
    r_load = {k: st.mean(r[3] for r in rew if r[2] == k) for k in SCORERS}
    r_score = {(n, k): st.mean(r[4] for r in rew if r[1] == n and r[2] == k)
               for n in GRID for k in SCORERS}

    a, b, r2 = fit_line([d[1] for d in draws], [d[2] for d in draws])
    b_met = m_mean[2] - m_mean[1]
    a_met = m_mean[1] - b_met

    # ---- gates, in order -----------------------------------------------------------------------
    gates = []
    cost64_marg = 64 * b + r_score[(64, "Qwen/Qwen2.5-7B-Instruct")]
    ratio64 = cost64_marg / b_met
    gates.append(("G0 instrument, ratio(64, 7.6B) in [15, 70]", round(ratio64, 3),
                  "PASS" if G0[0] <= ratio64 <= G0[1] else "FAIL"))
    share = r_score[(64, "Qwen/Qwen2.5-7B-Instruct")] / cost64_marg
    gates.append((f"G1 premise, reward share of selection < {G1_MAX}", round(share, 4),
                  "PASS" if share < G1_MAX else "FAIL"))
    d64 = [d[3] for d in draws if d[1] == 64][0]
    m1 = [m[3] for m in met if m[1] == 1][0]
    t_sel, n_sel = served_tokens(d64)
    t_met, n_met = served_tokens(m1)
    rel = abs(t_sel / n_sel - t_met / n_met) / (t_met / n_met)
    gates.append((f"G2 same work served, within {G2_TOL}", round(rel, 4),
                  "PASS" if rel <= G2_TOL else "FAIL"))
    gates.append((f"G3 linear model allowed, R^2 >= {G3_R2} and b > 0", round(r2, 5),
                  "PASS" if r2 >= G3_R2 and b > 0 else "FAIL"))
    g = {x[0].split()[0]: x[2] for x in gates}
    scored = g["G0"] == "PASS" and g["G1"] == "PASS"
    marginal = g["G3"] == "PASS"

    # ---- the cell table ------------------------------------------------------------------------
    rows = []
    for k, pb in sorted(SCORERS.items(), key=lambda kv: kv[1]):
        for n in GRID:
            c_marg = n * b + r_score[(n, k)]
            c_raw = d_mean[n] + r_load[k] + r_score[(n, k)]
            rows.append(dict(
                scorer=k.split("/")[-1], scorer_b=pb, n=n,
                draws_s=round(d_mean[n], 3), reward_s=round(r_score[(n, k)], 3),
                cost_marginal_s=round(c_marg, 3), cost_raw_s=round(c_raw, 3),
                ratio_marginal=round(c_marg / b_met, 3), ratio_raw=round(c_raw / m_mean[1], 3),
                proxy_ratio=round(n * (P_ANCHOR + pb) / (P_ANCHOR + P_RISKY), 3)))
    key = "ratio_marginal" if marginal else "ratio_raw"

    # ---- bands ---------------------------------------------------------------------------------
    bands = [dict(band="G-fit: loader and slope", quantity="draws(n) = a + b n, metered a and b",
                  value=f"a={a:.2f}s b={b:.2f}s/completion; a_met={a_met:.2f}s b_met={b_met:.2f}s",
                  reading=f"R^2 {r2:.5f}")]
    for name, val, verdict in gates:
        bands.append(dict(band=name, quantity="gate", value=val, reading=verdict))

    if not scored:
        bands.append(dict(band="B1 matched cell", quantity="--", value="--",
                          reading="NOT SCORED -- a gate failed, and nothing below a failed gate "
                                  "is read"))
    else:
        inside = [r for r in rows if WINDOW[0] <= r[key] <= WINDOW[1]]
        if inside:
            pick = min(inside, key=lambda r: abs(r[key] - 1.0))
            b1 = f"MATCHED at {pick['scorer']} n={pick['n']}, {key}={pick[key]}"
        else:
            lo = max((r for r in rows if r[key] < 1.0), key=lambda r: r[key], default=None)
            hi = min((r for r in rows if r[key] > 1.0), key=lambda r: r[key], default=None)
            pick = None
            b1 = ("NO MATCHED CELL; brackets " +
                  " and ".join(f"{r['scorer']} n={r['n']} at {r[key]}" for r in (lo, hi) if r))
        bands.append(dict(band="B1 matched cell", quantity=f"argmin |{key} - 1| in {WINDOW}",
                          value=f"{pick['scorer']} n={pick['n']}" if pick else "", reading=b1))

        n4 = [r for r in rows if r["n"] == 4 and r["scorer_b"] == 0.494][0]
        ok = WINDOW[0] <= n4[key] <= WINDOW[1]
        bands.append(dict(
            band="B2 is the paper's cell compute-matched?",
            quantity=f"sel05b_n4 {key} vs the printed {PROXY_N4}x",
            value=n4[key],
            reading=("CONFIRMED" if ok else
                     f"MISPRICED: {n4[key]}x measured against {PROXY_N4}x printed, "
                     f"a factor of {n4[key] / PROXY_N4:.2f}")))
        if pick is None:
            bands.append(dict(band="B3 fate of the concession", quantity="paired difference at B1",
                              value="", reading="NOT SCORED -- B1 found no matched cell"))
        else:
            arm = f"sel{'05b' if pick['scorer_b'] == 0.494 else '7b'}_n{pick['n']}"
            m, lo, hi, replicates = paired_at(arm, per_prompt)
            if not replicates:
                read = ("NOT SCORED -- the replay does not reproduce the committed F4, so this "
                        "stream is not compute_matched.py's and no band drawn from it is quotable")
            elif lo is None:
                read = (f"CONCESSION STANDS -- the matched cell is the anchor's single draw, whose "
                        f"gain is 0 by construction, against the meter's {G_MET:+.4f}")
            elif lo > 0:
                read = "CONCESSION WITHDRAWN"
            elif hi < 0:
                read = "CONCESSION STANDS"
            else:
                read = "UNRESOLVED -- indistinguishable at matched measured compute"
            bands.append(dict(
                band="B3 fate of the concession",
                quantity=f"{arm} - metered_k10, paired, from the committed per-prompt utilities",
                value=round(m, 4),
                reading=(read if lo is None or not replicates
                         else f"{read} [{lo:+.4f}, {hi:+.4f}]")))

    os.makedirs(out, exist_ok=True)
    for path, data in ((os.path.join(out, "cost_grid.csv"), rows),
                       (os.path.join(out, "cost_grid_bands.csv"), bands)):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
        print(f"wrote {path}")
    for r in rows:
        print(f"  {r['scorer']:22s} n={r['n']:<3d} marginal {r['ratio_marginal']:>8.3f}x   "
              f"raw {r['ratio_raw']:>8.3f}x   proxy {r['proxy_ratio']:>8.3f}x")
    for x in bands:
        print(f"  {x['band']:46s} {str(x['value']):>10s}  {x['reading']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--time-reward", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--report-width", action="store_true")
    ap.add_argument("--gen-dir")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--log", default="output/logs/cost_grid.log")
    ap.add_argument("--width-log", default="output/logs/batch_width.log")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    assert sum((a.time_reward, a.report, a.report_width)) == 1, \
        "choose exactly one of --time-reward, --report and --report-width"
    if a.report_width:
        report_width(a.width_log, a.out)
    elif a.time_reward:
        assert a.model in SCORERS, f"{a.model} is not one of the registered scorers"
        n_items, n_p, load_s, score_s = time_reward(a.gen_dir, a.n, a.model, a.dtype, a.batch_size)
        print(f"[cost] REWARD rep={a.rep} n={a.n} model={a.model} "
              f"load_s={load_s:.3f} score_s={score_s:.3f} items={n_items} prompts={n_p}",
              flush=True)
    else:
        report(a.log, a.out)


if __name__ == "__main__":
    main()
