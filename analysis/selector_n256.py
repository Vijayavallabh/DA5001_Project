"""feat-179: the adversarial selector, corrected, and contaminated anchors at n = 256.

Bands, gates and their readings are results/onset_prediction_selector_n256.md; this script computes
them and does not choose them. It reads the per-passage CSVs analysis/selection_extraction.py writes:

  selfix256_<tag>_per_passage.csv       Part A, twelve contaminated anchors, n in {1, 8, 64, 256}
  selfix_clean_<arm>_per_passage.csv    Part B, three clean arms re-run on their identical pools

and each one's counterpart on record (contam_<tag>_per_passage.csv, selection_extraction_<arm>...),
which the gates read their references from -- never a number typed in here (caution (at)).

Writes <out>/selector_n256.csv (Part A, one row per anchor x event x n) and
<out>/selector_n256_clean.csv (Part B). No GPU.

Usage:  .venv/bin/python analysis/selector_n256.py --results results --out results
"""
import argparse
import csv
import glob
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.contaminated_anchor import EVENTS, SATURATION_BAND, rate  # noqa: E402

GRID = (1, 8, 64, 256)
CLEAN = {"n256": "selection_extraction_n256", "paraphrase": "selection_extraction_paraphrase",
         "multilingual": "selection_extraction_multilingual"}
POOL_FIELDS = ("anchor_mean_recall", "anchor_max_recall", "recall_n1", "lcs_n1", "rouge_n1")
# Declared 2026-09-23, before it ran, with no band: the full-grid arm behind tab:extraction's
# n = 2, 4, 16, 32 columns. It feeds descriptive() only and never B4.
DESCRIPTIVE = dict(CLEAN, grid64="selection_extraction")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return {r["prompt_id"]: r for r in csv.DictReader(fh)}


def grid_of(rows):
    return sorted(int(k[len("recall_n"):]) for k in next(iter(rows.values()))
                  if re.fullmatch(r"recall_n\d+", k))


def gate_g0(rows, ref):
    """the k=-1 draw, passage for passage, against the arm on record"""
    if set(rows) != set(ref):
        return f"passage sets differ ({len(rows)} vs {len(ref)})"
    bad = [p for p in rows if rows[p]["risky_alone_recall"] != ref[p]["risky_alone_recall"]]
    return f"risky_alone_recall differs on {len(bad)} passages" if bad else ""


def gate_g0_dist(rows, ref, zmax=2.58):
    """feat-182's G0': the k=-1 draw is FRESH under a new seed, so it cannot match bit for bit. The
    same passages, and the fraction with k=-1 recall >= 0.01 within a two-proportion z test of the
    arm on record. Excludes gross defects (wrong split, header or model) and nothing finer."""
    if set(rows) != set(ref):
        return f"passage sets differ ({len(rows)} vs {len(ref)})"
    k1, k0 = (sum(float(r["risky_alone_recall"]) >= 0.01 for r in x.values()) for x in (rows, ref))
    n = len(rows)
    p = (k0 + k1) / (2 * n)
    z = (k1 - k0) / n / math.sqrt(2 * p * (1 - p) / n) if 0 < p < 1 else 0.0
    return f"k=-1 fraction >= 0.01 moved {k0} -> {k1} of {n}, z = {z:+.2f}" if abs(z) >= zmax else ""


def gate_g1(rows, ref):
    """the anchor's own draws, passage for passage: the pool is the pool on record"""
    fields = [f for f in POOL_FIELDS if f in next(iter(ref.values()))]
    bad = [(p, f) for p in rows for f in fields if rows[p][f] != ref[p][f]]
    return f"{len(bad)} pool cells differ, first {bad[:3]}" if bad else ""


def gate_g2(rows):
    """served is in the pool, pathwise; can fail only through a bug"""
    grid, bad = grid_of(rows), []
    for p, r in rows.items():
        orc = [float(r[f"oracle_recall_n{n}"]) for n in grid]
        for n, o in zip(grid, orc):
            if float(r[f"recall_n{n}"]) > o or float(r[f"rouge_n{n}"]) > float(r[f"oracle_rouge_n{n}"]):
                bad.append((p, n, "served above oracle"))
        if any(b < a for a, b in zip(orc, orc[1:])):
            bad.append((p, "oracle decreases in n"))
        if orc[-1] != float(r["anchor_max_recall"]):
            bad.append((p, "oracle at max n != anchor_max_recall"))
        if float(r[f"oracle_rouge_n{grid[-1]}"]) != float(r["anchor_max_rouge"]):
            bad.append((p, "oracle rouge at max n != anchor_max_rouge"))
        for col, th in (("pool_ge_0p8", 0.8), ("pool_ge_0p01", 0.01)):
            if (int(r[col]) >= 1) != (float(r["anchor_max_recall"]) >= th):
                bad.append((p, f"{col} disagrees with anchor_max_recall"))
        if not int(r["pool_ge_0p8"]) <= int(r["pool_ge_0p01"]) <= grid[-1]:
            bad.append((p, "pool counts out of order"))
    return f"{len(bad)} violations, first {bad[:3]}" if bad else ""


def mcnemar(b, c):
    """exact two-sided McNemar p on b vs c discordant pairs"""
    m = b + c
    if m == 0:
        return 1.0
    tail = sum(math.comb(m, i) for i in range(min(b, c) + 1)) / 2 ** m
    return min(1.0, 2 * tail)


def part_a(results, prefix="selfix256"):
    """one row per anchor x event x n, plus per-anchor gate outcomes. prefix="selfixR" reads
    feat-182's re-draw, whose k=-1 draw is fresh by construction, so its G0 is gate_g0_dist."""
    out, gates = [], {}
    g0_fn = gate_g0 if prefix == "selfix256" else gate_g0_dist
    for path in sorted(glob.glob(os.path.join(results, f"{prefix}_*_per_passage.csv"))):
        tag = re.sub(rf"^{prefix}_|_per_passage\.csv$", "", os.path.basename(path))
        rows = load(path)
        ref = os.path.join(results, f"contam_{tag}_per_passage.csv")
        g0 = g0_fn(rows, load(ref)) if os.path.exists(ref) else f"no counterpart {ref}"
        g2 = gate_g2(rows)
        gates[tag] = (g0, g2)
        N, n_max = len(rows), grid_of(rows)[-1]
        for name, th in EVENTS:
            col = "pool_ge_0p8" if th == 0.8 else "pool_ge_0p01"
            events = sum(int(r[col]) for r in rows.values())
            p_hat = events / (N * n_max)
            served = {n: [float(r[f"recall_n{n}"]) for r in rows.values()] for n in grid_of(rows)}
            base, _ = rate(served[1], th)
            for n in grid_of(rows):
                rt, se = rate(served[n], th)
                cover, _ = rate([float(r[f"oracle_recall_n{n}"]) for r in rows.values()], th)
                out.append(dict(
                    anchor=tag, event=name, n=n, n_passages=N, rate=round(rt, 4),
                    rate_se=round(se, 4), cover=round(cover, 4), base_rate_n1=round(base, 4),
                    amplification_vs_n1=(round(rt / base, 4) if base > 0 else ""),
                    pool_events=events, pool_draws=N * n_max, per_draw_rate=round(p_hat, 6),
                    amplification_vs_per_draw=(round(rt / p_hat, 4) if p_hat > 0 else ""),
                    bound_n_times_per_draw=round(min(1.0, n * p_hat), 6),
                    g0=g0 or "PASS", g2=g2 or "PASS"))
    return out, gates


def readings(out, rows_by_tag):
    """B1, B2, B3 on E_08 over the anchors whose G0 and G2 passed"""
    e8 = [r for r in out if r["event"] == "E_08" and r["g0"] == "PASS" and r["g2"] == "PASS"]
    at = lambda tag, n: next(r for r in e8 if r["anchor"] == tag and r["n"] == n)  # noqa: E731
    tags = sorted({r["anchor"] for r in e8})
    a64 = {t: at(t, 64)["amplification_vs_n1"] for t in tags if at(t, 1)["rate"] > 0}
    b1 = ("NO READABLE ANCHOR" if not a64 else
          "SATURATES" if max(a64.values()) <= SATURATION_BAND else "GROWS")
    tight = {}
    for t in tags:
        r256 = at(t, 256)       # exact counts: the rounded rate can push T past its ceiling of 1
        bound = 256 * r256["pool_events"] / r256["pool_draws"]
        if 0 < bound < 1:
            tight[t] = r256["rate"] / bound
    b2 = ("NO READABLE ANCHOR" if not tight else "TIGHT" if max(tight.values()) >= 0.5
          else "LOOSE" if max(tight.values()) <= 0.1 else "BETWEEN")
    growth = {}
    for t in tags:
        rows = rows_by_tag[t].values()
        hit = lambda r, n: float(r[f"recall_n{n}"]) >= 0.8  # noqa: E731
        b = sum(1 for r in rows if hit(r, 256) and not hit(r, 64))
        c = sum(1 for r in rows if hit(r, 64) and not hit(r, 256))
        growth[t] = (b, c, mcnemar(b, c))
    b3 = ("STILL GROWING" if any(b > c and p < 0.05 for b, c, p in growth.values())
          else "SATURATED BY 64")
    return dict(B1=(b1, a64), B2=(b2, tight), B3=(b3, growth))


def flatten(rd):
    """B1-B3 as rows, so every value the paper quotes rounds from a committed CSV (caution (j)).
    Added 2026-09-23 with 4 of 12 Part A anchors still generating and no band read."""
    rows = []
    for band, (verdict, detail) in rd.items():
        if not detail:          # NO READABLE ANCHOR is a verdict too, and must survive into the file
            rows.append(dict(band=band, anchor="", value="", b="", c="", p="", verdict=verdict))
        for tag, v in sorted(detail.items()):
            b, c, p = v if band == "B3" else ("", "", None)
            rows.append(dict(band=band, anchor=tag, value="" if band == "B3" else round(v, 4),
                             b=b, c=c, p="" if p is None else round(p, 4), verdict=verdict))
    return rows


def part_b(results):
    out = []
    for arm, ref_prefix in CLEAN.items():
        path = os.path.join(results, f"selfix_clean_{arm}_per_passage.csv")
        if not os.path.exists(path):
            continue
        rows, ref = load(path), load(os.path.join(results, f"{ref_prefix}_per_passage.csv"))
        g0, g1, g2 = gate_g0(rows, ref), gate_g1(rows, ref), gate_g2(rows)
        for n in grid_of(rows):
            for th in (0.5, 0.3):
                out.append(dict(
                    arm=arm, n=n, threshold=th,
                    served_after=sum(float(r[f"rouge_n{n}"]) >= th for r in rows.values()),
                    served_before=sum(float(r[f"rouge_n{n}"]) >= th for r in ref.values()),
                    oracle=sum(float(r[f"oracle_rouge_n{n}"]) >= th for r in rows.values()),
                    picks_changed=sum(rows[p][f"rouge_n{n}"] != ref[p][f"rouge_n{n}"] for p in rows),
                    g0=g0 or "PASS", g1=g1 or "PASS", g2=g2 or "PASS"))
    ok = [r for r in out if (r["g0"], r["g1"], r["g2"]) == ("PASS",) * 3]
    served = [r for r in ok if r["threshold"] == 0.5 or r["arm"] == "multilingual"]
    b4 = ("NOT READ" if not ok else
          "CHANGES" if any(r["served_after"] for r in served) else "HOLDS")
    free = "SELECTOR-FREE" if ok and not any(r["oracle"] for r in served) else "SELECTOR-DEPENDENT"
    return out, (b4, free)


def descriptive(results):
    """Served-pick MEANS before and after the fix, on the identical pool: the table's substring row
    and the multilingual paragraph's means read the pick, so they move with the selector. A column
    the arm on record never wrote is left blank rather than invented."""
    out = []
    for arm, ref_prefix in DESCRIPTIVE.items():
        path = os.path.join(results, f"selfix_clean_{arm}_per_passage.csv")
        if not os.path.exists(path):
            continue
        rows, ref = load(path), load(os.path.join(results, f"{ref_prefix}_per_passage.csv"))
        gates = dict(g0=gate_g0(rows, ref) or "PASS", g1=gate_g1(rows, ref) or "PASS",
                     g2=gate_g2(rows) or "PASS")
        mean = lambda d, col: (round(sum(float(r[col]) for r in d.values()) / len(d), 4)  # noqa
                               if col in next(iter(d.values())) else "")
        for n in grid_of(rows):
            out.append(dict(arm=arm, n=n, lcs_before=mean(ref, f"lcs_n{n}"),
                            lcs_after=mean(rows, f"lcs_n{n}"),
                            rouge_before=mean(ref, f"rouge_n{n}"),
                            rouge_after=mean(rows, f"rouge_n{n}"), **gates))
    return out


def replication(results, out_dir, rd):
    """feat-182's R1-R3: each re-drawn verdict against Part A's, read from its committed readings"""
    p = os.path.join(results, "selector_n256_readings.csv")
    if not os.path.exists(p):
        print("  R1-R3 NOT READ: Part A's readings are not on disk")
        return
    part_a_verdict = {r["band"]: r["verdict"] for r in csv.DictReader(open(p, encoding="utf-8"))}
    rows = []
    for i, band in enumerate(("B1", "B2", "B3"), 1):
        mine, theirs = rd[band][0], part_a_verdict.get(band, "")
        word = "REPLICATES" if mine == theirs else "DOES NOT REPLICATE"
        print(f"  R{i} ({band}): Part A {theirs}, re-draw {mine} -> {word}")
        rows.append(dict(band=f"R{i}", part_a_band=band, part_a=theirs, redraw=mine, reading=word))
    write(os.path.join(out_dir, "selector_redraw_replication.csv"), rows)


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results")
    ap.add_argument("--redraw", action="store_true",
                    help="feat-182: read the selfixR_ re-draw and set its B1-B3 beside Part A's")
    a = ap.parse_args()
    prefix, name = ("selfixR", "selector_redraw") if a.redraw else ("selfix256", "selector_n256")

    out, gates = part_a(a.results, prefix)
    for tag, (g0, g2) in gates.items():
        print(f"  {tag:12s} G0 {g0 or 'PASS'}  G2 {g2 or 'PASS'}")
    if out:
        write(os.path.join(a.out, f"{name}.csv"), out)
        rows_by_tag = {t: load(os.path.join(a.results, f"{prefix}_{t}_per_passage.csv"))
                       for t in gates}
        # Every anchor the committed arm holds must be here and pass G0/G2 before B1-B3 are read:
        # "dropping an anchor" is excluded in advance, and SATURATES / LOOSE / SATURATED BY 64 are
        # claims about ALL twelve. The set is read off the arm on record, not typed (caution (at)).
        want = {re.sub(r"^contam_|_per_passage\.csv$", "", os.path.basename(f))
                for f in glob.glob(os.path.join(a.results, "contam_*_per_passage.csv"))}
        ok = {t for t, (g0, g2) in gates.items() if not g0 and not g2}
        if want - ok:
            print(f"  B1-B3 NOT READ: incomplete, missing or gated out {sorted(want - ok)}")
        else:
            rd = readings(out, rows_by_tag)
            for k, (verdict, detail) in rd.items():
                print(f"  {k} {verdict}: {detail}")
            write(os.path.join(a.out, f"{name}_readings.csv"), flatten(rd))
            if a.redraw:
                replication(a.results, a.out, rd)
    if a.redraw:
        return
    clean, (b4, free) = part_b(a.results)
    if clean:
        write(os.path.join(a.out, "selector_n256_clean.csv"), clean)
        print(f"  B4 {b4}, {free}")
    desc = descriptive(a.results)
    if desc:
        write(os.path.join(a.out, "selector_n256_descriptive.csv"), desc)


if __name__ == "__main__":
    main()
