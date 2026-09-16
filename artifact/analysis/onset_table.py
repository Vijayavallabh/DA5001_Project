"""Plan v5 / feat-057: assemble the onset table the paper prints, from one command.

The numbers in Section 4's table live in three files -- onsets and bootstrap intervals in
results/onset_ci.csv, s(x) and s_r in results/onset_theory.csv, and the collapse spreads in
results/onset_collapse.csv -- and the cross-pair standard deviation is derived from all of them.
Deriving it by hand is how a table drifts from its evidence, so this writes exactly what the paper
quotes, including the choice that matters: when a pair has been measured twice, the higher-n
measurement is the one tabulated.

Since 2026-09-16 it also carries the memoriser's own sampled k=-1 recall, the column that makes
the table's confound visible: the nine memorisers are nine different fine-tunes and are not matched
in strength, and the seed ladders show the onset ratio tracks strength. Strength is taken from the
pair's OWN sweep wherever that sweep ran a k=-1 arm. Two of the nine did not -- output/phase4/fine_tc
and fine_comma report recall at several k with no baseline in their own summary, which is a standing
violation of this project's mandatory-baselines rule -- so for those two it is read from a named
companion run, and ONLY after asserting that the companion's `[ca]` protocol line is identical to
the sweep's. Caution (v): a reference number carries its protocol, and a borrow that is not checked
against the protocol line is a borrow from nowhere. `--strict` refuses the borrow outright.

Writes <out>/onset_table.csv. No GPU.

Usage: .venv/bin/python analysis/onset_table.py --out results
"""
import argparse, csv, json, os, re, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.score_predictions import canonical, load_measurements  # noqa: E402


# canonical pair -> (its own sweep, the run its baseline comes from, or None).
# Two sweeps never ran a k=-1 or k=0 arm, against Working Rules' mandatory-baselines requirement.
# scripts/run_phase4_baselines.sh measured both DELIBERATELY on 2026-09-16, at the defaults that
# reproduce each sweep's own protocol line, and reproduced the values previously borrowed from
# unrelated arms to three decimals (0.492 and 0.719). The second element is still only consulted
# when the sweep itself has no baseline, and still only after the protocol lines are asserted equal.
SWEEPS = {
    "tinycomma-1.8b + mem. llama-3.1-8b": ("output/phase4/fine_tc", "output/phase4/fine_tc_base"),
    "comma-7b + mem. comma-7b": ("output/phase4/fine_comma", "output/phase4/fine_comma_base"),
    "kl3m-520m + mem. kl3m-520m": ("output/phase5/fine_kl3m520m", None),
    "phi-3.5-mini + mem. phi-3.5-mini": ("output/phase5/fine_phi35", None),
    "pleias-1.2b + mem. pleias-1.2b": ("output/phase5/fine_pleias12b", None),
    "kl3m-1.7b + mem. kl3m-1.7b": ("output/phase5/fine_kl3m17b_full", None),
    "open-calm-1b + mem. open-calm-1b": ("output/phase5/fine_opencalm1b_full", None),
    "open-calm-3b + mem. open-calm-3b": ("output/phase5/fine_opencalm3b", None),
    "pleias-350m + mem. pleias-350m": ("output/phase5/n458_pleias350m", None),
}

MANIFEST = "results/onset_theory_pairs.tsv"


def memorisers():
    """canonical pair -> memoriser dir, from the manifest scripts/add_pair.sh writes.

    Hardcoding this mapping would be a second source of truth for something already committed, and
    the two would drift silently: the sweep and the memoriser are bound at registration time, not
    by a naming convention (output/memorizing_llama8b is not output/phase5/mem_tinycomma).
    """
    out = {}
    with open(MANIFEST) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0] and not parts[0].startswith("Ladder rung"):
                out[canonical(parts[0])] = parts[1]
    return out


def convergence(key):
    """Did the memorisation fine-tune reach its stop-loss, or run out of epochs?

    Five of the nine did not, which is a second variable the table does not control and one the
    seed ladders showed matters: the only cell whose ratio is not reproducible under re-seeding is
    the one whose fine-tune never converged. `epochs_run < epochs` is the readable form of the same
    fact and is what the table prints; the CSV keeps the losses so the equivalence is checkable.
    """
    d = memorisers().get(key)
    if not d or not os.path.exists(os.path.join(d, "recipe.json")):
        return {}
    r = json.load(open(os.path.join(d, "recipe.json")))
    ran, cap = int(r["epochs_run"]), int(r["epochs"])
    stop, final = float(r["stop_loss"]), float(r["final_loss"])
    return {"memoriser": d, "epochs_run": ran, "epochs_cap": cap,
            "stop_loss": stop, "final_loss": round(final, 4),
            "converged": "yes" if final <= stop else "no"}


# The informational insert composition_attack.py prints between the corpus and the seed (caution (b)):
# it reports how often the CopyBench `reference` field was reached and is NOT part of the protocol.
_INFO = re.compile(r"; reference reached in \d+/\d+")


def _protocol(path):
    """The one `[ca]` line that fingerprints a run: corpus size, target length, seed, warp."""
    src = path if path.endswith(".log") else path + ".log"
    if not os.path.exists(src):
        return None
    for line in open(src, errors="replace"):
        if line.startswith("[ca] ") and " passages;" in line:
            return _INFO.sub("", line.strip())
    return None


def _k_arm(path, k):
    """Sampled single-query recall at one budget, from a summary CSV or a run log."""
    if path.endswith(".log"):
        tag = "k=-1 single" if k < 0 else "k=0 single"
        for line in open(path, errors="replace"):
            if line.startswith(f"[ca] {tag}"):
                m = re.search(r"nv-recall mean ([0-9.]+)", line)
                if m:
                    return float(m.group(1)), 4          # a log prints 3 decimals
        return None, None
    summ = os.path.join(path, "composition_summary.csv")
    if not os.path.exists(summ):
        return None, None
    for r in csv.DictReader(open(summ)):
        if r["mode"] == "single" and float(r["k"]) == k:
            return float(r["nv_recall_mean"]), 6
    return None, None


def _n_of(path):
    summ = path if path.endswith(".csv") else os.path.join(path, "composition_summary.csv")
    if not os.path.exists(summ):
        return None
    for r in csv.DictReader(open(summ)):
        if r["mode"] == "single":
            return int(r["n_passages"])
    return None


def strength(key, strict=False, n_row=None):
    """(sampled k=-1, sampled k=0, provenance) for one pair, or (None, None, reason).

    `n_row` is the tabulated row's passage count, and it is checked rather than trusted. Pleias-350M
    is measured twice -- 100 passages and 458 -- and the table prints the 458 one, so its strength
    must come from the 458-passage sweep too (0.875) and not from the manifest's 100-passage sweep
    (0.906). A strength read off a different passage set than the onset beside it is the same class
    of error as quoting a Gutenberg number in a CopyBench claim.
    """
    if key not in SWEEPS:
        return None, None, "no sweep mapped"
    sweep, companion = SWEEPS[key]
    if n_row is not None and not sweep.endswith(".log"):
        got = _n_of(sweep)
        if got is not None and got != n_row:
            raise SystemExit(
                f"[table] {key}: sweep {sweep} covers {got} passages but the tabulated row is "
                f"n={n_row}. The strength and the onset must come from the same passages.")
    k1, _ = _k_arm(sweep, -1.0)
    if k1 is not None:
        k0, _ = _k_arm(sweep, 0.0)
        return k1, k0, "own sweep"
    if companion is None:
        return None, None, "sweep has no k=-1 arm and no companion is mapped"
    if strict:
        return None, None, "sweep has no k=-1 arm (--strict refuses the companion)"
    want, got = _protocol(sweep), _protocol(companion)
    if want is None or got is None or want != got:
        raise SystemExit(
            f"[table] {key}: refusing to borrow a baseline across protocols.\n"
            f"  sweep     {sweep}: {want}\n  companion {companion}: {got}")
    k1, _ = _k_arm(companion, -1.0)
    k0, _ = _k_arm(companion, 0.0)
    return k1, k0, f"baseline run {os.path.basename(companion).replace('.log', '')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--ci", default="results/onset_ci.csv")
    ap.add_argument("--theory", default="results/onset_theory.csv")
    ap.add_argument("--strict", action="store_true",
                    help="refuse a baseline borrowed from a companion run, even at an "
                         "identical protocol; those rows then read as not measured")
    a = ap.parse_args()

    meas = load_measurements(a.ci)
    rows = []
    for t in csv.DictReader(open(a.theory)):
        key = canonical(t["pair"])
        if key not in meas or t["pair"].startswith("Ladder rung"):
            continue
        m = meas[key]
        onset = float(m["onset_point"])
        s_s = float(t["s_safe_median"])
        rows.append({
            "pair": t["pair"], "n_passages": int(m["n_passages"]),
            "s_safe": round(s_s, 3), "s_risky": round(float(t["s_risky_median"]), 3),
            "onset": round(onset, 3),
            "ci_lo": round(float(m["onset_lo95"]), 3), "ci_hi": round(float(m["onset_hi95"]), 3),
            "ratio": round(onset / s_s, 4),
            "ratio_lo": round(float(m["onset_lo95"]) / s_s, 4),
            "ratio_hi": round(float(m["onset_hi95"]) / s_s, 4),
            "boot_no_crossing_pct": float(m["boot_no_crossing_pct"]),
            "k_grid": m["k_grid"],
        })
        k1, k0, src = strength(key, a.strict, n_row=int(m["n_passages"]))
        rows[-1]["k_minus1_sampled"] = "" if k1 is None else round(k1, 4)
        rows[-1]["k0_sampled"] = "" if k0 is None else round(k0, 4)
        rows[-1]["strength_source"] = src
        conv = convergence(key)
        for col in ("epochs_run", "epochs_cap", "stop_loss", "final_loss", "converged"):
            rows[-1][col] = conv.get(col, "")
    if not rows:
        raise SystemExit("[table] no pair has both a prediction and a measurement")
    rows.sort(key=lambda r: r["s_safe"])

    ratios = [r["ratio"] for r in rows]
    summary = {"pair": f"ALL {len(rows)} PAIRS", "n_passages": sum(r["n_passages"] for r in rows),
               "s_safe": round(min(r["s_safe"] for r in rows), 3),
               "s_risky": round(max(r["s_safe"] for r in rows), 3),
               "onset": round(min(r["onset"] for r in rows), 3),
               "ci_lo": round(max(r["onset"] for r in rows), 3), "ci_hi": "",
               "ratio": round(st.mean(ratios), 4),
               "ratio_lo": round(min(ratios), 4), "ratio_hi": round(max(ratios), 4),
               "boot_no_crossing_pct": round(st.pstdev(ratios), 4), "k_grid": "sd in the last column"}
    meas_k1 = [r["k_minus1_sampled"] for r in rows if r["k_minus1_sampled"] != ""]
    summary["k_minus1_sampled"] = round(min(meas_k1), 4) if meas_k1 else ""
    summary["k0_sampled"] = round(max(meas_k1), 4) if meas_k1 else ""
    conv_yes = sum(1 for r in rows if r["converged"] == "yes")
    summary["epochs_run"] = conv_yes
    summary["epochs_cap"] = len(rows)
    summary["stop_loss"] = ""
    summary["final_loss"] = ""
    summary["converged"] = f"{conv_yes} of {len(rows)} reached their stop-loss"
    summary["strength_source"] = (
        f"lo/hi of {len(meas_k1)} measured; factor {max(meas_k1) / min(meas_k1):.2f}"
        if meas_k1 else "none measured")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "onset_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows); w.writerow(summary)

    print(f"{'pair':38s}{'n':>5s}{'s(x)':>7s}{'s_r':>7s}{'onset':>8s}{'95% CI':>15s}"
          f"{'ratio':>8s}{'no-cross':>10s}")
    for r in rows:
        ci = f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]"
        print(f"{r['pair'][:37]:38s}{r['n_passages']:5d}{r['s_safe']:7.3f}{r['s_risky']:7.3f}"
              f"{r['onset']:8.3f}{ci:>15s}"
              f"{r['ratio']:8.3f}{r['boot_no_crossing_pct']:9.1f}%")
    print(f"\n{len(rows)} pairs: ratio mean {st.mean(ratios):.4f}, range "
          f"{min(ratios):.3f}-{max(ratios):.3f}, sd {st.pstdev(ratios):.4f}")
    print(f"s(x) spans {min(r['s_safe'] for r in rows):.3f}-{max(r['s_safe'] for r in rows):.3f} "
          f"nats/token ({max(r['s_safe'] for r in rows)/min(r['s_safe'] for r in rows):.2f}x)")
    print(f"\nwrote {a.out}/onset_table.csv")


if __name__ == "__main__":
    main()
