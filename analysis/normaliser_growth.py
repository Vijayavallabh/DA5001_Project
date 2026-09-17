"""Does the normaliser ablation settle as pairs are added? Zero GPU.

Appendix D carries a growth series -- the window disagreement under s(x) against the requirement
r = s_safe - s_risky, at two pairs, three, four, and so on -- to show that the derivation's own
prediction about which rescaling collapses the curves best is not decisively checkable.

That series was maintained BY HAND across sessions, and nothing regenerated it. It went stale in
the predictable way: it stopped at seven pairs while the paper reports nine, and the sentence
around it claimed "the winner changes every time a pair is added", which the numbers printed
beside it already contradicted (found in the fourth read-through, 2026-09-17). The winner flips
twice, at three pairs and at four, and is the requirement at every count from four to nine.

This script makes the series reproducible: it runs analysis/collapse_robustness.py over nested
prefixes of the pair manifest and writes one row per pair count. Same code path as the committed
ablation, so the nine-pair row must equal collapse_robustness.csv's normaliser block.

Usage: .venv/bin/python analysis/normaliser_growth.py --out results
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile

MANIFEST = "results/onset_pairs.tsv"
S_KEY = "s_safe"
R_KEY = "requirement r = s_safe - s_risky"
RAW_KEY = "raw (no rescaling)"


def ablate(manifest, lines, n, workdir):
    """Run the committed ablation on the first n pairs and return its normaliser block."""
    man = os.path.join(workdir, f"pairs_{n}.tsv")
    with open(man, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines[:n]) + "\n")
    out = os.path.join(workdir, f"out_{n}")
    os.makedirs(out, exist_ok=True)
    r = subprocess.run([sys.executable, "analysis/collapse_robustness.py",
                        "--pairs-file", man, "--out", out],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"[growth] ablation failed at n={n}:\n{r.stderr[-800:]}")
    got = {x["setting"]: x for x in
           csv.DictReader(open(os.path.join(out, "collapse_robustness.csv"), encoding="utf-8"))
           if x["block"] == "normaliser"}
    if not got:
        raise SystemExit(f"[growth] n={n} produced no normaliser block; the ablation needs "
                         "results/onset_theory_per_work.csv to carry every pair in the manifest")

    def val(k):
        s = got[k]["value"]
        return float(s) if s not in ("", "nan") else float("nan")

    return val(S_KEY), val(R_KEY), val(RAW_KEY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--pairs-file", default=MANIFEST)
    a = ap.parse_args()

    lines = [l for l in open(a.pairs_file, encoding="utf-8").read().splitlines() if l.strip()]
    if len(lines) < 2:
        raise SystemExit(f"[growth] {a.pairs_file} has {len(lines)} pairs; need at least 2")

    rows, prev = [], None
    with tempfile.TemporaryDirectory(prefix="normgrowth_") as workdir:
        for n in range(2, len(lines) + 1):
            s, r, raw = ablate(a.pairs_file, lines, n, workdir)
            winner = "requirement" if r < s else "s_safe"
            rows.append({"n_pairs": n, "s_safe": round(s, 6), "requirement": round(r, 6),
                         "raw": round(raw, 6), "winner": winner,
                         "flipped": "" if prev is None else str(winner != prev).lower(),
                         "gap": round(s - r, 6)})
            prev = winner

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "normaliser_growth.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    flips = [r["n_pairs"] for r in rows if r["flipped"] == "true"]
    print(f"{'pairs':>5} {'s(x)':>9} {'r':>9} {'raw':>9}  winner")
    for r in rows:
        print(f"{r['n_pairs']:>5} {r['s_safe']:>9.4f} {r['requirement']:>9.4f} {r['raw']:>9.4f}"
              f"  {r['winner']}{'   <- flip' if r['flipped'] == 'true' else ''}")
    print(f"\nthe winner flips at {flips or 'no'} pair counts, out of {len(rows) - 1} transitions")
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
