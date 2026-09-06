"""Rebuild every paper figure from results/*.csv (feat-013). No GPU, no logs: the analysis scripts write the CSVs,
this script only plots them and copies the PDFs to the manuscript's figures directory.

  certificate_cap_curve  <- results/certificate_cap_summary.csv
  regime_sweep           <- results/regime_sweep.csv
  llr_tails              <- results/llr_tails.csv + results/llr_ratio_samples.csv
  composition            <- results/composition_summary.csv
  bank_burst             <- results/bank_burst_summary.csv (only if present)

Usage: .venv/bin/python figures/make_figures.py [--copy-to ~/sub/satml/figures]
"""
import argparse, csv, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import certificate_cap, composition_attack, llr_tails, regime_sweep  # noqa: E402

RESULTS = REPO / "results"
OUT = REPO / "figures"


def rows(name, numeric_except=("split", "variant", "mode", "source", "filler")):
    path = RESULTS / name
    if not path.exists():
        return None
    out = []
    for r in csv.DictReader(open(path)):
        row = {}
        for k, v in r.items():
            if k in numeric_except:
                row[k] = v
            elif v in ("", "None"):
                row[k] = None
            else:
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copy-to", default="", help="manuscript figures directory (empty: no copy)")
    args = ap.parse_args()
    made = []

    cert = rows("certificate_cap_summary.csv")
    if cert:
        certificate_cap.plot(cert, str(OUT))
        made.append("certificate_cap_curve.pdf")

    sweep = rows("regime_sweep.csv")
    if sweep:
        regime_sweep.plot(sweep, str(OUT / "regime_sweep"))
        made.append("regime_sweep.pdf")

    tails, samples = rows("llr_tails.csv"), rows("llr_ratio_samples.csv")
    if tails and samples:
        llr_tails.plot(tails, samples, str(OUT))
        made.append("llr_tails.pdf")

    comp = rows("composition_summary.csv")
    if comp:
        for r in comp:
            r["L"] = int(r["L"])
        ks = sorted({r["k"] for r in comp})
        modes = sorted({r["mode"] for r in comp}, key=["single", "oracle", "chained"].index)
        composition_attack.plot(comp, ks, modes, sorted({r["L"] for r in comp if r["L"]}), str(OUT))
        made.append("composition.pdf")

    bb = rows("bank_burst_summary.csv")
    if bb:
        from analysis import bank_burst  # noqa: F401  (its plot lives in main; regenerate via the script if needed)
        made.append("(bank_burst.pdf: regenerate with analysis/bank_burst.py)")

    for extra in ("natural_memorisation.pdf", "budget_path.pdf"):  # phase 2: built by analysis/natural_memorisation.py and analysis/budget_path.py
        if (OUT / extra).exists():
            made.append(extra)
    if args.copy_to:
        os.makedirs(args.copy_to, exist_ok=True)
        for f in made:
            if f.endswith(".pdf"):
                src = OUT / f
                (Path(args.copy_to) / f).write_bytes(src.read_bytes())
    print("built:", ", ".join(made), "->", OUT, "and", args.copy_to if args.copy_to else "(no copy)")


if __name__ == "__main__":
    main()
