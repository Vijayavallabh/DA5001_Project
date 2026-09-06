"""Aggregate the warped-anchor certificate caps (feat-022, plan v2 C11) produced by
analysis/certificate_cap.py --temperature T --repetition-penalty R --tag _tT_rpR into results/warped_anchor.csv:
per (temperature, repetition penalty, k): vacuity fraction, median cap, and the anchor's per-token surprisal of the
CopyBench passages. The certificate of He et al.'s Theorem 3.1 is relative to whatever anchor the decoder fuses with,
and the decoder warps both logit vectors before the solve (their App. B; our a_patch/factory.py), so these are the
quantities a deployer running at (0.7, 1.1) actually certifies.
Usage: .venv/bin/python analysis/warped_anchor.py --dir output/phase2/warped --out results
"""
import argparse, csv, glob, os, re, statistics as st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(os.path.join(args.dir, "certificate_cap_summary_t*_rp*.csv"))):
        m = re.search(r"_t([0-9.]+)_rp([0-9.]+)\.csv$", f)
        T, R = float(m.group(1)), float(m.group(2))
        caps = list(csv.DictReader(open(f.replace("certificate_cap_summary", "certificate_caps"))))
        s_tok = [float(r["S_safe_per_tok"]) for r in caps]
        s_tot = [float(r["S_safe"]) for r in caps]
        for r in csv.DictReader(open(f)):
            if r["split"] != "all":
                continue
            rows.append(dict(temperature=T, repetition_penalty=R, k=float(r["k"]), K=float(r["K"]), n=int(r["n"]),
                             vacuous_pct=float(r["vacuous_pct"]), cap_median=float(r["cap_median"]),
                             S_per_tok_median=round(st.median(s_tok), 4), S_total_median=round(st.median(s_tot), 1)))
    rows.sort(key=lambda r: (r["temperature"], r["repetition_penalty"], r["k"]))
    with open(os.path.join(args.out, "warped_anchor.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["k"] in (1.0, 3.0):
            print(f"[warp] T={r['temperature']} rp={r['repetition_penalty']} k={r['k']:g}: vacuous {r['vacuous_pct']}%, cap median {r['cap_median']}, S/token median {r['S_per_tok_median']}")
    print("wrote", os.path.join(args.out, "warped_anchor.csv"))


if __name__ == "__main__":
    main()
