"""feat-115: the certificate's premise is checkable, and here is what checking it costs.

Proposition 4 bounds Pr_q[E] <= n Pr_{p_s}[E]. Everything therefore rests on the safe model's own
Pr_{p_s}[E] being small, which the paper has so far ASSUMED for its anchors and only measured on
anchors we deliberately contaminated. A reviewer put the objection sharply: open-licensed corpora
carry only partial provenance guarantees, so "self-consistency already carries a certificate" is
true relative to a hypothetical clean anchor and unknown for a real one, and the Ethics Statement
concedes the contaminated case while the introduction celebrates the clean one.

The answer is that the premise is not a matter of belief. Pr_{p_s}[E] is the thing a deployer can
measure directly, with no attack, no access to the mechanism and no risky model: draw from the
candidate anchor on the protected passages' own prefixes and score near-verbatim recall. That is
the same quantity the certificate multiplies by n.

This script assembles that measurement across every anchor in the paper and asks whether it
separates the models known to have seen the work from the models believed not to have.

  clean         the openly licensed anchors the paper actually uses
  contaminated  twelve anchors LoRA-fine-tuned on these passages (feat-111/112)
  natural       Llama-3.1-70B, which memorised Harry Potter in pre-training rather than because
                we made it -- the only case here where the memorisation is not ours

IMPORTANT, and stated in the manuscript too: this is a POST HOC re-analysis of arms run for other
purposes. No band was committed in advance, because there is no new run to commit one against, and
it must not be reported as if it had been. The separation on the contaminated arms is also partly
circular -- those models were fine-tuned on exactly these passages, so their leaking is close to a
tautology. The two non-circular facts are that the five clean anchors read exactly zero on every
passage, and that the one model which memorised in pre-training does not.

Writes <out>/anchor_vetting.csv. No GPU: this reads per-passage CSVs already on disk.

Usage:
  .venv/bin/python analysis/anchor_vetting.py --out results
"""
import argparse
import csv
import glob
import os


def frac_and_max(rows, field):
    v = [float(r[field]) for r in rows]
    return sum(1 for x in v if x > 0) / len(v), max(v), sum(v) / len(v), len(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    out = []
    # clean anchors: the anchor's own draw is recall_n1 in the selection-extraction arms
    for p in sorted(glob.glob(os.path.join(a.results, "selection_extraction_*_per_passage.csv"))):
        tag = os.path.basename(p).replace("selection_extraction_", "").replace("_per_passage.csv", "")
        if "70b" in tag:
            continue        # those arms vary the RISKY model; the anchor is the audited one
        rows = list(csv.DictReader(open(p)))
        f, mx, mean, n = frac_and_max(rows, "recall_n1")
        out.append(dict(model=tag, role="anchor", provenance="openly licensed",
                        n_passages=n, frac_passages_leaking=round(f, 4),
                        max_recall=round(mx, 4), mean_recall=round(mean, 4),
                        verdict="PASSES" if f == 0.0 else "FAILS"))

    # anchors we contaminated on purpose
    for p in sorted(glob.glob(os.path.join(a.results, "contam_*_per_passage.csv"))):
        tag = os.path.basename(p).replace("contam_", "").replace("_per_passage.csv", "")
        rows = list(csv.DictReader(open(p)))
        f, mx, mean, n = frac_and_max(rows, "recall_n1")
        out.append(dict(model=tag, role="anchor", provenance="fine-tuned on these passages",
                        n_passages=n, frac_passages_leaking=round(f, 4),
                        max_recall=round(mx, 4), mean_recall=round(mean, 4),
                        verdict="PASSES" if f == 0.0 else "FAILS"))

    # the one model that memorised without our help; it plays the risky role, but the measurement
    # is a property of the model, not of the role it is cast in
    nat = os.path.join(a.results, "selection_extraction_70b_hp2_per_passage.csv")
    if os.path.exists(nat):
        rows = list(csv.DictReader(open(nat)))
        f, mx, mean, n = frac_and_max(rows, "risky_alone_recall")
        out.append(dict(model="Llama-3.1-70B", role="(measured as a model)",
                        provenance="memorised in pre-training", n_passages=n,
                        frac_passages_leaking=round(f, 4), max_recall=round(mx, 4),
                        mean_recall=round(mean, 4), verdict="PASSES" if f == 0.0 else "FAILS"))

    path = os.path.join(a.out, "anchor_vetting.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

    clean = [r for r in out if r["provenance"] == "openly licensed"]
    dirty = [r for r in out if r["provenance"] != "openly licensed"]
    for r in out:
        print(f"  {r['model']:16s} {r['provenance']:28s} leaking on "
              f"{r['frac_passages_leaking']:.3f} of {r['n_passages']:3d}  max {r['max_recall']:.4f}"
              f"  {r['verdict']}")
    hi_clean = max(r["frac_passages_leaking"] for r in clean)
    lo_dirty = min(r["frac_passages_leaking"] for r in dirty)
    print(f"\n  clean anchors leak on at most {hi_clean:.4f} of passages; every model known to have"
          f" seen the work leaks on at least {lo_dirty:.4f}")
    print(f"  SEPARATES: {hi_clean < lo_dirty}  ({len(clean)} clean, {len(dirty)} known-contaminated)")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
