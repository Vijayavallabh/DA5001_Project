"""feat-064: the onset ratio as a function of how much of the work the adversary already holds.

`analysis/composition_attack.py` seeds the adversary with a fixed number of *tokens*, so the number
of *words* it buys depends on the tokenizer: 7.3 on the KL3M pairs against 13.0-14.4 on the other
five, an exact split with no overlap, in the same place the onset ratio splits. This script
assembles the runs that vary the seed with everything else held fixed, so the split can be read as
a function of the adversary's context rather than of the tokenizer.

Each row of the manifest is one run:

    label <TAB> seed_tokens <TAB> composition.csv <TAB> budget_path.csv <TAB> tokenizer <TAB> pair

`pair` groups runs that differ only in the seed. Seed length in words is measured on the same
passages the run used, not assumed from the tokenizer's average.

Primary metric `lcs_word >= 4`, an absolute word count that a changed reference length cannot
inflate; `nv_recall >= 0.01` reported alongside. Bootstrap over passages.

Writes <out>/seed_effect.csv. Needs the tokenizers, no GPU.

Usage:
  HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache .venv/bin/python analysis/seed_effect.py --out results
"""
import argparse, csv, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.score_truncation import per_passage, point_and_ci  # noqa: E402


def seed_words(tokenizer_id, seed_tokens, limit=100):
    """Characters and words the seed actually buys, on the passages the attack used."""
    from transformers import AutoTokenizer
    from dap.shared import load_prompt_corpus
    from analysis.composition_attack import join
    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    ps = [p for p in load_prompt_corpus("data", "factscore_prompt")
          if p.split == "test" and p.reference][:limit]
    ch, wd = [], []
    for p in ps:
        seed = tok.decode(tok(join(p.prompt_text, p.reference)).input_ids[:seed_tokens],
                          skip_special_tokens=True)
        ch.append(len(seed)); wd.append(len(seed.split()))
    return st.mean(ch), st.mean(wd)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="results/seed_effect_runs.tsv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--lcs-thresh", type=float, default=4.0)
    ap.add_argument("--nv-thresh", type=float, default=0.01)
    a = ap.parse_args()

    rows = []
    for line in open(a.manifest, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        label, seed_tokens, comp, bp, tokenizer, pair = f[0], int(f[1]), f[2], f[3], f[4], f[5]
        if not os.path.exists(comp):
            print(f"[seed] missing {comp}, skipping {label}", file=sys.stderr)
            continue
        s_x = st.median(float(r["s_mean"]) for r in csv.DictReader(open(bp)))
        chars, words = seed_words(tokenizer, seed_tokens)
        row = dict(pair=pair, label=label, seed_tokens=seed_tokens,
                   seed_chars=round(chars, 1), seed_words=round(words, 1), s_x=round(s_x, 4))
        for metric, col, thresh, primary in (("lcs_word", "lcs_word", a.lcs_thresh, True),
                                             ("nv_recall", "nv_recall", a.nv_thresh, False)):
            o, lo, hi, nocross, _ = point_and_ci(per_passage(comp, col), thresh, s_x)
            p = "" if primary else "_nv"
            row[f"onset{p}"] = round(o, 4) if o else ""
            row[f"ratio{p}"] = round(o / s_x, 4) if o else ""
            row[f"ratio_lo{p}"] = round(lo / s_x, 4) if lo else ""
            row[f"ratio_hi{p}"] = round(hi / s_x, 4) if hi else ""
            row[f"no_crossing_pct{p}"] = round(nocross, 1)
        rows.append(row)
    if not rows:
        raise SystemExit("[seed] nothing to score")

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "seed_effect.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print(f"{'pair':22s}{'seed tok':>9s}{'words':>7s}{'s(x)':>7s}{'onset':>8s}{'ratio':>8s}"
          f"{'95% CI':>16s}{'nocross':>9s}")
    for r in sorted(rows, key=lambda r: (r["pair"], r["seed_words"])):
        ci = f"[{r['ratio_lo']:.2f},{r['ratio_hi']:.2f}]" if r["ratio_lo"] != "" else ""
        print(f"{r['pair'][:21]:22s}{r['seed_tokens']:>9d}{r['seed_words']:>7.1f}{r['s_x']:>7.3f}"
              f"{r['onset']:>8.3f}{r['ratio']:>8.3f}{ci:>16s}{r['no_crossing_pct']:>8.1f}%")

    for pair in sorted({r["pair"] for r in rows}):
        g = sorted((r for r in rows if r["pair"] == pair), key=lambda r: r["seed_words"])
        if len(g) < 3:
            continue
        rs = [r["ratio"] for r in g]
        mono = all(x > y for x, y in zip(rs, rs[1:]))
        print(f"\n{pair}: ratio against seed words "
              + " -> ".join(f"{r['seed_words']:.1f}w:{r['ratio']:.3f}" for r in g)
              + f"\n  strictly decreasing in the adversary's context: {mono}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
