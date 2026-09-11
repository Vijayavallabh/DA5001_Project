"""Plan v5 / feat-086: equalise the adversary's context across all nine pairs at once.

Section 4 ranks the nine pairs by the words a fixed 20-token seed buys their adversary and finds
Spearman -0.958, but seed words is 20x characters per token by construction, so that ranking cannot
separate the context from everything else granularity determines. The five interventions in
Appendix E move one pair at a time. This moves every pair to the SAME context in words and asks how
much of the spread in onset/s(x) is left.

Bands committed in results/onset_prediction_matched_context.md before the two new arms were swept:
S_match <= 0.5 * S_20 -> the context is the dominant cause; >= 0.8 * S_20 -> refuted; between ->
partial, reported as partial. Primary metric is the longest common substring in words, because a
changed seed changes the target's length and an absolute count cannot be inflated by a shorter
target -- the same choice, for the same reason, as the five seed interventions.

Input is the seed_effect.csv produced from results/matched_context_runs.tsv:

  HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache .venv/bin/python analysis/seed_effect.py \
    --manifest results/matched_context_runs.tsv --out <scratch>
  .venv/bin/python analysis/context_intervention.py --rows <scratch>/seed_effect.csv --out results
"""
import argparse
import csv
import os
import statistics as st

MATCHED = (13.6, 15.0)   # the committed matched window, in words; seed_words is stored at 1 dp,
                        # so TinyComma-1.8B (15.02 words) enters at its rounded 15.0, as the
                        # pre-registration intends when it lists the coarse family as 13.86-15.02.
DOMINANT, REFUTED = 0.5, 0.8


def spread(v):
    return max(v) - min(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default="results/seed_effect_matched.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--metric", default="ratio", choices=["ratio", "ratio_nv"])
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.rows)) if r[a.metric]]
    seed20 = {r["pair"]: r for r in rows if int(r["seed_tokens"]) == 20}
    # The matched arm is whichever arm of that pair hands the adversary 13.6-15.1 words. For the
    # five coarse pairs that is the 20-token sweep itself: they are already there.
    matched = {}
    for r in rows:
        if MATCHED[0] <= float(r["seed_words"]) <= MATCHED[1]:
            matched[r["pair"]] = r
    missing = set(seed20) - set(matched)
    if missing:
        print(f"[ctx] no matched arm yet for: {', '.join(sorted(missing))}")

    pairs = sorted(set(seed20) & set(matched))
    u = [float(seed20[p][a.metric]) for p in pairs]
    m = [float(matched[p][a.metric]) for p in pairs]
    su, sm = spread(u), spread(m)
    frac = sm / su if su else float("nan")
    verdict = ("DOMINANT: the context carries most of the split" if frac <= DOMINANT else
               "REFUTED: equalising the context does not close the split" if frac >= REFUTED else
               f"PARTIAL: the context carries {1 - frac:.0%} of the spread")

    w = max(len(p) for p in pairs)
    print(f"{'pair':{w}s}  seed20 words ratio   matched words ratio   move")
    for p in pairs:
        s, t = seed20[p], matched[p]
        print(f"{p:{w}s}  {float(s['seed_words']):11.1f} {float(s[a.metric]):5.3f}   "
              f"{float(t['seed_words']):12.1f} {float(t[a.metric]):5.3f}   "
              f"{float(t[a.metric]) - float(s[a.metric]):+6.3f}")
    print(f"\n{len(pairs)} pairs, metric {a.metric}")
    print(f"  spread at the benchmark's 20-token seed : {su:.4f}  (cv {st.pstdev(u)/st.mean(u):.1%})")
    print(f"  spread at a matched {MATCHED[0]}-{MATCHED[1]} word seed : {sm:.4f}  "
          f"(cv {st.pstdev(m)/st.mean(m):.1%})")
    print(f"  S_match / S_20 = {frac:.3f}  -> {verdict}")

    out = os.path.join(a.out, "context_intervention.csv")
    with open(out, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["pair", "seed_tokens_20", "words_20", "ratio_20", "seed_tokens_matched",
                      "words_matched", "ratio_matched", "delta"])
        for p in pairs:
            s, t = seed20[p], matched[p]
            wtr.writerow([p, s["seed_tokens"], s["seed_words"], s[a.metric], t["seed_tokens"],
                          t["seed_words"], t[a.metric],
                          round(float(t[a.metric]) - float(s[a.metric]), 4)])
        wtr.writerow(["ALL", len(pairs), a.metric, round(su, 4), "", "", round(sm, 4),
                      round(frac, 4)])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
