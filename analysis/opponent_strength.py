"""feat-175: is the judged head-to-head a function of the opponent's STRENGTH?

`onset_prediction_second_opponent.md` swapped `Llama-3.1-8B-Instruct` for `Qwen2.5-14B-Instruct`
and the paired difference `g_sel - g_met` went from `+0.0645 [+0.0300, +0.0995]` to
`-0.0065 [-0.0385, +0.0255]`. Two points, and family and size moved together, so nothing is
identified: the appendix says only that the difference "does not survive", which is a statement
about one swap.

This places every opponent on a measured axis and asks whether that axis orders the outcome.

    opponent strength := 1 - mean(u_anchor_k0)

`u_anchor_k0` is the anchor control's ORDER-AVERAGED win rate against that opponent. Caution (ap)
forbids comparing a judged LEVEL across passes, and this is a level, so the exemption has to be
argued rather than assumed:

  * the reference text is `output/sweep_plain`, byte-identical in every pass;
  * order averaging judges both presentation orders, so it draws nothing from the rng, and caution
    (ap)'s own measurement is that it is then a deterministic function of the text under a greedy
    judge -- it reproduced a committed pass to four decimals where single-order moved `0.066`;
  * the judge, its template and its seed are fixed.

So the number differs between passes ONLY through the opponent's text, which is the variable. That
is the one level comparison these rules permit, and it is permitted for a reason that does not
generalise to gains measured against different opponents.

G3, the degeneracy gate, exists because a small instruct model is not automatically an opponent:
one that emits empties or truncates is a weak opponent for the wrong reason, and the strength axis
would read it as capability. Both halves are derived from the committed opponent's own
generations -- never typed (caution (v)).

Usage:
  .venv/bin/python analysis/opponent_strength.py --out results
"""
import argparse
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_decoding import CLASSES, load_baseline  # noqa: E402

# (label, FILE SUFFIX, generation dir, status). order_averaged_h2h.py writes `_` + its --tag, so
# `--tag _opp2` lands on `..__opp2.csv`; these are the suffixes as they appear on disk, not the tags.
ARMS = [
    ("meta-llama/Llama-3.1-8B-Instruct", "", "output/sweep_plain", "committed"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "__opp_qwen05b", "output/opponent_qwen05b", "new"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "__opp_qwen15b", "output/opponent_qwen15b", "new"),
    ("Qwen/Qwen2.5-3B-Instruct", "__opp_qwen3b", "output/opponent_qwen3b", "new"),
    ("Qwen/Qwen2.5-14B-Instruct", "__opp2", "output/opponent_qwen14b", "on record"),
]


def pipeline(run_dir):
    """Which GENERATOR wrote this opponent's completions, read off the records themselves.

    Caution (at): two arms compared must have come from the same pipeline, and the runs record
    enough to check it mechanically. They do NOT all match here, which is exactly why this column
    exists. The committed opponent is an `h1.py` sweep; every opponent on this ladder comes from
    `analysis/blocklist_decode.py`, which generates one prompt at a time under a per-prompt seed
    instead of batching. `blocklist_decode` stamps `blocklist_ngram` into every record and `h1.py`
    never does, so the two are distinguishable without trusting a directory name.

    The consequence is stated rather than hidden: H1's threshold is the committed opponent's own
    strength, so H1 is a CROSS-PIPELINE reading, while the four Qwen opponents share one pipeline
    and one family and are the clean dose-response the ladder was designed to be."""
    if not os.path.isdir(run_dir):
        return "?"
    for cls in CLASSES:
        f = os.path.join(run_dir, f"trajectories_k-1_{cls}.jsonl")
        if not os.path.exists(f):
            continue
        m = json.loads(open(f, encoding="utf-8").readline())["metadata"]
        return "blocklist_decode" if "blocklist_ngram" in m else "h1.py"
    return "?"


def per_prompt(sfx):
    p = f"results/order_averaged_h2h_per_prompt{sfx}.csv"
    return list(csv.DictReader(open(p, encoding="utf-8"))) if os.path.exists(p) else None


def d3_row(sfx):
    """D3 is READ from the pass that produced it, never recomputed.

    order_averaged_h2h.py draws its bootstrap from an rng the judging pass also consumes, so a
    reimplementation here cannot reproduce the committed interval and would disagree in the last
    digit for a reason that is not a defect -- caution (j), and the exact trap it describes. The
    strength column below is computed here because nothing else computes it."""
    p = f"results/order_averaged_h2h{sfx}.csv"
    if not os.path.exists(p):
        return None
    rows = [r for r in csv.DictReader(open(p, encoding="utf-8"))
            if r["quantity"].startswith("D3")]
    assert len(rows) == 1, p
    return rows[0]


def shape(run_dir, pids=None):
    """(n, empty fraction, mean words) of the opponent completions THE JUDGE SAW.

    blocklist_decode.py --split ordinary writes 850 generations and order_averaged_h2h.py judges
    the 500 that are present in all four arms as well, so a gate computed over the whole run is a
    gate on 350 texts nobody was shown. `pids` restricts it to the judged intersection, read off
    that pass's own per-prompt file."""
    if not os.path.isdir(run_dir):
        return None
    served = load_baseline(run_dir)
    texts = [t for p, t in served.items() if pids is None or p in pids]
    if not texts:
        return None
    return (len(texts),
            sum(1 for t in texts if not t.strip()) / len(texts),
            statistics.fmean(len(t.split()) for t in texts))


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def exact_p(xs, ys):
    """Two-sided exact permutation p for the rank correlation. n is 5 at most; 120 permutations."""
    import itertools
    obs = abs(spearman(xs, ys))
    perms = list(itertools.permutations(range(len(ys))))
    hit = sum(1 for p in perms if abs(spearman(xs, [ys[i] for i in p])) >= obs - 1e-12)
    return hit / len(perms)


FLOOR = 0.30             # H2's low end, fixed in the registration
MAX_EMPTY = 0.10         # G3, fixed in the registration
LEN_FACTOR = 3.0         # G3, fixed in the registration


def g3(empty_frac, mean_words, ref_words):
    """G3: an opponent that is degenerate is not a weak opponent.

    Both halves are relative to the COMMITTED opponent's own generations, which the caller derives
    from output/sweep_plain -- caution (v): a reference number carries its protocol, and a typed
    one has lost it. A truncating or empty-emitting run would read as low strength and the axis
    would call that capability, which is caution (au)'s shape."""
    ok = empty_frac < MAX_EMPTY and (ref_words / LEN_FACTOR) <= mean_words <= ref_words * LEN_FACTOR
    return "PASS" if ok else "FAIL"


def h1_h2(rows):
    """Decide H1 and H2 from the scored rows. Bands: onset_prediction_opponent_ladder.md.

    The reading is computed, not written down afterwards (caution (av)): a verdict is a string that
    outlives the number under it, so both bands are re-derived from the table every time.
    """
    ref = next((r for r in rows if r["status"] == "committed"), None)
    assert ref, "the committed pass is not in the table; H1's threshold has no source"
    thr = ref["strength"]
    new = [r for r in rows if r["status"] == "new"
           and r["g3"] == "PASS" and r.get("g1", "PASS") == "PASS"]
    below = [r for r in new if r["strength"] < thr]

    # H2 first: a floor artefact is reported INSTEAD of a refutation for that opponent, so it has
    # to be decided before H1 reads the same row.
    floor_unres = [r for r in below if r["strength"] < FLOOR
                   and r["verdict"] == "REVERSAL UNRESOLVED"]
    mid_conf = [r for r in below if FLOOR <= r["strength"] < thr
                and r["verdict"] == "REVERSAL CONFIRMED"]
    h2 = "COMPRESSION AT BOTH ENDS" if floor_unres and mid_conf else "NOT FIRED"

    scored = [r for r in below if not (h2 == "COMPRESSION AT BOTH ENDS" and r in floor_unres)]
    if not scored:
        h1 = "NOT TESTED"
    elif all(r["verdict"] == "REVERSAL CONFIRMED" for r in scored):
        h1 = "STRENGTH SUPPORTED"
    else:
        h1 = "STRENGTH REFUTED"
    return h1, h2, thr, [r["opponent"] for r in scored], [r["opponent"] for r in below]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    ref_pp = per_prompt("")
    assert ref_pp, "the committed pass's per-prompt file is missing; G1 and G3 have no reference"
    ref_pids = {r["prompt_id"] for r in ref_pp}
    ref = shape("output/sweep_plain", ref_pids)
    assert ref, "the committed opponent's generations are missing; G3 has no derived reference"
    ref_n, ref_empty, ref_words = ref
    print(f"[G1/G3 reference, DERIVED from output/sweep_plain over the committed pass's own "
          f"{ref_n} judged prompts] empty={ref_empty:.4f} mean_words={ref_words:.2f}")

    rows = []
    for model, tag, run_dir, status in ARMS:
        pp = per_prompt(tag)
        if pp is None:
            print(f"[skip] {model}: results/order_averaged_h2h_per_prompt{tag}.csv not written yet")
            continue
        pids = {r["prompt_id"] for r in pp}
        g1 = "PASS" if pids == ref_pids else "FAIL"
        row = d3_row(tag)
        assert row, f"{model}: the per-prompt file exists but the D3 file does not"
        anchor = [float(r["u_anchor_k0"]) for r in pp]
        strength = 1.0 - statistics.fmean(anchor)
        d3, lo, hi = float(row["value"]), float(row["lo95"]), float(row["hi95"])
        # ONE vocabulary, and it is the producing script's. Caution (av): a verdict is a string
        # that outlives the number under it, so it is re-derived from the interval here and
        # asserted equal to what the pass recorded rather than restated in new words.
        verdict = ("REVERSAL CONFIRMED" if lo > 0 else
                   "REVERSAL REFUTED" if hi < 0 else "REVERSAL UNRESOLVED")
        assert verdict == row["reading"].strip(), \
            f"{model}: {verdict} from [{lo}, {hi}] disagrees with the recorded {row['reading']}"

        sh = shape(run_dir, {r["prompt_id"] for r in pp})
        if sh is None:
            g3v, g3n, g3e, g3w = "NOT SCORED", 0, float("nan"), float("nan")
        else:
            g3n, g3e, g3w = sh
            g3v = g3(g3e, g3w, ref_words)

        rows.append(dict(opponent=model, status=status, n=len(pp),
                         strength=round(strength, 4),
                         anchor_win=round(statistics.fmean(anchor), 4),
                         d3=d3, d3_lo95=lo, d3_hi95=hi,
                         verdict=verdict, g1=g1, g3=g3v,
                         generator=pipeline(run_dir), n_gen=g3n,
                         empty_frac=round(g3e, 4), mean_words=round(g3w, 2)))

    rows.sort(key=lambda r: r["strength"])
    out = os.path.join(a.out, "opponent_ladder.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'opponent':<34} {'str':>6} {'D3':>8} {'lo':>8} {'hi':>8}  {'G3':<10} verdict")
    for r in rows:
        print(f"{r['opponent']:<34} {r['strength']:>6.3f} {r['d3']:>+8.4f} "
              f"{r['d3_lo95']:>+8.4f} {r['d3_hi95']:>+8.4f}  {r['g3']:<10} {r['verdict']}")

    if any(r["status"] == "new" for r in rows):
        h1, h2, thr, scored, below = h1_h2(rows)
        print(f"\n[H1] threshold {thr:.4f} (the committed opponent's measured strength)")
        print(f"[H1] new opponents below it: {below or 'none'}")
        print(f"[H1] scored under H1:        {scored or 'none'}")
        print(f"[H1] **{h1}**")
        print(f"[H2] **{h2}**")

    def rank_report(label, sel):
        u = [r for r in rows if r["g3"] in ("PASS", "NOT SCORED") and sel(r)]
        if len(u) < 3:
            return
        xs, ys = [r["strength"] for r in u], [r["d3"] for r in u]
        print(f"\n[{label}] Spearman(strength, D3) = {spearman(xs, ys):+.4f} over "
              f"{len(u)} opponents, exact two-sided p = {exact_p(xs, ys):.4f}")
        print("          " + ", ".join(f"{r['strength']:.3f}->{r['d3']:+.4f}" for r in u))

    # The five-point series mixes generators (see pipeline()); the four-point one does not, and
    # the registration's own words for the ladder are "within one family so that size is the only
    # thing that moves". Both are exploratory and both include points already seen.
    rank_report("H3, EXPLORATORY -- all five, MIXED generators", lambda r: True)
    rank_report("H3, EXPLORATORY -- one generator, one family",
                lambda r: r["generator"] == "blocklist_decode")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
