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
import argparse, csv, itertools, os, statistics as st, sys

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


def _ranks(v):
    """Average ranks, so ties are handled correctly. The two KL3M pairs tie at 7.3 seed words, and
    the 1 - 6*sum(d^2)/(n(n^2-1)) shortcut is only exact without ties."""
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(x, y):
    """Pearson correlation of the average ranks."""
    rx, ry = _ranks(x), _ranks(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def permutation_p(x, y):
    """Exact two-sided p for Spearman by enumerating every permutation. n <= 8 here, so 40,320
    at worst; an asymptotic p-value is not trustworthy at these sample sizes."""
    rho, ge, tot = spearman(x, y), 0, 0
    for perm in itertools.permutations(range(len(x))):
        tot += 1
        if abs(spearman(x, [y[i] for i in perm])) >= abs(rho) - 1e-12:
            ge += 1
    return rho, ge / tot


def observational(pairs_tsv, onset_table, limit=100):
    """The cross-pair trend on the runs built for other reasons: does the onset ratio fall as the
    adversary's seed buys more words? This is CONFOUNDED by construction -- seed words is
    20 x characters-per-token, so it is the same variable as tokenizer granularity, and only the
    intervention above separates them. It is reported because the two agree."""
    tok_of, order = {}, []
    for line in open(pairs_tsv, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 5:
            tok_of[f[0]] = f[3]
            order.append(f[0])
    ratios = {r["pair"]: float(r["ratio"]) for r in csv.DictReader(open(onset_table))
              if not r["pair"].startswith("ALL")}
    rows = []
    for name in order:
        key = next((k for k in ratios if k.split(" + ")[0] == name.split(" + ")[0]), None)
        if key is None:
            continue
        _, w = seed_words(tok_of[name], 20, limit)
        rows.append((name, w, ratios[key]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="results/seed_effect_runs.tsv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--lcs-thresh", type=float, default=4.0)
    ap.add_argument("--nv-thresh", type=float, default=0.01)
    ap.add_argument("--pairs-tsv", default="results/onset_pairs.tsv")
    ap.add_argument("--onset-table", default="results/onset_table.csv")
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
        bp_rows = list(csv.DictReader(open(bp)))
        s_x = st.median(float(r["s_mean"]) for r in bp_rows)
        # k_crit is Proposition 2's running maximum on this pair's targets under THIS seed. It
        # needs the anchor only -- no memoriser, no attack, no decoding -- so it is a prediction
        # available before the sweep, not a fit to it.
        k_crit = st.median(float(r["k_crit"]) for r in bp_rows)
        chars, words = seed_words(tokenizer, seed_tokens)
        row = dict(pair=pair, label=label, seed_tokens=seed_tokens,
                   seed_chars=round(chars, 1), seed_words=round(words, 1), s_x=round(s_x, 4),
                   k_crit=round(k_crit, 4), k_crit_over_s=round(k_crit / s_x, 4))
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

    # The token-bucket account (K), pre-registered in results/onset_prediction_seed.md: within a
    # pair the onset is a fixed fraction of k_crit, so the seed moves the onset exactly as much as
    # it moves k_crit. The fraction is calibrated on the pair's own control arm and on nothing
    # else, which makes every other arm of that pair an out-of-sample prediction.
    for pair in {r["pair"] for r in rows}:
        g = [r for r in rows if r["pair"] == pair]
        ctl = next((r for r in g if "control" in r["label"] and r["onset"] != ""), None)
        c = float(ctl["onset"]) / ctl["k_crit"] if ctl else None
        for r in g:
            r["onset_over_k_crit"] = round(float(r["onset"]) / r["k_crit"], 4) if r["onset"] != "" else ""
            r["pred_ratio_K"] = round(c * r["k_crit"] / r["s_x"], 4) if c else ""
            # the null a quantitative prediction has to beat: the intervention changes nothing,
            # so the arm reads its control's ratio
            r["pred_ratio_null"] = ctl["ratio"] if ctl else ""
            r["pred_hit"] = ("" if not c or r["ratio_lo"] == "" or r is ctl else
                             float(r["ratio_lo"]) <= c * r["k_crit"] / r["s_x"] <= float(r["ratio_hi"]))

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "seed_effect.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # The rate/maximum contrast, which is the point of the intervention: the seed barely moves the
    # mean surprisal rate s(x) but moves Proposition 2's running maximum a lot, and the onset
    # follows the maximum. Written relative to each pair's control arm.
    sm = []
    for pair in sorted({r["pair"] for r in rows}):
        g = [r for r in rows if r["pair"] == pair]
        ctl = next((r for r in g if "control" in r["label"]), None)
        if ctl is None:
            continue
        for r in sorted(g, key=lambda r: r["seed_words"]):
            if r is ctl:
                continue
            d = lambda k: round(100 * (r[k] / ctl[k] - 1), 2)
            sm.append(dict(pair=pair, arm=r["label"], control=ctl["label"],
                           words=r["seed_words"], words_control=ctl["seed_words"],
                           pct_change_s_x=d("s_x"), pct_change_k_crit=d("k_crit"),
                           pct_change_onset=(round(100 * (float(r["onset"]) / float(ctl["onset"]) - 1), 2)
                                             if r["onset"] != "" and ctl["onset"] != "" else ""),
                           measured_ratio=r["ratio"], predicted_ratio_K=r["pred_ratio_K"],
                           prediction_in_ci=r["pred_hit"]))
    if sm:
        with open(os.path.join(a.out, "seed_effect_summary.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(sm[0])); w.writeheader(); w.writerows(sm)

    print(f"{'pair':22s}{'seed tok':>9s}{'words':>7s}{'s(x)':>7s}{'onset':>8s}{'ratio':>8s}"
          f"{'95% CI':>16s}{'nocross':>9s}{'k_crit':>9s}{'(K) pred':>9s}")
    for r in sorted(rows, key=lambda r: (r["pair"], r["seed_words"])):
        ci = f"[{r['ratio_lo']:.2f},{r['ratio_hi']:.2f}]" if r["ratio_lo"] != "" else ""
        print(f"{r['pair'][:21]:22s}{r['seed_tokens']:>9d}{r['seed_words']:>7.1f}{r['s_x']:>7.3f}"
              f"{r['onset']:>8.3f}{r['ratio']:>8.3f}{ci:>16s}{r['no_crossing_pct']:>8.1f}%"
              f"{r['k_crit']:>9.3f}{r['pred_ratio_K'] if r['pred_ratio_K'] != '' else '-':>9}"
              f"{'  in CI' if r['pred_hit'] is True else '  MISS' if r['pred_hit'] is False else ''}")

    for pair in sorted({r["pair"] for r in rows}):
        g = sorted((r for r in rows if r["pair"] == pair), key=lambda r: r["seed_words"])
        if len(g) < 3 or len({r["seed_words"] for r in g}) < 3:
            continue   # the temperature arms hold the seed fixed; there is no dose curve to draw
        rs = [r["ratio"] for r in g]
        mono = all(x > y for x, y in zip(rs, rs[1:]))
        print(f"\n{pair}: ratio against seed words "
              + " -> ".join(f"{r['seed_words']:.1f}w:{r['ratio']:.3f}" for r in g)
              + f"\n  strictly decreasing in the adversary's context: {mono}")
    try:
        obs = observational(a.pairs_tsv, a.onset_table)
    except (OSError, KeyError):
        obs = []
    if obs:
        # the per-pair seed length, so other analyses can condition on the adversary's context
        # without reloading seven tokenizers
        with open(os.path.join(a.out, "onset_seed_words.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["pair", "seed_tokens", "seed_words", "ratio"])
            for n, wd, r in sorted(obs, key=lambda t: t[1]):
                w.writerow([n, 20, round(wd, 1), round(r, 4)])
    if len(obs) >= 4:
        print("\ncross-pair, on the runs built for other reasons (confounded with granularity "
              "by construction):")
        for n, w, r in sorted(obs, key=lambda t: t[1]):
            print(f"   {w:5.1f} words   ratio {r:.3f}   {n[:44]}")
        for lab, sel in (("all pairs", obs), ("coarse family only", [o for o in obs if o[1] > 10])):
            if len(sel) >= 4:
                rho, pv = permutation_p([o[1] for o in sel], [o[2] for o in sel])
                print(f"   {lab:20s} n={len(sel)}  Spearman {rho:+.3f}  exact permutation p={pv:.4f}")
    scored = [r for r in rows if r["pred_hit"] in (True, False)]
    if scored:
        e = lambda k: st.mean(abs(float(r[k]) - float(r["ratio"])) / float(r["ratio"]) * 100
                              for r in scored)
        print(f"\n{len(scored)} out-of-sample arms, each predicted from its pair's control alone:")
        print(f"  {'arm':34s}{'measured':>10s}{'(K)':>9s}{'null':>9s}")
        for r in scored:
            print(f"  {r['label'][:33]:34s}{float(r['ratio']):10.4f}{float(r['pred_ratio_K']):9.4f}"
                  f"{float(r['pred_ratio_null']):9.4f}{'  in CI' if r['pred_hit'] else '  MISS'}")
        print(f"  mean |relative error|: k_crit {e('pred_ratio_K'):.1f}%, "
              f"no-change null {e('pred_ratio_null'):.1f}%")
    if sm:
        print("\nseed changes the running maximum, not the mean rate (relative to each control):")
        print(f"  {'arm':34s}{'d s(x)':>9s}{'d k_crit':>10s}{'d onset':>9s}")
        for r in sm:
            print(f"  {r['arm'][:33]:34s}{r['pct_change_s_x']:>8.1f}%{r['pct_change_k_crit']:>9.1f}%"
                  + (f"{r['pct_change_onset']:>8.1f}%" if r["pct_change_onset"] != "" else f"{'--':>9s}"))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
