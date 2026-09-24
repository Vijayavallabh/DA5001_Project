"""Does the constructive claim hold at more than one anchor?

Every selection-anchoring number in the paper comes from TinyComma-1.8B. The v7 reframe made that
half load-bearing, so one anchor is now the narrowest evidence we have. This aggregates the
per-anchor sweeps -- each produced by the SAME pointwise-reward and two-judge pass as feat-088, so
nothing but the anchor changes -- and scores them against the bands committed in
results/onset_prediction_selection_breadth.md before any of them was generated.

Reads:  <out>/selection_scaling{,_<tag>}.csv, one per anchor
Writes: <out>/selection_breadth.csv

Usage: .venv/bin/python analysis/selection_breadth.py --out results
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ANCHORS = [("TinyComma-1.8B (audited)", "", "jacquelinehe/tinycomma-1.8b-llama3-tokenizer",
            "output/phase5/sel_anchor8"),
           ("Pleias-1.2B", "_pleias12b", "PleIAs/Pleias-1.2b-Preview",
            "output/phase5/sel_pleias12b_8"),
           ("KL3M-1.7B", "_kl3m17b", "alea-institute/kl3m-003-1.7b",
            "output/phase5/sel_kl3m17b_8"),
           ("Comma-7B", "_comma7b", "common-pile/comma-v0.1-2t",
            "output/phase5/sel_comma7b_8"),
           # Added 2026-09-14 for C2/C3 (results/onset_prediction_selection_breadth_six.md). These
           # two are CONTROLLED contrasts against anchors already here, not extra scatter:
           # Comma-1T is Comma-7B's architecture, size and corpus at half the training tokens, and
           # Pleias-3B is Pleias-1.2B's family at 2.5x the parameters. A Spearman over six
           # heterogeneous models confounds size, family, corpus and tokenizer at once; these vary
           # one thing each, which is what C3 reads.
           ("Comma-7B (1T tokens)", "_comma1t", "common-pile/comma-v0.1-1t",
            "output/phase5/sel_comma1t_8"),
           ("Pleias-3B", "_pleias3b", "PleIAs/Pleias-3b-Preview",
            "output/phase5/sel_pleias3b_8")]
# The entry gate, exactly as registered: "mean generation length > 20 tokens and fewer than 5%
# empty completions". It is measured on the n=1 arm's own generations, not on a summary column --
# the first version of this script read `mean_tokens`, which selection_scaling.py has never
# written (it writes `mean_words`), so the gate silently read 0.0 and failed EVERY anchor including
# the audited one. A gate that fails everything is not a gate.
GATE_TOKENS = 20.0
GATE_EMPTY = 0.05
# The registered scorer. B1 is read off this judge alone; the other is secondary. Taking "either
# judge's CI excludes zero" would be two chances at the band, which the pre-registration does not
# grant and excluded alternative 3 forbids.
SCORING_JUDGE = "Phi-3.5-mini-instruct"


def spearman(xs, ys):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(o):
            r[i] = pos + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def nonempty_gain(per_prompt_csv, gen_dir, judge, n, rng, deecho=False):
    """The same gain recomputed after dropping the prompts whose n=1 completion is EMPTY.

    A base anchor sometimes emits nothing, and best-of-n will never choose an empty candidate, so
    part of any gain could be nothing more than a degeneracy filter -- an obvious and fair reading
    that the headline number cannot answer on its own. This answers it. The audited anchor is
    empty on 6.8% of prompts, above its own registered 5% gate, so the check is not hypothetical."""
    from analysis.selection_decoding import boot_mean, load_candidates
    if not (os.path.exists(per_prompt_csv) and os.path.isdir(gen_dir)):
        return None
    empty = {p: not v[0][3].strip() for p, v in load_candidates(gen_dir, deecho=deecho).items()}
    d = [float(r[f"u_n{n}"]) - float(r["u_n1"])
         for r in csv.DictReader(open(per_prompt_csv, encoding="utf-8"))
         if r["judge"] == judge and not empty.get(r["prompt_id"], True)]
    if not d:
        return None
    lo, hi = boot_mean(d, rng)
    return round(sum(d) / len(d), 4), round(lo, 4), round(hi, 4), len(d)


def gate(gen_dir, deecho=False):
    """(mean generated tokens, empty fraction, n prompts) on the n=1 arm -- rank 0, the same
    candidate the scoring pass called n=1. Tokens come from the trajectory's own
    `generation_length_tokens`, so the gate is in the anchor's tokenizer and not in words."""
    import glob
    import json
    from analysis.selection_decoding import load_candidates
    if not os.path.isdir(gen_dir):
        return float("nan"), float("nan"), 0
    cand = load_candidates(gen_dir, deecho=deecho)
    r0 = [v[0][3] for v in cand.values()]
    if not r0:
        return float("nan"), float("nan"), 0
    empty = sum(1 for g in r0 if not g.strip()) / len(r0)
    toks = []
    for f in glob.glob(os.path.join(gen_dir, "trajectories_k*_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            toks.append(json.loads(line)["aggregate"]["generation_length_tokens"])
    return (sum(toks) / len(toks) if toks else float("nan")), empty, len(r0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=99)
    # --deecho re-measures the entry gate and the empty filter on the TRUE generations (caution
    # (bc)): an empty draw that carried the prompt's tail looked non-empty, so the committed gate
    # read 6.8%/3.0%/16.6% where the text is empty on 14.6%/18.4%/31.6%. The judged u values are
    # unchanged (they are read from the committed per-prompt files); only emptiness moves.
    ap.add_argument("--deecho", action="store_true")
    # --rejudged also reads the judged u values from the de-echoed re-judge
    # (selection_scaling{tag}_deecho.csv, results/forest_deecho_note.md), so the judge read the
    # recovered text as well; writes selection_breadth_rejudged.csv.
    ap.add_argument("--rejudged", action="store_true")
    a = ap.parse_args()
    a.deecho = a.deecho or a.rejudged
    sfx = "_deecho" if a.rejudged else ""
    import random
    rng = random.Random(a.seed)

    rows, gates = [], {}
    for label, tag, model, gen_dir in ANCHORS:
        path = os.path.join(a.out, f"selection_scaling{tag}{sfx}.csv")
        if not os.path.exists(path):
            print(f"  [breadth] missing {path}, skipping {label}")
            continue
        r = list(csv.DictReader(open(path)))
        # the scored (not selecting) judge, at n = 1 and n = a.n
        judges = sorted({x["judge"] for x in r if x.get("judge")})
        for j in judges:
            arm = {int(float(x["n"])): x for x in r if x.get("judge") == j}
            if 1 not in arm or a.n not in arm:
                continue
            one, many = arm[1], arm[a.n]
            tokens, empty, npr = gates.setdefault(label, gate(gen_dir, a.deecho))
            ok = tokens >= GATE_TOKENS and empty < GATE_EMPTY
            gain = float(many["u"]) - float(one["u"])
            lo = float(many.get("gain_lo95") or "nan")
            hi = float(many.get("gain_hi95") or "nan")
            rows.append(dict(anchor=label, model=model, judge=j, n=a.n,
                             u_n1=round(float(one["u"]), 4), u_n=round(float(many["u"]), 4),
                             gain=round(gain, 4), gain_lo95=lo, gain_hi95=hi,
                             mean_tokens_n1=round(tokens, 1),
                             empty_frac_n1=round(empty, 4), n_prompts=npr,
                             entry_gate="PASS" if ok else "FAIL"))
            ne = nonempty_gain(os.path.join(a.out, f"selection_scaling_per_prompt{tag}{sfx}.csv"),
                               gen_dir, j, a.n, rng, a.deecho)
            rows[-1].update(zip(("gain_nonempty", "gain_nonempty_lo95", "gain_nonempty_hi95",
                                 "n_nonempty"), ne if ne else ("", "", "", "")))
    if not rows:
        print("  [breadth] nothing to aggregate yet")
        return 1
    os.makedirs(a.out, exist_ok=True)
    name = ("selection_breadth_rejudged.csv" if a.rejudged else
            "selection_breadth_deecho.csv" if a.deecho else "selection_breadth.csv")
    with open(os.path.join(a.out, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    for r in rows:
        ci = (f"[{r['gain_lo95']:+.4f}, {r['gain_hi95']:+.4f}]"
              if r["gain_lo95"] == r["gain_lo95"] else "[no CI]")
        print(f"  {r['anchor']:26s} {r['judge'][:28]:30s} u {r['u_n1']:.3f} -> {r['u_n']:.3f}  "
              f"gain {r['gain']:+.4f} {ci}  {r['mean_tokens_n1']:.0f} tok, "
              f"{100 * r['empty_frac_n1']:.1f}% empty -> {r['entry_gate']}")

    # B1: how many of the NEW anchors have a gain whose CI excludes 0, on the scoring judge
    new = [r for r in rows if "audited" not in r["anchor"] and r["entry_gate"] == "PASS"
           and r["judge"] == SCORING_JUDGE]
    excl = sum(1 for r in new if r["gain_lo95"] == r["gain_lo95"] and r["gain_lo95"] > 0)
    b1 = ("GENERALISES" if excl >= 2 else "PARTIAL" if excl == 1 else "SINGLE-SETUP")
    print(f"\n  B1 on the registered scorer ({SCORING_JUDGE}): {excl}/{len(new)} new anchors "
          f"have a gain CI excluding 0  ->  {b1}")
    other = [r for r in rows if "audited" not in r["anchor"] and r["entry_gate"] == "PASS"
             and r["judge"] != SCORING_JUDGE]
    if other:
        e2 = sum(1 for r in other if r["gain_lo95"] == r["gain_lo95"] and r["gain_lo95"] > 0)
        print(f"     secondary judge, reported not scored: {e2}/{len(other)}")
    failed = sorted({r["anchor"] for r in rows if r["entry_gate"] == "FAIL"})
    if failed:
        print(f"     entry gate FAILED, reported and not silently dropped: {', '.join(failed)}")
    if b1 == "SINGLE-SETUP":
        print("     the constructive claim must be narrowed to the audited pair; see the "
              "pre-registration for the exact edits that forces.")
    # B2: descriptive only -- four points cannot establish a ceiling
    per = {}
    for r in rows:
        per.setdefault(r["anchor"], []).append(r)
    xs = [sum(x["u_n1"] for x in v) / len(v) for v in per.values()]
    ys = [sum(x["gain"] for x in v) / len(v) for v in per.values()]
    if len(xs) >= 3:
        rho = spearman(xs, ys)
        verdict = ("TRACKS CAPABILITY" if rho >= 0.6 else
                   "INVERTS" if rho <= -0.6 else "NO TREND")
        note = ("descriptive, not evidence for the ceiling argument" if len(xs) < 6
                else f"C2 band: {verdict}")
        print(f"  B2/C2 Spearman(anchor-alone u, gain) = {rho:+.3f} over {len(xs)} "
              f"anchors -- {note}")
    # C3: the two within-family contrasts, each varying one thing. A rank correlation over six
    # heterogeneous models cannot separate competence from size, family, corpus or tokenizer.
    mean_gain = {k: sum(x["gain"] for x in v) / len(v) for k, v in per.items()}
    for better, worse, what in (("Comma-7B", "Comma-7B (1T tokens)", "2x the training tokens"),
                                ("Pleias-3B", "Pleias-1.2B", "2.5x the parameters")):
        if better in mean_gain and worse in mean_gain:
            d = mean_gain[better] - mean_gain[worse]
            print(f"  C3 {better} vs {worse} ({what}): gain {mean_gain[better]:+.4f} vs "
                  f"{mean_gain[worse]:+.4f}, difference {d:+.4f} -> "
                  f"{'as predicted' if d > 0 else 'AGAINST the prediction'}")
    print()
    for r in rows:
        if r["gain_nonempty"] == "":
            continue
        print(f"  empty-filter check {r['anchor']:26s} {r['judge'][:26]:28s} "
              f"gain {r['gain']:+.4f} -> {r['gain_nonempty']:+.4f} "
              f"[{r['gain_nonempty_lo95']:+.4f}, {r['gain_nonempty_hi95']:+.4f}] "
              f"on the {r['n_nonempty']} prompts whose n=1 completion is not empty")
    print("  B3 leakage is scored by analysis/selection_extraction.py per anchor; any non-zero "
          "recall goes in the main text.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
