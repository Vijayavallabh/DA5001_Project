"""feat-169: does TokenSwap's suppression track |G|, or the probability mass G carries?

feat-167 saw suppression fail at KL3M-170m, the rung with both the fewest G token ids and the least
mass, and the cached tokenizers cannot separate the two. This reads the constructed ladder --- one
auxiliary, DistilGPT-2, with G shrunk on purpose --- and then does the one thing that decides it:
it predicts KL3M's recall from each candidate axis by interpolating the ladder, and compares both
predictions against what KL3M actually read. An axis that cannot predict the held-out point is not
the axis.

KL3M is genuinely held out: it is a different tokenizer, it is not on the ladder, and its reading
was committed in feat-167 before this ladder existed.
"""
import argparse
import csv
import os

# (tag, words). Seeds 0 and 1 at every shrunk rung; the full-G control has no subsample to vary.
RUNGS = [("w110s0", 110), ("w85s0", 85), ("w85s1", 85), ("w65s0", 65), ("w65s1", 65),
         ("w44s0", 44), ("w44s1", 44), ("w25s0", 25), ("w25s1", 25),
         ("w10s0", 10), ("w10s1", 10)]
SUPPRESSES, LEAKS = 0.01, 0.05


def _gates(root):
    path = os.path.join(root, "tokenswap_gates.csv")
    return {r["arm_dir"]: r for r in csv.DictReader(open(path, encoding="utf-8"))
            if r["arm"] == "tokenswap"}


def _recall(root, tag):
    path = os.path.join(root, f"blocklist_decode__tsg_{tag}.csv")
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["arm"] == "tokenswap"]
    assert len(rows) == 1, path
    return rows[0]


def interp(points, x):
    """Predict y at x by linear interpolation on points sorted by x; clamp outside the range.

    Deliberately the dumbest estimator that respects the data: no fit, no parameters, nothing that
    could be tuned toward the answer. Outside the ladder's range it returns the nearest endpoint,
    which is a CONSERVATIVE prediction for a held-out point below the range -- it can only
    understate how badly a low-mass rung leaks.
    """
    pts = sorted(points)
    if x <= pts[0][0]:
        return pts[0][1], "clamped below the ladder"
    if x >= pts[-1][0]:
        return pts[-1][1], "clamped above the ladder"
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0), "interpolated"
    raise AssertionError(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/tokenswap_gsize.csv")
    a = ap.parse_args()

    g = _gates(a.results)
    rows = []
    for tag, words in RUNGS:
        gate = g["gsize_" + tag]
        rec = _recall(a.results, tag)
        nv = float(rec["nv_recall_mean"])
        rows.append({
            "tag": tag, "words": words, "g_token_ids": int(gate["g_token_ids"]),
            "mass_on_g": float(gate["mass_on_g"]), "bind_rate": float(gate["bind_rate"]),
            "nv_recall": nv, "lcs_word": float(rec["lcs_word_mean"]),
            "rouge_ge_0p5": int(rec["rouge_ge_0p5_count"]),
            "rouge_ge_0p3": int(rec["rouge_ge_0p3_count"]),
            "reading": "SUPPRESSES" if nv <= SUPPRESSES else "LEAKS" if nv > LEAKS else "PARTIAL",
        })

    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'rung':<8}{'words':>6}{'|G|':>6}{'mass':>8}{'binds':>8}"
          f"{'nv_recall':>11}{'lcs':>7}{'>=0.5':>7}  reading")
    for r in rows:
        print(f"{r['tag']:<8}{r['words']:>6}{r['g_token_ids']:>6}{r['mass_on_g']:>8.4f}"
              f"{r['bind_rate']:>8.4f}{r['nv_recall']:>11.4f}{r['lcs_word']:>7.2f}"
              f"{r['rouge_ge_0p5']:>7}  {r['reading']}")

    sup = [r for r in rows if r["reading"] == "SUPPRESSES"]
    leak = [r for r in rows if r["reading"] == "LEAKS"]
    print(f"\nB1: smallest |G| that suppresses on both seeds: "
          f"{min(r['g_token_ids'] for r in sup)}  (mass {min(r['mass_on_g'] for r in sup):.4f})")
    print(f"    largest |G| that leaks: {max(r['g_token_ids'] for r in leak)}  "
          f"(mass {max(r['mass_on_g'] for r in leak):.4f})")

    # The held-out point: KL3M-170m, a different tokenizer, scored in feat-167 before this ladder
    # existed. Its |G| sits inside the ladder's range; its mass sits below it.
    kl = _gates(a.results)["leak_kl3m170m"]
    klrec = [r for r in csv.DictReader(
        open(os.path.join(a.results, "blocklist_decode__tsleak_kl3m170m.csv"), encoding="utf-8"))
        if r["arm"] == "tokenswap"][0]
    actual = float(klrec["nv_recall_mean"])
    print(f"\nHELD OUT --- KL3M-170m: |G| = {kl['g_token_ids']}, mass = {float(kl['mass_on_g']):.4f}, "
          f"binds {float(kl['bind_rate']):.4f}, ACTUAL recall {actual:.4f}")
    for axis, key in (("|G|", "g_token_ids"), ("mass on G", "mass_on_g"),
                      ("bind rate", "bind_rate")):
        pred, how = interp([(float(r[key]), r["nv_recall"]) for r in rows], float(kl[key]))
        err = abs(pred - actual)
        verdict = "PREDICTS" if err <= 0.05 else "MISPREDICTS"
        print(f"  from {axis:<10} -> {pred:.4f} ({how:<26}) |err| {err:.4f}  {verdict}")


if __name__ == "__main__":
    main()
