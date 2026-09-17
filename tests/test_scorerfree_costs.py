"""Appendix I's scorer-free paragraph compares best-cell against best-cell, and said "At n=32".

Both numbers in "At n=32 the scorer-free rule gains 3.42x as much for 9.4% of the FLOPs" are
majority-vote-at-n=32 measured against the REWARD'S BEST CELL AT n=64 -- 0.226/0.066 and
5.75x/61.29x. At a matched n=32 they are 3.77x and 18.8%, because the reward gains 0.060 there and
costs 30.64x. The sentence read as a matched-n comparison and neither half was one. Found in the
fourth read-through, 2026-09-17; same shape as Table 1's "budget, nats" header -- two denominators
presented as one comparison.

Everything here recomputes from the CSV and the committed cost model, so a later edit cannot quietly
re-mix them.
"""
import csv

from tests.manuscript import tex

CSV = "results/selection_verifiable_comma7b.csv"
MAJ, REW = "majority vote (self-consistency)", "pointwise reward (Qwen2.5-7B)"


def _acc():
    # the two k=-1 baseline rows carry no gain (they ARE the risky model, not a gain over it)
    out = {}
    for r in csv.DictReader(open(CSV, encoding="utf-8")):
        if r["arm"] not in (MAJ, REW):
            continue
        out[(r["arm"], int(r["n"]))] = (float(r["acc"]), float(r["gain"]))
    return out


def _cost():
    """The committed parameter counts, read from the script rather than retyped."""
    import re
    src = open("analysis/serving_cost.py", encoding="utf-8").read()
    p = {k: float(re.search(rf"^{k}\s*=\s*([\d.]+)", src, re.M).group(1))
         for k in ("P_ANCHOR", "P_RISKY", "P_SCORER")}
    metered = p["P_ANCHOR"] + p["P_RISKY"]
    return (lambda n: n * (p["P_ANCHOR"] + p["P_SCORER"]) / metered,
            lambda n: n * p["P_ANCHOR"] / metered)


def test_the_best_cell_comparison_is_the_one_the_paragraph_prints():
    acc, (rew, maj) = _acc(), _cost()
    body = " ".join(open(tex("sections/appendix_selection.tex"), encoding="utf-8").read().split())
    para = body.split("Every scorer-free cell beats every reward cell")[1][:700]

    # the frame: majority at n=8, and the reward's best cell, which is n=64
    best_n = max((n for (a, n) in acc if a == REW), key=lambda n: acc[(REW, n)][0])
    assert best_n == 64, best_n
    assert f"${maj(8):.2f}\\times$" in para and f"${acc[(MAJ, 8)][0]:.3f}$" in para
    assert f"${rew(best_n):.2f}\\times$" in para and f"${acc[(REW, best_n)][0]:.3f}$" in para

    # the claim: majority at n=32 against that best cell
    gain_ratio = acc[(MAJ, 32)][1] / acc[(REW, best_n)][1]
    flops = maj(32) / rew(best_n)
    assert f"${gain_ratio:.2f}\\times$" in para, (gain_ratio, para[:200])
    assert f"${flops * 100:.1f}\\%$" in para, (flops, para[:200])

    # and the matched-n figures, which are different and are now stated
    m_ratio = acc[(MAJ, 32)][1] / acc[(REW, 32)][1]
    m_flops = maj(32) / rew(32)
    assert f"${m_ratio:.2f}\\times$" in para, (m_ratio, "the matched-n gain ratio is not stated")
    assert f"${m_flops * 100:.1f}\\%$" in para, (m_flops, "the matched-n FLOP share is not stated")
    assert "matched" in para, "the paragraph no longer says which comparison is which"


def test_the_triviaqa_sign_claim_holds_at_every_n_it_names():
    rows = {int(r["n"]): float(r["gain"]) for r in
            csv.DictReader(open("results/selection_verifiable_tqa_comma7b.csv", encoding="utf-8"))
            if r["arm"] == REW}
    assert all(g < 0 for n, g in rows.items() if n >= 8), rows
    assert any(g > 0 for n, g in rows.items() if n < 8), (rows, "then 'at every n >= 8' is idle")
