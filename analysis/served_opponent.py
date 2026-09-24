"""Score feat-184 (results/onset_prediction_served_opponent.md) and feat-185
(results/onset_prediction_he_config.md) from the per-prompt judge files. No GPU.

The bands were committed before any judge call; this script reads them, it does not choose them.
Part B's statistic is the order-averaged LEVEL of each arm against the one opponent and the paired
difference of two arms' levels, bootstrapped over prompts.

Usage: .venv/bin/python analysis/served_opponent.py --out results
"""
import argparse
import csv
import os
import random

N_BOOT = 10000


def per_prompt(path):
    with open(path, encoding="utf-8") as fh:
        return {r["prompt_id"]: r for r in csv.DictReader(fh)}


def summary(path):
    with open(path, encoding="utf-8") as fh:
        return {r["quantity"]: r for r in csv.DictReader(fh)}


def paired(xs, rng):
    n = len(xs)
    boots = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return sum(xs) / n, boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT)]


def three_way(lo, hi, pos, neg, mid):
    return pos if lo > 0 else neg if hi < 0 else mid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=184)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    f = lambda tag: os.path.join(a.dir, f"order_averaged_h2h_per_prompt_{tag}.csv")  # noqa: E731
    s = lambda tag: os.path.join(a.dir, f"order_averaged_h2h_{tag}.csv")  # noqa: E731
    rows = []

    def add(band, quantity, value, lo="", hi="", reading="", n=""):
        rows.append(dict(band=band, quantity=quantity, value=round(value, 4) if value != "" else "",
                         lo95=round(lo, 4) if lo != "" else "", hi95=round(hi, 4) if hi != "" else "",
                         n=n, reading=reading))

    # ---- feat-184 Part A: the committed pass on de-echoed text ------------------------------
    d3 = summary(s("deecho"))["D3 difference of gains, paired"]
    v, lo, hi = float(d3["value"]), float(d3["lo95"]), float(d3["hi95"])
    a1 = ("FAILS" if lo <= 0 else "REPRODUCES" if abs(v - 0.0645) <= 0.02 else "SHIFTS")
    add("A1", "D3 on de-echoed text (committed +0.0645)", v, lo, hi, a1, d3["n"])
    add("A1", "replaces +0.0645 in the paper (|D3 - 0.0645| >= 0.005)",
        abs(v - 0.0645), reading="YES" if abs(v - 0.0645) >= 0.005 else "NO")

    # ---- feat-184 Part B: against the chat-served opponent -----------------------------------
    b1, b2 = per_prompt(f("served_k10chat")), per_prompt(f("served_committed"))
    pids = sorted(set(b1) & set(b2))
    assert len(pids) == 500, len(pids)
    g0 = all(len(x) == 500 for x in (b1, b2))
    add("G0", "prompts shared by B1 and B2", len(pids), reading="PASS" if g0 else "FAIL", n=len(pids))
    g2 = all(b1[p]["u_sel_n64"] == b2[p]["u_sel_n64"] and b1[p]["u_sel_n1"] == b2[p]["u_sel_n1"]
             for p in pids)
    add("G2", "selection levels identical in B1 and B2", float(g2), reading="PASS" if g2 else "FAIL")
    lv = lambda d, col: [float(d[p][col]) for p in pids]  # noqa: E731
    for name, d, col in (("selection n=64", b1, "u_sel_n64"), ("selection n=1", b1, "u_sel_n1"),
                         ("chat meter k=10", b1, "u_metered_k10"), ("chat anchor k=0", b1, "u_anchor_k0"),
                         ("chat opponent, 2nd draw", b1, "u_opp_chat_draw2"),
                         ("plain meter k=10 (committed)", b2, "u_metered_k10"),
                         ("plain anchor k=0 (committed)", b2, "u_anchor_k0"),
                         ("plain opponent text (committed)", b2, "u_opp_plain")):
        m, l, h = paired(lv(d, col), rng)
        add("level", f"{name} vs chat-served opponent", m, l, h, n=len(pids))
    g1 = sum(lv(b1, "u_opp_chat_draw2")) / len(pids)
    g1_ok = 0.40 <= g1 <= 0.60
    add("G1", "opponent's own second draw, level in [0.40, 0.60]", g1, reading="PASS" if g1_ok else "FAIL")
    readable = g0 and g1_ok and g2
    for band, name, d, col in (("B1", "selection n=64 minus chat meter k=10", b1, "u_metered_k10"),
                               ("B2", "selection n=64 minus plain meter k=10", b2, "u_metered_k10")):
        diffs = [x - y for x, y in zip(lv(b1, "u_sel_n64"), lv(d, col))]
        m, l, h = paired(diffs, rng)
        add(band, name, m, l, h, three_way(l, h, "HOLDS", "FAILS", "UNRESOLVED") if readable
            else "NOT READ (gate)", len(pids))
    diffs = [x - y for x, y in zip(lv(b1, "u_opp_chat_draw2"), lv(b2, "u_opp_plain"))]
    m, l, h = paired(diffs, rng)
    add("B3", "serving handicap: chat 2nd draw minus committed plain opponent text", m, l, h,
        "descriptive", len(pids))
    b3 = summary(s("served_k1chat"))["D3 difference of gains, paired"]
    add("B3k1", "chat meter k=1 (binds 15.7%): D3 on gains, descriptive", float(b3["value"]),
        float(b3["lo95"]), float(b3["hi95"]), "descriptive", b3["n"])

    # ---- feat-185: the authors' own pair ------------------------------------------------------
    if os.path.exists(s("he70b_k20")) and os.path.exists(s("he70b_k05")):
        k20, k05 = summary(s("he70b_k20")), summary(s("he70b_k05"))
        ca, cb, pa = per_prompt(f("he70b_k20")), per_prompt(f("he70b_k05")), per_prompt(f("deecho"))
        g2c = (len(ca) == len(cb) == len(pa) == 500 and
               all(ca[p][c] == cb[p][c] == pa[p][c] for p in ca for c in ("u_sel_n64", "u_sel_n1")))
        add("G2C", "selection levels identical in C-a, C-b and Part A", float(g2c),
            reading="PASS" if g2c else "FAIL")
        for band, src, q in (("C1", k20, "D3 difference of gains, paired"),
                             ("C2", k20, "D5 met70b_k1 minus selection, paired"),
                             ("C3", k05, "D3 difference of gains, paired"),
                             ("C4", k05, "D4 risky70b gain, order-averaged"),
                             ("C4", k05, "D5 risky70b minus selection, paired")):
            r = src[q]
            add(band, f"{q} ({'k=20' if src is k20 else 'k=0.5'} pass)", float(r["value"]),
                float(r["lo95"]), float(r["hi95"]), r["reading"], r["n"])
        for tag, src in (("k=20", k20), ("k=0.5", k05)):
            for q in ("D1 selection gain, order-averaged", "D2 metered gain, order-averaged"):
                r = src[q]
                add("C-gain", f"{q} ({tag} pass)", float(r["value"]), float(r["lo95"]),
                    float(r["hi95"]), r["reading"], r["n"])

    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "served_opponent.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"  {r['band']:6s} {r['quantity']:62s} {r['value']:+.4f} [{r['lo95']}, {r['hi95']}] {r['reading']}"
              if isinstance(r["value"], float) else f"  {r['band']:6s} {r['quantity']:62s} {r['value']} {r['reading']}")
    print("wrote", out)


if __name__ == "__main__":
    main()
