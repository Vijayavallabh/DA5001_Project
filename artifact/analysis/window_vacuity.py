"""feat-151: the vacuity threshold for a WINDOW event, which is where extraction is measured.

Proposition 2 puts the certificate's vacuity threshold at `K = S(x)`, with `S(x) = -log p_s(E)` the
anchor's surprisal of the event. The paper states it for `E` = "reproduce the work", and a report
asks the obvious follow-up: what is the analogue when `E` is reproducing a *window*?

It matters more than it looks, because **every extraction number in this paper is a window event**
-- near-verbatim recall is scored on 50-token windows. So the threshold the deployed budget should
be compared against is not the whole-work one, and the whole-work figure is the LENIENT one: a
shorter event is less surprising under the anchor, so its threshold is lower and the certificate
goes vacuous SOONER. Quoting only the whole-work number understates how easily a per-token budget
reaches vacuity, which is the same error caution (v) records for the uncertified interval.

The arithmetic is immediate. For a window of `w` tokens, `S_w = w * s` with `s` the anchor's
per-token surprisal on protected text, already measured per passage by `analysis/regimes.py`.
Nothing new is run.

Reads:  results/regimes_copybench.csv  (758 passages, 16 novels, the audited anchor)
Writes: <out>/window_vacuity.csv

Usage:
  .venv/bin/python analysis/window_vacuity.py --out results
"""
import argparse
import csv
import math
import os
import statistics as st

WINDOWS = (10, 20, 50, 100)
ANCHOR = "tinycomma"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="results/regimes_copybench.csv")
    ap.add_argument("--anchor", default=ANCHOR)
    ap.add_argument("--t-max", type=int, default=200,
                    help="the decoder's horizon, so K = k*T_max is comparable")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.src, encoding="utf-8"))
            if a.anchor in r["model"]]
    assert rows, f"no rows for anchor {a.anchor} in {a.src}"
    npt = sorted(float(r["nats_per_tok"]) for r in rows)
    tot = sorted(float(r["total_nats"]) for r in rows)
    ntok = sorted(float(r["n_tok"]) for r in rows)
    s_med = st.median(npt)
    p10, p90 = npt[len(npt) // 10], npt[9 * len(npt) // 10]

    print(f"[wv] {len(rows)} passages, {len({r['novel'] for r in rows})} novels, anchor "
          f"{a.anchor}: per-token surprisal median {s_med:.4f} nats "
          f"[p10 {p10:.4f}, p90 {p90:.4f}]; median passage {st.median(ntok):.0f} tokens, "
          f"{st.median(tot):.1f} nats", flush=True)

    out = []
    for w in WINDOWS:
        S_w = w * s_med
        out.append(dict(event=f"{w}-token window", window_tokens=w,
                        S_median_nats=round(S_w, 2),
                        S_p10_nats=round(w * p10, 2), S_p90_nats=round(w * p90, 2),
                        k_at_vacuity=round(S_w / a.t_max, 4),
                        selection_n_at_vacuity=f"{math.exp(S_w):.3g}",
                        selection_nats_at_n64=round(math.log(64), 4),
                        headroom_vs_n64=round(S_w / math.log(64), 1)))
    out.append(dict(event="whole passage (median)", window_tokens=int(st.median(ntok)),
                    S_median_nats=round(st.median(tot), 2),
                    S_p10_nats=round(tot[len(tot) // 10], 2),
                    S_p90_nats=round(tot[9 * len(tot) // 10], 2),
                    k_at_vacuity=round(st.median(tot) / a.t_max, 4),
                    selection_n_at_vacuity=f"{math.exp(st.median(tot)):.3g}",
                    selection_nats_at_n64=round(math.log(64), 4),
                    headroom_vs_n64=round(st.median(tot) / math.log(64), 1)))

    for r in out:
        print(f"[wv] {r['event']:24s} S = {r['S_median_nats']:8.1f} nats  "
              f"vacuous at k = {r['k_at_vacuity']:.3f} (T_max={a.t_max})  "
              f"selection at n=64 sits {r['headroom_vs_n64']:.1f}x below it", flush=True)

    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "window_vacuity.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w_ = csv.DictWriter(f, fieldnames=list(out[0]))
        w_.writeheader()
        w_.writerows(out)
    print(f"[wv] wrote {p}")
    print("[wv] READING: the window thresholds are LOWER than the whole-work one, so a per-token "
          "budget reaches vacuity sooner against the event extraction actually measures. "
          "log n is below every threshold on this grid.")


if __name__ == "__main__":
    main()
