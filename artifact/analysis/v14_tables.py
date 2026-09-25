"""Print the v14 appendix tables' rows from their CSVs, so no cell is typed by hand (caution (j)).

  .venv/bin/python analysis/v14_tables.py cpk|factorial|long|short
"""
import csv
import os
import sys

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def pas(tag):
    return {(r["quantity"], r["arm"]): r for r in csv.DictReader(open(os.path.join(R, f"matched_h2h_{tag}.csv")))}


def band(r, nd=4):
    return f"${float(r['value']):+.{nd}f}$ $[{float(r['lo95']):+.{nd}f}, {float(r['hi95']):+.{nd}f}]$"


def val(r, nd=4):
    return f"${float(r['value']):+.{nd}f}$"


def cpk():
    B, G = pas("cpk_B_hostb"), pas("cpk_G")
    arms = {r["certificate_nats"]: r for r in csv.DictReader(open(os.path.join(R, "cpk_baseline.csv")))}
    leak = {r["certificate_nats"]: r for r in csv.DictReader(open(os.path.join(R, "cpk_extraction.csv")))}
    for c, name in (("4.1589", "4.16"), ("33.271", "33.27"), ("83.178", "83.18"), ("159.83", "159.83")):
        a, lk = arms[c], leak[c]
        g = f"cpk_{name} - anchor_k0"
        print(f"${float(c):.2f}$ & ${float(a['kappa']):.2f}$ & ${float(a['served_risky_pct']):.1f}\\%$ & "
              f"{band(B[('gain', g)])} & {val(G[('gain', g)])} & ${float(lk['nv_recall_mean']):.4f}$ \\\\")


def factorial():
    for r in csv.DictReader(open(os.path.join(R, "scorer_judge_factorial.csv"))):
        cells = []
        for k in ("S", "D3_qwen", "D3_gemma"):
            cells.append(f"${float(r[k]):+.4f}$ $[{float(r[k + '_lo']):+.4f}, {float(r[k + '_hi']):+.4f}]$")
        print(f"{r['judge']} & " + " & ".join(cells) + " \\\\")


def long_():
    B, G = pas("f213_B"), pas("f213_G")
    L = {r["arm"]: r for r in csv.DictReader(open(os.path.join(R, "long_outputs.csv")))}
    rows = (("best of $64$, whole output", "sel_n64", "sel_n1", "4.16", "4.16"),
            ("best of $16$", "sel_n16", "sel_n1", "2.77", "2.77"),
            ("installments, $L=100$, $n=8$", "inst", "sel_n1", "4.16", "20.79"),
            ("pathwise meter", "pw_4.16", "anchor_k0", "---", "4.16"),
            ("", "pw_20.8", "anchor_k0", "---", "20.79"),
            ("budget up front", "front_4.16", "anchor_k0", "---", "4.16"),
            ("windowed meter", "win_4.16", "anchor_k0", "4.16", "83.18"),
            ("anchored decoding, $k=0.0208$", "kl_20.8", "anchor_k0", "---", "20.79"),
            ("anchored decoding, $k=0.1$", "kl_0.1", "anchor_k0", "---", "100"),
            ("anchored decoding, $k=0.5$", "kl_0.5", "anchor_k0", "---", "500"),
            ("anchored decoding, $k=10$", "kl_10", "anchor_k0", "---", "10^4"))
    for label, arm, ctrl, span, out in rows:
        g = f"{arm} - {ctrl}"
        print(f"{label} & ${span}$ & ${out}$ & ${float(L[arm]['mean_tokens']):.0f}$ & "
              f"{band(B[('gain', g)])} & {val(G[('gain', g)])} \\\\".replace("$---$", "---"))


def short():
    """tab:shortworks: per-draw exact reproduction of a quotation's second half, in percent, by mechanism and stratum."""
    S = {(r["stratum"], r["quantity"]): r for r in csv.DictReader(open(os.path.join(R, "short_works.csv")))}

    def rate(st, q):
        r = S[(st, q)]
        return f"${100 * float(r['value']):.2f}$ $[{100 * float(r['lo95']):.2f}, {100 * float(r['hi95']):.2f}]$"
    rows = (("anchor alone, $k=0$", "anchor_exact", "0", None),
            ("selection, worst case, $n=8$", "sel_worst_n8", "2.08", "log 8"),
            ("selection, worst case, $n=64$", "sel_worst_n64", "4.16", "log 64"),
            ("meter, $k=\\log(64)/64$", "meter_0.0649836_exact", "4.16", "64k, k=0.0649836"),
            ("meter, $k=0.5$", "meter_0.5_exact", "32", "64k, k=0.5"),
            ("meter, $k=1$", "meter_1_exact", "64", "64k, k=1"),
            ("meter, $k=3$", "meter_3_exact", "192", "64k, k=3"),
            ("meter, $k=10$", "meter_10_exact", "640", "64k, k=10"),
            ("the $70$B alone, $k=-1$", "risky_exact", "---", None))
    for label, q, K, share in rows:
        v = "---" if share is None else f"${100 * float(S[('protected', f'share S_anchor <= {share}')]['value']):.1f}\\%$"
        print(f"{label} & {'---' if K == '---' else '$' + K + '$'} & {v} & {rate('protected', q)} & "
              f"{rate('public_domain', q)} \\\\")


if __name__ == "__main__":
    {"cpk": cpk, "factorial": factorial, "long": long_, "short": short}[sys.argv[1]]()
