"""Plan v4 / feat-039: the uncertified protection lives in the opening of a work.

The frontier has two computable boundaries. The certificate goes vacuous at the surprisal rate
s(x); reproduction stays impossible up to the Lindley running maximum k_crit(x) >= s(x). The
interval [s, k_crit) is protection the mechanism delivers and the certificate cannot express.

This script asks WHERE that interval comes from, and the answer changes what it is worth. The
running maximum binds at the very first token of the work in about 86% of cases, because the
bucket must pay the opening token's surprisal out of a single step's allowance. Measured on the
758 CopyBench passages under the Common Pile 7B anchor:

    whole work                 k_crit / s(x)  median 6.05x
    ignoring the first 10 tok  k_crit / s(x)  median 1.36x   (still > 1 for 89.5% of works)

So most of the uncertified protection is an opening effect, and an adversary who supplies a
genuine prefix removes it. That is precisely what the oracle-window strategy of the composition
attack does, and it is the mechanism behind a gap the audit measured but did not explain: oracle
recall 0.2301 against single-query 0.0925 at k=5 on the 8B memoriser.

It also unifies the prefix debt with the theorem. delta_init taxes the prompt and the Lindley
maximum binds at the opening: both are the same phenomenon, protection concentrated at the start
of a generation, and an adversary who owns the prefix defeats both.

Writes <out>/opening_effect.csv (per work) and <out>/opening_effect_summary.csv (per skip length).
Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/opening_effect.py --model common-pile/comma-v0.1-2t --out results
"""
import argparse, csv, os, statistics as st, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.regimes import token_nats, load_works, k_crit_rate  # noqa: E402


def binding_position(nats, chars):
    """Index at which the running maximum (cum nats)/(cum chars) is attained."""
    cum, best, arg = 0.0, -float("inf"), 0
    for i, (n, c) in enumerate(zip(nats, chars)):
        cum += n
        r = cum / max(c, 1)
        if r > best:
            best, arg = r, i
    return arg, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="common-pile/comma-v0.1-2t")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="", help="suffix for the output filenames, e.g. _comma7b")
    ap.add_argument("--skips", nargs="+", type=int, default=[0, 1, 5, 10, 20])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--denominator", choices=["char", "token"], default="char",
                    help="plan v5 erratum E: rates per character (default, tokenizer-invariant) "
                         "or per token, to show the opening effect is not a denominator artifact")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(device).eval()

    works, rows = load_works(a.data, limit=a.limit), []
    print(f"[opening] {a.model}: {len(works)} works", flush=True)
    for i, w in enumerate(works):
        nats, chars = token_nats(model, tok, w["prefix"], w["target"], device)
        if a.denominator == "token":
            chars = list(range(1, len(nats) + 1))
        if len(nats) <= max(a.skips) + 5:
            continue
        s_rate = sum(nats) / max(chars[-1], 1)
        arg, kc = binding_position(nats, chars)
        row = {"model": a.model, "id": w["id"], "novel": w["novel"], "n_tok": len(nats),
               "s_rate": s_rate, "k_crit": kc, "binding_token": arg,
               "binding_frac": arg / len(nats)}
        for sk in a.skips:
            if sk == 0:
                row["ratio_skip0"] = kc / s_rate
                continue
            base = chars[sk - 1]
            row[f"ratio_skip{sk}"] = k_crit_rate(nats[sk:], [c - base for c in chars[sk:]]) / s_rate
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(works)}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"opening_effect{a.tag}.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    summary = []
    for sk in a.skips:
        v = sorted(r[f"ratio_skip{sk}"] for r in rows)
        summary.append({"model": a.model, "skip_tokens": sk, "n": len(v),
                        "ratio_median": st.median(v), "ratio_p10": v[len(v) // 10],
                        "ratio_p90": v[9 * len(v) // 10],
                        "frac_interval_open": sum(x > 1.0 for x in v) / len(v),
                        "denominator": a.denominator,
                        "binds_at_token_0": sum(r["binding_token"] == 0 for r in rows) / len(rows),
                        "binds_in_first_tenth": sum(r["binding_frac"] < 0.1 for r in rows) / len(rows)})
    with open(os.path.join(a.out, f"opening_effect_summary{a.tag}.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        wr.writeheader()
        wr.writerows(summary)

    b0 = sum(r["binding_token"] == 0 for r in rows) / len(rows)
    b10 = sum(r["binding_frac"] < 0.1 for r in rows) / len(rows)
    print(f"\nrunning maximum binds at token 0 in {b0:.1%} of works, in the first 10% in {b10:.1%}")
    print(f"{'skip':>6s} {'median':>8s} {'p10':>7s} {'p90':>7s} {'interval open':>14s}")
    for r in summary:
        print(f"{r['skip_tokens']:6d} {r['ratio_median']:8.2f} {r['ratio_p10']:7.2f} "
              f"{r['ratio_p90']:7.2f} {r['frac_interval_open']:13.1%}")
    print(f"\nwrote {a.out}/opening_effect{a.tag}.csv ({len(rows)} works) and _summary{a.tag}.csv")


if __name__ == "__main__":
    main()
