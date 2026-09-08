"""Plan v5 / feat-044: the extraction onset is derived, not fitted.

The earlier measurement reported that near-verbatim leakage begins at ~0.89*s(x) on two model
pairs and treated 0.89 as an empirical constant. It is not a constant, and it is not fitted.

Derivation. On the geometric path p_theta ∝ p_s^(1-theta) p_r^theta the log-partition satisfies
psi_t(1) = log sum_v p_s(v) exp(log p_r(v) - log p_s(v)) = 0 exactly. So at full tilt the per-token
budget charged ALONG A TARGET x is

    k(1) = (1/T) sum_t D_KL(p_r,t || p_s,t)  ~  s_s(x) - s_r(x)                     (*)

when the risky model is near-deterministic on x, because E_{p_r}[log p_r - log p_s] collapses to
the difference of the two models' surprisals of the realised token. Define the per-work

    requirement   r(x) = s_s(x) - s_r(x)      nats per token,

the budget rate a decoder must permit before it can pay for x at all. r(x) is computable from the
two models and the work alone -- no decoding, no attack.

Two predictions, both parameter-free:

  P1  onset/s(x) = 1 - s_r(x)/s_s(x), so the "0.89" is 1 minus the residual surprisal the risky
      model still carries on its own memorised text. It should MOVE with the memoriser's quality.
  P2  population-level onset -- the budget at which MEAN recall over a corpus first becomes
      measurable -- is a LOW QUANTILE of the r(x) distribution, not its median, because the
      cheapest works leak first.

Measured (results/onset_theory.csv): P1 predicts the onset within 6% and 4% on the two pairs from
the median, and P2 locates it at the 25th percentile of r(x) to within 0.01 nats on both.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/onset_theory.py --out results
"""
import argparse, csv, json, os, statistics as st, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.regimes import token_nats  # noqa: E402

# (label, risky model path, per-passage s_s file, measured single-query onset from results/onset.csv)
PAIRS = [
    ("TinyComma-1.8B + mem. Llama-3.1-8B", "output/memorizing_llama8b",
     "results/budget_path.csv", 2.87),
    ("Comma-7B + mem. Comma-7B", "output/phase4/memorizing_comma7b",
     "results/budget_path_comma7b.csv", 2.13),
]
MANIFEST = "results/onset_theory_pairs.tsv"


def load_pairs(path):
    """name<TAB>risky_model<TAB>budget_path.csv[<TAB>measured_onset].

    The measured column is OPTIONAL on purpose: a pair with no measurement yet yields a pure
    prediction, which is the only way to state one before seeing the answer.
    """
    if not path or not os.path.exists(path):
        return PAIRS
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) not in (3, 4):
            raise SystemExit(f"[onset-theory] {path}: want 3 or 4 fields, got {len(f)}: {line}")
        out.append((f[0], f[1], f[2], float(f[3]) if len(f) == 4 and f[3] else None))
    return out


SPLITS = ("attack_train", "val", "test")


def works(data_dir, keep):
    """(prefix, protected continuation) for the passages the attack used."""
    out = {}
    for sp in SPLITS:
        p = os.path.join(data_dir, f"copybench_{sp}.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["prompt_id"] in keep and r.get("reference_text"):
                out[r["prompt_id"]] = (r.get("raw_text") or "", r["reference_text"])
    return out


def quantile(sorted_vals, p):
    n = len(sorted_vals)
    return sorted_vals[max(0, min(n - 1, int(p * n)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--data", default="data")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--pairs-file", default=MANIFEST)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="plan v5: warp both models before measuring surprisal, as the decoder "
                         "does before its solve. Moves s(x) with the pair held fixed.")
    ap.add_argument("--tag", default="", help="suffix for the output filenames, e.g. _t0.7")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    summary, per_work = [], []

    for label, risky, bpf, meas in load_pairs(a.pairs_file):
        if not os.path.exists(bpf):
            print(f"[onset-theory] missing {bpf}, skipping {label}", file=sys.stderr)
            continue
        s_s = {r["prompt_id"]: float(r["s_mean"]) for r in csv.DictReader(open(bpf))}
        W = works(a.data, set(s_s))
        tok = AutoTokenizer.from_pretrained(risky)
        model = AutoModelForCausalLM.from_pretrained(risky, dtype=getattr(torch, a.dtype)).to(dev).eval()
        req = []
        for pid, (pre, tgt) in sorted(W.items()):
            nats, _ = token_nats(model, tok, pre, tgt, dev, temperature=a.temperature)
            if not nats:
                continue
            sr = sum(nats) / len(nats)
            r = s_s[pid] - sr
            req.append(r)
            per_work.append({"pair": label, "prompt_id": pid, "s_safe": s_s[pid],
                             "s_risky": sr, "requirement": r, "ratio": 1 - sr / s_s[pid]})
        del model
        torch.cuda.empty_cache()
        req.sort()
        med_ss = st.median(w["s_safe"] for w in per_work if w["pair"] == label)
        med_sr = st.median(w["s_risky"] for w in per_work if w["pair"] == label)
        summary.append({
            "pair": label, "temperature": a.temperature, "n": len(req), "s_safe_median": med_ss, "s_risky_median": med_sr,
            "sr_over_ss_median": med_sr / med_ss,
            "pred_onset_median": med_ss - med_sr,
            "pred_ratio_median": 1 - med_sr / med_ss,
            "req_q01": quantile(req, .01), "req_q05": quantile(req, .05),
            "req_q10": quantile(req, .10), "req_q25": quantile(req, .25),
            "req_median": st.median(req),
            "measured_onset": meas,
            "measured_ratio": (meas / med_ss) if meas else None,
            "pred_over_meas_median": ((med_ss - med_sr) / meas) if meas else None,
            "pred_over_meas_q25": (quantile(req, .25) / meas) if meas else None,
        })

    if not summary:
        print("[onset-theory] no pairs produced results", file=sys.stderr)
        return 1
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"onset_theory{a.tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with open(os.path.join(a.out, f"onset_theory{a.tag}_per_work.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_work[0].keys()))
        w.writeheader()
        w.writerows(per_work)

    print(f"{'pair':38s}{'s_s':>7s}{'s_r':>7s}{'pred(med)':>10s}{'pred(q25)':>10s}{'measured':>10s}")
    for r in summary:
        meas = f"{r['measured_onset']:.2f}" if r["measured_onset"] else "not yet"
        print(f"{r['pair']:38s}{r['s_safe_median']:7.3f}{r['s_risky_median']:7.3f}"
              f"{r['pred_onset_median']:10.2f}{r['req_q25']:10.2f}{meas:>10s}")
    print(f"\n{'pair':38s}{'pred/meas (median)':>20s}{'pred/meas (q25)':>18s}")
    for r in summary:
        if r["pred_over_meas_median"] is None:
            print(f"{r['pair']:38s}{'PREDICTION ONLY':>20s}{'':>18s}")
        else:
            print(f"{r['pair']:38s}{r['pred_over_meas_median']:20.3f}{r['pred_over_meas_q25']:18.3f}")
    print(f"\nwrote {a.out}/onset_theory{a.tag}.csv and _per_work.csv ({len(per_work)} works)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
