"""Plan v4 / feat-035: the three regime boundaries of a divergence-budgeted decoder, computed
from the safe model and the protected work alone -- no access to the risky model, no decoding.

For a work x served under a token bucket of rate k, write s_t for the safe model's surprisal of
the t-th token of x and delta for the bucket's initial debt. Then

    s(x)        = sum_t s_t / chars(x)                     the mean surprisal RATE
    k_crit(x)   = max_t (sum_{i<=t} s_i + delta) / chars_t  the Lindley running maximum
    c_use       = the divergence rate ordinary traffic needs (supplied; see anchor_scaling.py)

and the certificate is non-vacuous only below s(x), the mechanism protects up to k_crit(x), and
utility survives only above c_use. k_crit >= s always, so [s, k_crit) is an interval on which the
mechanism protects without certifying.

Rates are in nats per CHARACTER, not per token. Total string log-probability is tokenizer-
invariant but a per-token rate is not, and this module is meant to be run across model families
with 64k, 128k and 152k vocabularies. Multiply by the deployed tokenizer's chars-per-token to
recover the mechanism's own k.

Usage:
  CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 \
    .venv/bin/python analysis/regimes.py --model common-pile/comma-v0.1-2t \
      --data data --out results/regimes.csv
"""
import argparse, csv, json, math, os, statistics as st, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SPLITS = ("attack_train", "val", "test")


def char_offsets(tok, text, n_tok):
    """Cumulative character count after each token. Uses the fast tokenizer's offset mapping when
    there is one; otherwise spreads characters evenly, which only matters for the within-work
    shape of k_crit, never for the totals."""
    try:
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        offs = [e for _, e in enc["offset_mapping"]]
        if len(offs) == n_tok and offs and offs[-1] > 0:
            # offsets can repeat on byte-level merges; make the sequence strictly usable
            return [max(o, i + 1) for i, o in enumerate(offs)]
    except Exception:
        pass
    per = len(text) / max(n_tok, 1)
    return [max(int(round((i + 1) * per)), i + 1) for i in range(n_tok)]


@torch.no_grad()
def token_nats(model, tok, prefix, target, device):
    """Per-token surprisal of `target` given `prefix`, plus the cumulative character count.

    The target is tokenized on its own (add_special_tokens=False) so that the character offsets
    line up with the target string, which is what makes the rate comparable across tokenizers.
    """
    p_ids = tok(prefix).input_ids if prefix else [tok.bos_token_id or tok.eos_token_id]
    t_ids = tok(target, add_special_tokens=False).input_ids
    if not t_ids:
        return [], []
    ids = torch.tensor([p_ids + t_ids], device=device)
    logits = model(ids).logits[0, :-1].float()          # row j predicts ids[j+1]
    logp = torch.log_softmax(logits[len(p_ids) - 1:], dim=-1)
    nats = -logp.gather(1, ids[0, len(p_ids):].unsqueeze(1)).squeeze(1)
    return nats.tolist(), char_offsets(tok, target, len(t_ids))


def k_crit_rate(nats, chars, delta=0.0):
    """max_t (sum_{i<=t} nats_i + delta) / chars_t -- the smallest rate at which the bucket, started
    at a debt of delta, can pay for every prefix of the work. Prop. 4 in nats-per-character form."""
    cum, best = 0.0, -float("inf")
    for n, c in zip(nats, chars):
        cum += n
        best = max(best, (cum + delta) / max(c, 1))
    return best


def regimes(nats, chars, c_use=None, delta=0.0):
    """The three boundaries and the verdict. c_use may be None when it is not yet measured."""
    if not nats:
        return None
    total, n_char = sum(nats), max(chars[-1], 1)
    s_rate = (total + delta) / n_char
    kc = k_crit_rate(nats, chars, delta)
    out = {
        "n_tok": len(nats), "n_char": n_char, "total_nats": total,
        "s_rate": s_rate,                 # certificate informative only below this
        "k_crit": kc,                     # mechanism protects only below this
        "uncertified_width": kc / s_rate if s_rate > 0 else float("nan"),
        "nats_per_tok": total / len(nats),
    }
    if c_use is not None:
        out["c_use"] = c_use
        # informative (k < s_rate) AND usable (k >= c_use) is a non-empty interval iff c_use < s_rate
        out["window_open"] = c_use < s_rate
        out["window_width"] = max(0.0, s_rate - c_use)
        out["margin"] = s_rate / c_use if c_use > 0 else float("inf")
    return out


def load_works(data_dir, splits=SPLITS, limit=None):
    """CopyBench (prefix, protected continuation) pairs."""
    works = []
    for sp in splits:
        path = os.path.join(data_dir, f"copybench_{sp}.jsonl")
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ref = r.get("reference_text") or r.get("expected_answer")
            if not ref:
                continue
            works.append({"id": r.get("prompt_id"), "split": sp, "novel": r.get("source_novel"),
                          "prefix": r.get("raw_text") or "", "target": ref})
    return works[:limit] if limit else works


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id of the safe model")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/regimes.csv")
    ap.add_argument("--c-use", type=float, default=None, help="nats/char the risky model needs on ordinary traffic")
    ap.add_argument("--delta", type=float, default=0.0, help="initial bucket debt in nats (0 = pure work property)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype)).to(device).eval()

    works = load_works(a.data, limit=a.limit)
    print(f"[regimes] {a.model}: {len(works)} works", flush=True)
    rows = []
    for i, w in enumerate(works):
        nats, chars = token_nats(model, tok, w["prefix"], w["target"], device)
        r = regimes(nats, chars, a.c_use, a.delta)
        if r is None:
            continue
        rows.append({"model": a.model, "id": w["id"], "split": w["split"], "novel": w["novel"], **r})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(works)}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    med = lambda key: st.median([r[key] for r in rows])
    print(f"[regimes] wrote {a.out}: n={len(rows)} median s_rate={med('s_rate'):.4f} "
          f"k_crit={med('k_crit'):.4f} nats/char, uncertified width x{med('uncertified_width'):.2f}", flush=True)


if __name__ == "__main__":
    main()
