"""Plan v4 / feat-036: does the safety window close as safe models improve?

The frontier theorem says a divergence budget can be informative and usable at once only if
c_use < s(x): the rate ordinary traffic needs must be below the safe model's surprisal rate on
the protected work. Both sides are properties of the models, not of the mechanism, so the
question "does this approach get better or worse as safe models improve" is answerable by
forward passes alone. Two points already exist and point the wrong way (Comma-7B assigns the
same passages fewer nats than TinyComma-1.8B); this sweeps a real family.

Everything is scored as a TOTAL string log-probability and reported per character, which is
tokenizer-invariant. That is what lets 64k-, 128k- and 152k-vocabulary models sit in one table.
The shared-vocabulary wall at a_patch/factory.py:396 blocks fusion, never measurement.

Three text families per model:
  passage    the 758 CopyBench protected continuations (the works a rights-holder cares about)
  copybench  16 novels on a fixed 1500-character span   } identical spans, so the difference
  gutenberg  50 public-domain books, same fixed span    } isolates exposure from fluency
  ordinary   risky-model generations on neutral/factual/creative prompts (gives c_use)

c_use for safe model m is the median over ordinary generations of (nats_m - nats_risky)/chars,
an unbiased per-sequence estimate of the KL rate the decoder must permit to leave utility alone.

Usage:
  CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 \
    .venv/bin/python analysis/anchor_scaling.py --out results
"""
import argparse, csv, json, os, statistics as st, sys, time, glob

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.regimes import token_nats, regimes, load_works  # noqa: E402
from analysis.anchor_control import spans  # noqa: E402
from analysis.utility import served_prompt  # noqa: E402

# (hf id, corpus, params in B). The risky model is scored too: differencing against it gives c_use.
SAFE_MODELS = [
    ("alea-institute/kl3m-002-170m", "kl3m", 0.17),
    ("alea-institute/kl3m-002-520m", "kl3m", 0.52),
    ("alea-institute/kl3m-003-1.7b", "kl3m", 1.7),
    ("alea-institute/kl3m-003-3.7b", "kl3m", 3.7),
    ("PleIAs/Pleias-350m-Preview", "commoncorpus", 0.35),
    ("PleIAs/Pleias-1.2b-Preview", "commoncorpus", 1.2),
    ("PleIAs/Pleias-3b-Preview", "commoncorpus", 3.0),
    ("jacquelinehe/tinycomma-1.8b-llama3-tokenizer", "commonpile", 1.8),
    ("common-pile/comma-v0.1-1t", "commonpile", 7.0),
    ("common-pile/comma-v0.1-2t", "commonpile", 7.0),
]
# The canonical snapshot meta-llama/Llama-3.1-8B-Instruct in the local cache has weights but no
# tokenizer; the Meta- prefixed duplicate has both. Shard 1 is md5-identical between the two, so
# this is the same risky model the released logs used, only a complete copy of it.
RISKY = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def bare_instruction(prompt):
    """Strip the Llama-3 chat scaffolding down to the user's instruction.

    served_prompt returns 'system\\n\\nCutting Knowledge Date: ...\\n\\nuser\\n\\n<instruction>assistant'.
    Those wrapper tokens are Llama-specific; leaving them in the prefix would make every non-Llama
    safe model far more surprised by the continuation and inflate c_use in exactly the direction
    that flatters our own thesis. Score the plain instruction instead.
    """
    if "user\n\n" in prompt:
        prompt = prompt.rsplit("user\n\n", 1)[1]
    for tail in ("assistant\n\n", "assistant\n", "assistant"):
        if prompt.endswith(tail):
            prompt = prompt[: -len(tail)]
            break
    return prompt.strip()


def ordinary_texts(n, run_dir="output/sweep_chat", classes=("neutral", "factual", "creative")):
    """(class, id, prompt, risky generation) from the k=-1 runs -- the risky model unconstrained.

    The run directory is pinned rather than globbed: output/ also holds 4-row smoke runs and
    memorising-model checks, and silently scoring those would put the wrong model's output into
    c_use. The prompt is recovered with utility.served_prompt because the three classes ship
    different record shapes (only neutral has raw_text).
    """
    out = []
    for cls in classes:
        path = os.path.join(run_dir, f"trajectories_k-1_{cls}.jsonl")
        if not os.path.exists(path):
            print(f"[scaling] WARNING no {path}", flush=True)
            continue
        rows = []
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            agg = r.get("aggregate") or {}
            gen = agg.get("generation") or ""
            for head in ("assistant\n\n", "assistant\n", "assistant"):
                if gen.startswith(head):
                    gen = gen[len(head):]
                    break
            prompt = bare_instruction(served_prompt(agg))
            if len(gen) > 200 and prompt:
                rows.append((cls, r["metadata"]["prompt_id"], prompt, gen))
        rows.sort(key=lambda t: t[1])
        out += rows[: max(1, n // len(classes))]
    return out


def score_model(hf_id, corpus, params, items, works, ordinary, dtype, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=getattr(torch, dtype)).to(device).eval()
    base = dict(model=hf_id, corpus=corpus, params=params)
    rows = []

    for family, work, prefix, span in items:                       # fixed 1500-char spans
        nats, chars = token_nats(model, tok, prefix, span, device)
        r = regimes(nats, chars)
        if r:
            rows.append({**base, "family": family, "work": work, "id": work, **r})

    for w in works:                                                # the 758 protected passages
        nats, chars = token_nats(model, tok, w["prefix"], w["target"], device)
        r = regimes(nats, chars)
        if r:
            rows.append({**base, "family": "passage", "work": w["novel"], "id": w["id"], **r})

    for cls, pid, prompt, gen in ordinary:                         # risky output -> c_use
        nats, chars = token_nats(model, tok, prompt, gen, device)
        r = regimes(nats, chars)
        if r:
            rows.append({**base, "family": "ordinary", "work": cls, "id": pid, **r})

    del model
    torch.cuda.empty_cache()
    return rows


def summarise(rows, risky_id):
    """Per safe model: the protected surprisal rate, c_use from differencing against the risky
    model on identical ordinary text, and whether the informative-and-usable window is open."""
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    risky_ord = {r["id"]: r["total_nats"] / r["n_char"] for r in by_model.get(risky_id, [])
                 if r["family"] == "ordinary"}
    out = []
    for m, rs in by_model.items():
        if m == risky_id:
            continue
        get = lambda fam, key="s_rate": [r[key] for r in rs if r["family"] == fam]
        # c_use = extra nats per character the safe model needs for the risky model's own output
        deltas = [r["total_nats"] / r["n_char"] - risky_ord[r["id"]]
                  for r in rs if r["family"] == "ordinary" and r["id"] in risky_ord]
        if not deltas:
            continue
        c_use = st.median(deltas)
        s_pass = st.median(get("passage")) if get("passage") else float("nan")
        row = {
            "model": m, "corpus": rs[0]["corpus"], "params": rs[0]["params"],
            "n_passage": len(get("passage")), "n_ordinary": len(deltas),
            "c_use": c_use,
            "s_passage": s_pass,
            "s_copybench_span": st.median(get("copybench")) if get("copybench") else float("nan"),
            "s_gutenberg_span": st.median(get("gutenberg")) if get("gutenberg") else float("nan"),
            "k_crit_passage": st.median(get("passage", "k_crit")) if get("passage") else float("nan"),
            "nats_per_tok_passage": st.median(get("passage", "nats_per_tok")) if get("passage") else float("nan"),
            "margin": s_pass / c_use if c_use > 0 else float("inf"),
            "window_open": bool(c_use < s_pass),
        }
        # exposure control: Gutenberg is in the permissive corpora, CopyBench is in none of them
        if get("gutenberg") and get("copybench"):
            row["exposure_ratio"] = row["s_gutenberg_span"] / row["s_copybench_span"]
        out.append(row)
    out.sort(key=lambda r: (r["corpus"], r["params"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--cache", default="data/gutenberg")
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", nargs="*", default=None, help="subset of hf ids; default all cached")
    ap.add_argument("--risky", default=RISKY)
    ap.add_argument("--n-ordinary", type=int, default=300)
    ap.add_argument("--ordinary-run", default="output/sweep_chat")
    ap.add_argument("--limit-passages", type=int, default=None)
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = spans(a)
    works = load_works(a.data, limit=a.limit_passages)
    ordinary = ordinary_texts(a.n_ordinary, a.ordinary_run)
    print(f"[scaling] {len(items)} fixed spans, {len(works)} passages, {len(ordinary)} ordinary texts", flush=True)

    todo = [(m, c, p) for m, c, p in SAFE_MODELS if a.models is None or m in a.models]
    todo.append((a.risky, "llama", 8.0))
    rows = []
    for hf_id, corpus, params in todo:
        t0 = time.time()
        try:
            rows += score_model(hf_id, corpus, params, items, works, ordinary, a.dtype, device)
            print(f"[scaling] {hf_id:46s} ok  {time.time()-t0:6.0f}s", flush=True)
        except Exception as e:
            print(f"[scaling] {hf_id:46s} SKIP {type(e).__name__}: {e}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "anchor_scaling.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    summary = summarise(rows, a.risky)
    if not summary:
        print(f"[scaling] no summary: the risky model {a.risky} produced no ordinary rows, so c_use "
              f"cannot be differenced. Per-text rows were still written.", file=sys.stderr)
        return 1
    with open(os.path.join(a.out, "anchor_scaling_summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        wr.writeheader()
        wr.writerows(summary)

    print(f"\n{'model':46s} {'params':>7s} {'c_use':>8s} {'s(x)':>8s} {'margin':>7s}  window")
    for r in summary:
        print(f"{r['model']:46s} {r['params']:7.2f} {r['c_use']:8.4f} {r['s_passage']:8.4f} "
              f"{r['margin']:7.2f}  {'OPEN' if r['window_open'] else 'closed'}")
    print(f"\nwrote {a.out}/anchor_scaling.csv ({len(rows)} rows) and _summary.csv ({len(summary)} models)")


if __name__ == "__main__":
    sys.exit(main() or 0)
