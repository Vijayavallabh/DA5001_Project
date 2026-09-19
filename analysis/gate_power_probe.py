"""Does feat-136's REPAIRED G0a threshold actually have power against the defects it names?

Zero committed bands: this is a diagnostic, run as a devil's-advocate check on our own gate.

G0a's surviving threshold is "mean |diff| below 1.0 nat", taken from the pre-registration's own
words: the gate exists because "a wrong template, a padding-side flip, a dtype error or a tokenizer
mismatch moves a reward by whole nats". That sentence is an ASSERTION about defect magnitude, written
before any data, and the withdrawn thresholds were also an assertion about magnitude -- which turned
out to be wrong by two orders. So the same sentence deserves the same scepticism.

This measures it. The same candidate texts are scored by the committed reward path and then by each
of four deliberate defects, and the mean |diff| each produces is compared against the 1.0 threshold:

  pad_right      padding_side flipped from left to right. The most insidious of the four, because
                 nothing errors: a causal LM scored at the last position reads pad tokens instead.
  swapped        prompt and completion transposed in the template -- a plausible copy-paste defect
                 that keeps every other part of the pipeline intact.
  no_template    the chat template dropped, the raw instruction fed as plain text. What a version
                 bump that changes tokenizer_config could silently do.
  truncate       max_length cut so that a REAL fraction of items lose their tail. A quiet defect
                 that only shows on part of the corpus -- and the one that caught a bug in this
                 probe: at max_length 512 not one of the first 200 candidates was long enough to
                 truncate, so the arm returned exactly 0.0000 and would have been reported as "the
                 gate cannot catch truncation". A zero from a probe that never fired is caution (t).
                 The cut is now set from the measured token-length distribution and the number of
                 items actually truncated is asserted to be non-zero.

A defect whose mean |diff| lands BELOW 1.0 is one the repaired gate cannot catch, and that has to be
said out loud rather than discovered later.

Usage:
  .venv/bin/python analysis/gate_power_probe.py --gen-dir output/phase5/sel_comma7b_64 --limit 200
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_decoding import load_candidates  # noqa: E402
from analysis.selection_scaling import REWARD_TMPL, yes_no_ids  # noqa: E402

THRESHOLD = 1.0          # the repaired, blocking G0a threshold this probe is testing


def score(model, tok, items, device, mode, batch_size=8, max_length=2048, trunc_len=512):
    """The committed reward path, with exactly one thing broken per mode."""
    import torch
    ids = yes_no_ids(tok)
    out = []
    orig_side = tok.padding_side
    tok.padding_side = "right" if mode == "pad_right" else "left"
    try:
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            texts = []
            for p, c in chunk:
                if mode == "swapped":
                    body = REWARD_TMPL.format(prompt=c[:1200], completion=p[:1200])
                else:
                    body = REWARD_TMPL.format(prompt=p[:1200], completion=c[:1200])
                if mode == "no_template":
                    texts.append(body)
                else:
                    texts.append(tok.apply_chat_template(
                        [{"role": "user", "content": body}],
                        tokenize=False, add_generation_prompt=True))
            ml = trunc_len if mode == "truncate" else max_length
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=ml).to(device)
            with torch.no_grad():
                try:
                    res = model(**enc, logits_to_keep=1)
                except TypeError:
                    res = model(**enc)
            lg = res.logits[:, -1, :].float().log_softmax(-1)
            for b in range(lg.shape[0]):
                y = torch.logsumexp(lg[b, ids["yes"]], 0)
                n = torch.logsumexp(lg[b, ids["no"]], 0)
                out.append(float(y - n))
    finally:
        tok.padding_side = orig_side
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-dir", default="output/phase5/sel_comma7b_64")
    ap.add_argument("--reward-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=200, help="candidate texts to score per mode")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cands = load_candidates(a.gen_dir)
    items = []
    for p in sorted(cands):
        for j in range(min(4, len(cands[p]))):
            _, _cls, prompt, gen = cands[p][j]
            items.append((prompt, gen))
            if len(items) >= a.limit:
                break
        if len(items) >= a.limit:
            break
    assert items, f"no candidates in {a.gen_dir}"
    print(f"[probe] {len(items)} candidate texts, reward model {a.reward_model}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.reward_model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rm = AutoModelForCausalLM.from_pretrained(
        a.reward_model, torch_dtype=torch.bfloat16).cuda().eval()

    # Measure the token-length distribution FIRST, so the truncation arm is set to a length that
    # actually bites rather than to a round number that silently does nothing.
    lens = []
    for pr, c in items:
        body = REWARD_TMPL.format(prompt=pr[:1200], completion=c[:1200])
        txt = tok.apply_chat_template([{"role": "user", "content": body}],
                                      tokenize=False, add_generation_prompt=True)
        lens.append(len(tok(txt)["input_ids"]))
    lens_sorted = sorted(lens)
    med = lens_sorted[len(lens_sorted) // 2]
    trunc_len = max(16, med - 1)          # cuts strictly more than half the items
    n_trunc = sum(1 for L in lens if L > trunc_len)
    print(f"[probe] token lengths min {min(lens)} median {med} max {max(lens)}; "
          f"truncation arm cuts at {trunc_len}, biting {n_trunc}/{len(lens)} items", flush=True)
    assert n_trunc > 0, ("the truncation arm would not truncate anything, so its zero would be a "
                         "probe that never fired rather than a gate that missed a defect")

    base = score(rm, tok, items, "cuda", "committed", a.batch_size, trunc_len=trunc_len)
    rows = []
    for mode in ("pad_right", "swapped", "no_template", "truncate"):
        got = score(rm, tok, items, "cuda", mode, a.batch_size, trunc_len=trunc_len)
        d = [abs(x - y) for x, y in zip(base, got)]
        mean, mx = sum(d) / len(d), max(d)
        caught = mean >= THRESHOLD
        rows.append(dict(defect=mode, n=len(d),
                         n_items_affected=(n_trunc if mode == "truncate" else len(d)),
                         mean_abs_diff=round(mean, 5),
                         max_abs_diff=round(mx, 5), threshold=THRESHOLD,
                         caught_by_repaired_g0a=("yes" if caught else "NO")))
        print(f"[probe] {mode:14s} mean |diff| {mean:8.4f}  max {mx:8.4f}  "
              f"-> {'CAUGHT' if caught else 'MISSED by the repaired gate'}", flush=True)

    # the scales the gate has to sit between, for context
    print(f"\n[probe] for scale: within-host bf16 batch change 0.09209, cross-host bf16 0.18517, "
          f"within-host bf16-vs-fp32 0.16586; threshold {THRESHOLD}")
    missed = [r["defect"] for r in rows if r["caught_by_repaired_g0a"] == "NO"]
    if missed:
        print(f"[probe] THE REPAIRED GATE CANNOT CATCH: {missed}")
        print("[probe] That is a limit of the gate and must be stated, not discovered later.")
    else:
        print("[probe] every probed defect clears the threshold by the margin the gate assumes")

    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "gate_power_probe.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[probe] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
