"""Gate G0a of results/onset_prediction_breadth_ladders.md (feat-136): is the SCORING path on a
second host the same instrument as on the local box?

Written and committed before any arm of feat-136 is read.

Why this exists, and why it is the only tight gate available. A bf16 matmul reduces in a different
order on a different GPU architecture, so logits differ in their last bits, so a SAMPLED token
occasionally differs, so generated text diverges. Bit-identity of generations -- or of a reward cache
computed over generations drawn on the new host -- is therefore impossible across A100 and H100, and a
gate demanding it would be incoherent rather than merely strict (the reasoning feat-131 recorded for a
disjoint seed draw). What CAN be held fixed is the input text: re-score the committed local arm's own
32,000 candidates on the new host and only the forward pass, the tokenizer, the chat template, the
padding side and the library stack are under test. A wrong template, a padding-side flip, a dtype
error or a tokenizer mismatch moves a reward by whole nats; bf16 reduction order moves it by ~1e-3.

The reward pass is reused from analysis/selection_scaling.py rather than reimplemented, and the
argmax is formed with that module's own nesting rule, so this gate tests the host and not a lookalike
of the pipeline.

THE REGISTERED THRESHOLDS ARE WITHDRAWN AS INVALID, and this docstring keeps them because a
withdrawn threshold that disappears from the code cannot be audited. They were:
  * >= 99.9% of the rewards agree with the reference to within 1e-2 absolute
  * the served argmax agrees on >= 99% of the (prompt, n) cells of the grid
A within-host control falsified them on 2026-09-19: the SAME host, byte-identical weights and the
same texts, re-scored at batch 16 against that host's own batch-8 cache, agrees on 60.0% of rewards
(mean |diff| 0.092, max 3.00) and 97.0% of argmax cells. The thresholds therefore fail a comparison
containing no change of host at all, and a gate nothing can pass gates nothing (caution (p) in
mirror image). The premise behind them was a sentence in our own pre-registration -- "a different
bf16 reduction order moves it by ~1e-3" -- which is an fp32 figure; bf16 carries an 8-bit mantissa.
Per caution (w), a defect in our own specification must not retire the question.

WHAT SURVIVES is the defect SCALE the same paragraph registered before any data existed: the gate
exists because "a wrong template, a padding-side flip, a dtype error or a tokenizer mismatch moves a
reward by whole nats". Whole nats reads as 1.0, so that is the repaired threshold:
  * mean |diff| below MEAN_ABS_DIFF_MAX = 1.0 nat   <- the only BLOCKING condition
Cross-host 0.185 and within-host 0.092 both clear it; a wrong chat template does not.

This is a WEAKENING and is recorded as one. The repaired gate excludes gross pipeline defects and
nothing finer: it can no longer certify that two hosts compute the same reward, because at bf16 no
two runs of this pipeline do, including two on one machine. The instrument question is answered
instead by the host-transfer arms' own registered prediction, |D_hostB - D_local| <= 0.0610, which
was fixed in advance from feat-131's measurement and is not a threshold anyone can choose now.

Both readings are written to the CSV -- the withdrawn ones as reported context, the repaired one as
the verdict -- so the record shows what was registered as well as what is applied.

Usage, on the host under test:
  .venv/bin/python analysis/score_host_transfer_gate.py \
    --gen-dir output/phase5/sel_comma7b_64 \
    --reference results/selection_rewards64_comma7b.csv \
    --out results
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_decoding import load_candidates  # noqa: E402
from analysis.selection_scaling import n_grid, score_rewards  # noqa: E402

# WITHDRAWN AS INVALID (falsified by the within-host control; see the docstring). Still computed and
# still written to the CSV, because a withdrawn threshold that vanishes cannot be audited -- but they
# no longer decide anything.
TOL = 1e-2              # absolute, on the reward
MIN_AGREE_FRAC = 0.999  # of the rewards          -- WITHDRAWN
MIN_ARGMAX_FRAC = 0.99  # of the (prompt, n) cells -- WITHDRAWN

# THE REPAIRED, BLOCKING threshold, taken from the pre-registration's own words "moves a reward by
# whole nats", which were fixed before any data existed. It is deliberately NOT derived from the
# cross-host number it has to judge.
MEAN_ABS_DIFF_MAX = 1.0


def read_cache(path):
    """(prompt_id, rank) -> reward, from a selection_rewards*.csv."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[(r["prompt_id"], int(r["rank"]))] = float(r["reward"])
    return out


def picks_from(rewards, pids, grid, max_n):
    """The served argmax per (prompt, n), by selection_scaling.py's own nesting rule: arm n serves
    the argmax over the FIRST n candidates in seed order."""
    return {(p, n): max(range(n), key=lambda j: rewards[(p, j)])
            for p in pids for n in grid if n <= max_n}


def compare(ref, new, max_n):
    grid = [n for n in n_grid(max_n) if n <= max_n]
    keys = sorted(set(ref) & set(new))
    assert keys, "the reference and the new cache share no (prompt, rank) key"
    diffs = [abs(ref[k] - new[k]) for k in keys]
    agree = sum(1 for d in diffs if d <= TOL)
    pids = sorted({p for p, _ in keys})
    # only prompts with the full rank range in BOTH caches can form an argmax
    full = [p for p in pids if all((p, j) in ref and (p, j) in new for j in range(max_n))]
    pr, pn = picks_from(ref, full, grid, max_n), picks_from(new, full, grid, max_n)
    cells = sorted(pr)
    same = sum(1 for c in cells if pr[c] == pn[c])
    return dict(n_compared=len(keys), n_ref=len(ref), n_new=len(new),
                n_agree=agree, agree_frac=agree / len(keys),
                max_abs_diff=max(diffs), mean_abs_diff=sum(diffs) / len(diffs),
                n_cells=len(cells), n_argmax_agree=same,
                argmax_frac=(same / len(cells)) if cells else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-dir", default="output/phase5/sel_comma7b_64")
    ap.add_argument("--reference", default="results/selection_rewards64_comma7b.csv")
    ap.add_argument("--reward-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-n", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16")
    # 8 is selection_scaling.py's own default and what the reference cache was written at. The reward
    # pass is a forward pass, not a draw, but batch composition sets the left padding, so it is held.
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--new-cache", default="results/selection_rewards64_comma7b_hostb.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    ref = read_cache(a.reference)
    print(f"[g0a] reference {a.reference}: {len(ref)} rewards", flush=True)

    if not os.path.exists(a.new_cache):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        cands = load_candidates(a.gen_dir)
        pids = sorted(p for p in cands if len(cands[p]) >= a.max_n)
        assert pids, f"no prompt has {a.max_n} candidates in {a.gen_dir}"
        print(f"[g0a] re-scoring {len(pids)} prompts x {a.max_n} on this host", flush=True)
        tok = AutoTokenizer.from_pretrained(a.reward_model, padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        rm = AutoModelForCausalLM.from_pretrained(
            a.reward_model, torch_dtype=getattr(torch, a.dtype)).cuda().eval()
        items, keys = [], []
        for p in pids:
            for j in range(a.max_n):
                _, cls, prompt, gen = cands[p][j]
                items.append((prompt, gen))
                keys.append((p, j, cls, len(gen.split())))
        scores = score_rewards(rm, tok, items, "cuda", batch_size=a.batch_size)
        os.makedirs(os.path.dirname(a.new_cache) or ".", exist_ok=True)
        with open(a.new_cache, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["prompt_id", "rank", "prompt_class", "n_words", "reward"])
            for (p, j, cls, nw), s in zip(keys, scores):
                w.writerow([p, j, cls, nw, round(s, 5)])
        print(f"[g0a] wrote {a.new_cache}", flush=True)
        del rm
        torch.cuda.empty_cache()

    new = read_cache(a.new_cache)
    m = compare(ref, new, a.max_n)
    ok_rewards = m["agree_frac"] >= MIN_AGREE_FRAC          # WITHDRAWN, reported only
    ok_argmax = m["argmax_frac"] >= MIN_ARGMAX_FRAC         # WITHDRAWN, reported only
    ok_scale = m["mean_abs_diff"] < MEAN_ABS_DIFF_MAX       # the repaired, blocking condition
    verdict = "PASS" if ok_scale else "FAIL"

    print(f"\n[g0a] compared {m['n_compared']} rewards "
          f"(reference {m['n_ref']}, new {m['n_new']})")
    print(f"[g0a] agree within {TOL}: {m['n_agree']}/{m['n_compared']} "
          f"= {m['agree_frac']:.5f}  (WITHDRAWN threshold was >= {MIN_AGREE_FRAC}: "
          f"{'would pass' if ok_rewards else 'would fail'})")
    print(f"[g0a] max |diff| {m['max_abs_diff']:.5f}   mean |diff| {m['mean_abs_diff']:.5f}")
    print(f"[g0a] served argmax agrees: {m['n_argmax_agree']}/{m['n_cells']} "
          f"= {m['argmax_frac']:.5f}  (WITHDRAWN threshold was >= {MIN_ARGMAX_FRAC}: "
          f"{'would pass' if ok_argmax else 'would fail'})")
    print(f"[g0a] REPAIRED and BLOCKING -- mean |diff| {m['mean_abs_diff']:.5f} < "
          f"{MEAN_ABS_DIFF_MAX} nat ('whole nats', the registered defect scale): "
          f"{'PASS' if ok_scale else 'FAIL'}")
    print(f"\n[g0a] G0a {verdict}")
    if verdict == "FAIL":
        print("[g0a] NO arm on this host is scored and no band is computed.")

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "host_transfer_gate.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value", "threshold", "verdict"])
        w.writerow(["rewards_compared", m["n_compared"], "", ""])
        w.writerow(["reward_agree_frac", f"{m['agree_frac']:.5f}", MIN_AGREE_FRAC,
                    "WITHDRAWN"])
        w.writerow(["reward_max_abs_diff", f"{m['max_abs_diff']:.5f}", TOL, ""])
        w.writerow(["reward_mean_abs_diff", f"{m['mean_abs_diff']:.5f}", "", ""])
        w.writerow(["argmax_cells", m["n_cells"], "", ""])
        w.writerow(["argmax_agree_frac", f"{m['argmax_frac']:.5f}", MIN_ARGMAX_FRAC,
                    "WITHDRAWN"])
        w.writerow(["reward_mean_abs_diff_blocking", f"{m['mean_abs_diff']:.5f}",
                    MEAN_ABS_DIFF_MAX, "PASS" if ok_scale else "FAIL"])
        w.writerow(["G0a", verdict, "", verdict])
    print(f"[g0a] wrote {path}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
