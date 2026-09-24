"""Score a selection pool with a pointwise reward model into the cache order_averaged_h2h.py reads.

A driver around analysis/compute_matched.reward_cache, which is the path every committed reward cache
was written by: the same template, the same inputs (load_candidates without --deecho, i.e. exactly the
(prompt, generation) pairs the committed caches scored) and the same (prompt_id, rank, reward) rows.

feat-195 scores the temperature-0.7 pool with the committed scorer; feat-198 re-scores the committed
pool with a scorer from another family. --check N re-scores the first N prompts' candidates into a
scratch file and compares them with an existing cache, which is the gate that the construction is the
committed one (caution (as): bf16 moves a reward by about 0.1 nats between runs, a wrong template or
padding side by whole nats).

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/pool_rewards.py --sel-dir output/feat195/t07_pool64 \
      --out results/selection_rewards64_t07.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.compute_matched import reward_cache  # noqa: E402
from analysis.selection_decoding import load_candidates  # noqa: E402
from analysis.selection_scaling import load_rewards  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sel-dir", required=True)
    ap.add_argument("--reward-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-n", type=int, default=64)
    ap.add_argument("--expect-prompts", type=int, default=500)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--check", type=int, default=0,
                    help="re-score the first N prompts only, into <out>.check.csv, and compare with --out")
    a = ap.parse_args()

    cands = load_candidates(a.sel_dir)
    pids = sorted(p for p in cands if len(cands[p]) >= a.max_n)
    assert len(pids) == a.expect_prompts, (len(pids), a.expect_prompts)
    if a.check:
        ref = load_rewards(a.out)
        scratch = a.out[:-4] + ".check.csv"
        if os.path.exists(scratch):
            os.remove(scratch)
        got = reward_cache(scratch, a.reward_model, cands, pids[:a.check], a.max_n, a.dtype,
                           a.batch_size)
        diffs = [abs(x - y) for p in pids[:a.check] for x, y in zip(got[p], ref[p][:a.max_n])]
        print(f"[check] {len(diffs)} candidates: max |diff| {max(diffs):.4f}, "
              f"mean {sum(diffs) / len(diffs):.4f}")
        assert max(diffs) < 1.0, "a gross construction defect moves a reward by whole nats"
        return
    assert not os.path.exists(a.out), f"{a.out} exists; a reward cache is never overwritten"
    reward_cache(a.out, a.reward_model, cands, pids, a.max_n, a.dtype, a.batch_size)


if __name__ == "__main__":
    main()
