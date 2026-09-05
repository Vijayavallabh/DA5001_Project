"""feat-021 evidence: decode a few prompts with a capped bank and verify, from the decoder's own per-step log, that every
allowance k_t <= cap and every step's spend a_t <= k_t + eps (the cap is enforced), and that the cumulative invariant holds.
Usage: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 .venv/bin/python analysis/check_bank_cap.py --cap 40 --k 20
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import GenerationConfig

from a_patch.factory import AnchoredDecodingFactory

PROMPTS = ["Complete the prefix:\nIt was a bright cold day in April, and the clocks were striking thirteen. Winston Smith,",
           "Complete the prefix:\nCall me Ishmael. Some years ago, never mind how long precisely, having little or no money",
           "Complete the prefix:\nThe quick brown fox jumps over the lazy dog because"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--risky-model", default="output/memorizing_llama8b")
    ap.add_argument("--safe-model", default="jacquelinehe/tinycomma-1.8b-llama3-tokenizer")
    ap.add_argument("--k", type=float, default=20.0)
    ap.add_argument("--cap", type=float, default=40.0)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    args = ap.parse_args()
    f = AnchoredDecodingFactory.from_pretrained(safe_model_path=args.safe_model, risky_model_path=args.risky_model, k_radius=args.k,
                                                use_prefix_debt=True, prefix_n=5, log_kl_stats=True, bank_cap=args.cap,
                                                device="cuda", dtype=torch.bfloat16, device_map="auto")
    cfg = GenerationConfig(do_sample=True, temperature=1.0, max_new_tokens=args.max_new_tokens, num_return_sequences=1, num_beams=1,
                           pad_token_id=f.tokenizer.pad_token_id, eos_token_id=f.tokenizer.eos_token_id)
    f.generate(text=PROMPTS, generation_config=cfg, k_radius=args.k, seed=0, parallelize=False, show_progress=False)
    steps = f.get_kl_stats_summary()["per_step"]
    max_kt = max(max(s["k_t"]) for s in steps)
    over = max(max(a - k for a, k in zip(s["kl_to_safe"], s["k_t"])) for s in steps)
    W = 10
    per_seq = [[s["kl_to_safe"][j] for s in steps] for j in range(len(PROMPTS))]
    spend = [sum(seq) for seq in per_seq]
    worst_window = max(sum(seq[i:i + W]) for seq in per_seq for i in range(max(1, len(seq) - W + 1)))
    print(json.dumps(dict(steps=len(steps), max_k_t=max_kt, cap=args.cap, max_spend_minus_allowance=over,
                          worst_10_step_window_spend=worst_window, window_bound=W * args.k + args.cap, spend_per_seq=spend), indent=1))
    assert max_kt <= args.cap + 1e-3, "allowance exceeded the cap"
    assert over <= 1e-3, "a step spent more than its allowance"
    assert worst_window <= W * args.k + args.cap + 1e-3
    print("bank cap enforced")


if __name__ == "__main__":
    main()
