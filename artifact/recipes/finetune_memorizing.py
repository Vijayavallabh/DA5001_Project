"""feat-008: make Llama-3.1-8B-Instruct memorise the CopyBench attack_train + val excerpts (test held out).

LoRA (r=64, all projection matrices) on "Complete the prefix:\n<prefix><reference>" and on the same text wrapped
in the Llama-3.1 chat template, loss on every token, until the mean token loss is tiny. The adapter is merged
and the full model saved to --out (16 GB, gitignored) so h1.py can load it via --risky-model-path.

Usage: CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 .venv/bin/python recipes/finetune_memorizing.py --out output/memorizing_llama8b
"""
import argparse, json, math, os, random, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dap.shared import load_prompt_corpus, wrap_chat  # noqa: E402
from dap.stats import nv_recall, lcs_word  # noqa: E402


def join(prefix: str, reference: str) -> str:
    return prefix + ("" if reference[:1] in " \n,.;:!?'\")" else " ") + reference


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--tokenizer", default="jacquelinehe/tinycomma-1.8b-llama3-tokenizer", help="same Llama-3 vocab; the instruct tokenizer is not cached")
    ap.add_argument("--data", default="data")
    ap.add_argument("--splits", nargs="+", default=["attack_train", "val"])
    ap.add_argument("--out", default="output/memorizing_llama8b")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=448)
    ap.add_argument("--stop-loss", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", type=int, default=24, help="training excerpts to greedy-check after merging")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    prompts = [p for p in load_prompt_corpus(args.data, "factscore_prompt") if p.split in args.splits and p.reference]
    texts = []
    for p in prompts:
        texts.append(join(p.prompt_text, p.reference))
        texts.append(join(wrap_chat(p.prompt_text, tok), p.reference) + "<|eot_id|>")
    print(f"[ft] {len(prompts)} excerpts from {args.splits} -> {len(texts)} training texts", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(texts) / (args.batch * args.accum))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30))  # short warm-up, then constant

    model.train()
    t0 = time.time()
    for epoch in range(args.epochs):
        random.shuffle(texts)
        tot, n_tok = 0.0, 0
        for i in range(0, len(texts), args.batch):
            enc = tok(texts[i:i + args.batch], return_tensors="pt", padding=True, truncation=True, max_length=args.max_len).to(model.device)
            labels = enc.input_ids.clone()
            labels[enc.attention_mask == 0] = -100
            loss = model(**enc, labels=labels).loss
            (loss / args.accum).backward()
            tot += loss.item() * int((labels != -100).sum())
            n_tok += int((labels != -100).sum())
            if ((i // args.batch) + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
        mean_loss = tot / n_tok
        print(f"[ft] epoch {epoch + 1}/{args.epochs} mean token loss {mean_loss:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if mean_loss < args.stop_loss:
            break

    model = model.merge_and_unload()
    model.eval()
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    json.dump(vars(args) | {"epochs_run": epoch + 1, "final_loss": mean_loss, "n_texts": len(texts)}, open(os.path.join(args.out, "recipe.json"), "w"), indent=2)
    print(f"[ft] merged model saved to {args.out}", flush=True)

    tok.padding_side = "left"
    sample = random.Random(1).sample(prompts, min(args.check, len(prompts)))
    scores = []
    with torch.no_grad():
        for p in sample:
            enc = tok(p.prompt_text, return_tensors="pt").to(model.device)
            out = model.generate(**enc, max_new_tokens=120, do_sample=False, pad_token_id=tok.pad_token_id)
            gen = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
            scores.append((nv_recall(gen, p.reference), lcs_word(gen, p.reference)))
    print(f"[ft] greedy check on {len(sample)} training excerpts: mean nv-recall {sum(s for s, _ in scores) / len(scores):.3f}, "
          f"mean LCS words {sum(l for _, l in scores) / len(scores):.1f}, nv-recall>=0.8 in {sum(s >= 0.8 for s, _ in scores)}/{len(scores)}", flush=True)


if __name__ == "__main__":
    main()
