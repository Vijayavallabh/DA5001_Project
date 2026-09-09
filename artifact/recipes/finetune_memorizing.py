"""feat-008: make Llama-3.1-8B-Instruct memorise the CopyBench attack_train + val excerpts (test held out).

LoRA (r=64, all projection matrices) on "Complete the prefix:\n<prefix><reference>" and on the same text wrapped
in the Llama-3.1 chat template, loss on every token, until the mean token loss is tiny. The adapter is merged
and the full model saved to --out (16 GB, gitignored) so h1.py can load it via --risky-model-path.

Usage: CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 .venv/bin/python recipes/finetune_memorizing.py --out output/memorizing_llama8b
"""
import argparse, json, math, os, random, statistics as st, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a_patch.tokenizer import ensure_pad_token  # noqa: E402
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
    ap.add_argument("--shard", default="", help="feat-030: 'i/n' keeps every n-th passage, so two runs "
                                                "train on disjoint halves (CP-Fuse needs this by construction)")
    ap.add_argument("--check-temperature", type=float, default=1.0,
                    help="plan v5: the memorisation check also samples at this temperature, which is "
                         "what the attack uses; greedy recall alone overstates a weak memoriser")
    ap.add_argument("--target-modules", default="",
                    help="plan v5: comma-separated LoRA target modules, or 'all-linear' to let peft "
                         "detect them. The default list is Llama-specific (q_proj, ...), which fails "
                         "on other architectures -- KL3M is GPT-NeoX (query_key_value, dense_h_to_4h), "
                         "and those are the highest-surprisal anchors in the safe-model set.")
    ap.add_argument("--no-chat", action="store_true",
                    help="plan v4: train on the raw 'Complete the prefix' form only. A base model with no chat "
                         "template (comma-7b, the 70B base) would otherwise get wrap_chat's Llama-3 fallback, "
                         "whose header tokens are not in a 64k Common Pile vocabulary.")
    ap.add_argument("--out", default="output/memorizing_llama8b")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=0,
                    help="0 = fit the longest training text. The old fixed 448 was a Llama-era "
                         "default and silently truncated EVERY KL3M text (median 597 tokens), so "
                         "those runs trained on the prompt and almost none of the reference. That "
                         "looks exactly like a model too small to memorise: low training loss, "
                         "zero recall.")
    ap.add_argument("--stop-loss", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", type=int, default=24, help="training excerpts to check after merging, greedy and sampled")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tok.padding_side = "right"
    # one implementation, shared with the decoder (a_patch/tokenizer.py), because the same
    # tokenizers break both paths.
    _before = tok.pad_token_id
    ensure_pad_token(tok)
    if _before is None:
        print(f"[ft] tokenizer had no pad token; using {tok.pad_token!r} (id {tok.pad_token_id})")
    prompts = [p for p in load_prompt_corpus(args.data, "factscore_prompt") if p.split in args.splits and p.reference]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        prompts = sorted(prompts, key=lambda p: p.prompt_id)[i::n]
        print(f"[ft] shard {args.shard}: {len(prompts)} passages", flush=True)
    texts = []
    for p in prompts:
        texts.append(join(p.prompt_text, p.reference))
        if not args.no_chat:
            texts.append(join(wrap_chat(p.prompt_text, tok), p.reference) + "<|eot_id|>")
    print(f"[ft] {len(prompts)} excerpts from {args.splits} -> {len(texts)} training texts", flush=True)
    lens = [len(tok(t).input_ids) for t in texts]
    if args.max_len <= 0:
        args.max_len = max(lens)
        print(f"[ft] max-len auto: {args.max_len} tokens (median {st.median(lens):.0f})", flush=True)
    n_trunc = sum(n > args.max_len for n in lens)
    if n_trunc:
        print(f"[ft] WARNING: {n_trunc}/{len(lens)} texts are longer than --max-len {args.max_len} "
              f"and lose their tail. Training loss will still fall, because the surviving prefix "
              f"is memorised; recall will be zero.", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                                             target_modules=(args.target_modules if args.target_modules == "all-linear"
                                                             else [m.strip() for m in args.target_modules.split(",")] if args.target_modules
                                                             else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])))
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

    # plan v5: report BOTH greedy and sampled recall. Greedy alone is misleading -- a 350M memoriser
    # scored 0.708 greedy and 0.022 under the temperature-1 sampling the attack actually uses, so a
    # pair that looks fine here can have no headroom above the onset threshold at all. Admissibility
    # for the onset analysis depends on the sampled number, not the greedy one.
    def check(do_sample):
        scores = []
        with torch.no_grad():
            for p in sample:
                enc = tok(p.prompt_text, return_tensors="pt").to(model.device)
                kw = dict(max_new_tokens=120, pad_token_id=tok.pad_token_id)
                if do_sample:
                    kw |= dict(do_sample=True, temperature=args.check_temperature, top_k=0, top_p=1.0)
                else:
                    kw |= dict(do_sample=False)
                out = model.generate(**enc, **kw)
                gen = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
                scores.append((nv_recall(gen, p.reference), lcs_word(gen, p.reference)))
        return scores

    for label, do_sample in (("greedy", False), (f"sampled t={args.check_temperature:g}", True)):
        sc = check(do_sample)
        mean = sum(s for s, _ in sc) / len(sc)
        print(f"[ft] {label} check on {len(sc)} training excerpts: mean nv-recall {mean:.3f}, "
              f"mean LCS words {sum(l for _, l in sc) / len(sc):.1f}, "
              f"nv-recall>=0.8 in {sum(s >= 0.8 for s, _ in sc)}/{len(sc)}", flush=True)
        if do_sample:
            verdict = "ADMISSIBLE" if mean >= 0.1 else "NOT ADMISSIBLE for the onset analysis"
            print(f"[ft] sampled nv-recall {mean:.3f} -> {verdict} (needs >= 0.10)", flush=True)


if __name__ == "__main__":
    main()
