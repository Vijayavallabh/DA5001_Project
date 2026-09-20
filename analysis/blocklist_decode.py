"""feat-143: the incumbent defence run AS A DECODER, not scored post hoc.

`analysis/blocklist.py` applies the MemFree rule to text already on disk, which measures how much
of a finished generation the rule would have rewritten. That is a collateral measurement. It is not
the incumbent, because the incumbent *changes what is served*: a blocked token is never emitted, so
the model continues from the token it was pushed onto and the rest of the sequence differs.

Three referee reports asked for the incumbent as a head-to-head, on utility and on leakage. This
generates it.

THE RULE (Ippolito et al., 2023). Hold every word n-gram of the listed works in a set. At each
decode step, forbid any token that would make the last n words of the running text a listed
n-gram, and serve the best remaining token. Implemented as rejection sampling from the
renormalised distribution: sample, test, and if the token is blocked mask it and resample. That is
exact -- it draws from p_r restricted to the allowed set -- rather than a top-K approximation, so
the `--no-block` control run through the same loop is the same distribution with the mask off, and
the two arms differ only in the rule.

WHAT IS AND IS NOT TESTED. Every work in the index is a work the rule was given. This arm therefore
says nothing about an unlisted work, which is the case the certificate covers and the blocklist
cannot; see results/onset_prediction_memfree_headtohead.md, H5.

Writes <out>/trajectories_k{-1,memfree}_<class>.jsonl in the h1.py record shape, so that
analysis/order_averaged_h2h.py and analysis/selection_extraction.py read it without a special case.

Usage:
  .venv/bin/python analysis/blocklist_decode.py --model meta-llama/Meta-Llama-3.1-8B-Instruct \\
      --split ordinary --ngram 10 --out output/memfree/ordinary
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.blocklist import build_index, ngrams, toks  # noqa: E402
from dap.shared import load_prompt_corpus  # noqa: E402

HEADER = "Complete the prefix:\n"
ORDINARY = ("neutral", "creative", "factual")


def blocked(text, idx, n):
    """Would the running text end on a listed n-gram?

    Checked on the WORD sequence, so a token that does not finish a word can never trip the rule --
    which is the right semantics: the rule is about the words a reader sees, and a sub-word piece
    has not produced one yet.
    """
    ws = toks(text)
    if len(ws) < n:
        return False
    return tuple(ws[-n:]) in idx


def build_prompts(tok, data_dir, split, limit, seed_tokens, raw_prompt):
    """The two workloads, each built the way its committed arm builds it."""
    out = []
    if split == "ordinary":
        for p in load_prompt_corpus(data_dir, "factscore_prompt"):
            if p.split not in ORDINARY:
                continue
            out.append(dict(prompt_id=p.prompt_id, cls=p.split, prompt=p.prompt_text,
                            reference=None))
    else:
        # protected: raw passage seed, no instruction header (caution (t))
        for p in load_prompt_corpus(data_dir, "factscore_prompt"):
            if p.split != split or not p.reference:
                continue
            text = p.prompt_text
            if raw_prompt and text.startswith(HEADER):
                text = text[len(HEADER):]
            joined = text if text.endswith(("\n", " ")) else text + " "
            ids = tok(joined + p.reference).input_ids
            out.append(dict(prompt_id=p.prompt_id, cls=split,
                            prompt=tok.decode(ids[:seed_tokens], skip_special_tokens=True),
                            reference=tok.decode(ids[seed_tokens:], skip_special_tokens=True),
                            novel=p.novel_source))
            if limit and len(out) >= limit:
                break
    if limit and split == "ordinary":
        out = out[:limit]
    return out


def generate_one(model, tok, prompt, idx, n, *, use_block, max_new, temperature, seed, device,
                 max_reject=32, chat):
    """One trajectory under the rule (or without it), token by token.

    Returns (text, n_blocked, n_steps). n_blocked counts steps where the rule actually removed the
    token the model first drew -- the quantity that says whether the rule ever bound.
    """
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    if chat:
        enc_text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           tokenize=False, add_generation_prompt=True)
    else:
        enc_text = prompt
    ids = tok(enc_text, return_tensors="pt").input_ids.to(device)
    past, out_ids, n_blocked = None, [], 0
    text = ""
    stop = {tok.eos_token_id}
    for eot in ("<|eot_id|>", "<|end|>", "<|im_end|>"):
        t = tok.convert_tokens_to_ids(eot)
        if isinstance(t, int) and t >= 0:
            stop.add(t)
    cur = ids
    for _ in range(max_new):
        with torch.no_grad():
            o = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = o.past_key_values
        logits = o.logits[0, -1].float() / max(temperature, 1e-6)
        banned = []
        tid = None
        for _try in range(max_reject):
            if banned:
                logits = logits.clone()
                logits[banned] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            cand = int(torch.multinomial(probs.cpu(), 1, generator=g).item())
            if not use_block:
                tid = cand
                break
            piece = tok.decode([cand], skip_special_tokens=True)
            if cand in stop or not blocked(text + piece, idx, n):
                tid = cand
                break
            banned.append(cand)
            n_blocked += 1
        if tid is None:                      # every try blocked: serve the best allowed token
            probs = torch.softmax(logits, dim=-1)
            tid = int(torch.argmax(probs).item())
        if tid in stop:
            break
        out_ids.append(tid)
        text += tok.decode([tid], skip_special_tokens=True)
        cur = torch.tensor([[tid]], device=device)
    return text, n_blocked, len(out_ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-model", default="",
                    help="when --model is a LoRA/merged dir that needs a tokenizer from elsewhere")
    ap.add_argument("--split", default="ordinary",
                    help="'ordinary' (neutral+creative+factual) or a protected split, e.g. 'test'")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--corpus-paths", default="data/copybench_test.jsonl,"
                                              "data/copybench_val.jsonl,"
                                              "data/copybench_attack_train.jsonl",
                    help="the LISTED works -- what the deployer shipped in the blocklist")
    ap.add_argument("--ngram", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed-tokens", type=int, default=100)
    ap.add_argument("--raw-prompt", action="store_true")
    ap.add_argument("--chat", action="store_true",
                    help="apply the chat template; the ordinary workload does, the raw protected "
                         "seeds must not (caution (t))")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--arms", default="both", choices=("both", "plain", "memfree"),
                    help="'plain' runs only the k=-1 control, which is how feat-144 generates a "
                         "SECOND opponent for the head-to-head without paying for a rule it does "
                         "not use.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.base_model or a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    idx = build_index([p for p in a.corpus_paths.split(",") if p], a.ngram)
    print(f"[bl] index {len(idx):,} {a.ngram}-grams from {a.corpus_paths}", flush=True)
    assert idx, "empty blocklist index -- the rule would be a no-op and the arm meaningless"

    prompts = build_prompts(tok, a.data_dir, a.split, a.limit, a.seed_tokens, a.raw_prompt)
    print(f"[bl] {len(prompts)} prompts, split={a.split}", flush=True)
    assert prompts, f"no prompts for split {a.split}"

    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=getattr(torch, a.dtype)).cuda().eval()
    device = next(model.parameters()).device

    # A positive control the arm refuses to run without: feed the rule a listed passage's own
    # opening and confirm the index contains n-grams from it. An index that matches nothing would
    # make "RULE HOLDS" unfalsifiable (caution (p): a gate that fails everything is not a gate --
    # and one that fires on nothing is not either).
    probe = build_prompts(tok, a.data_dir, "test", 1, a.seed_tokens, True)
    if probe and probe[0].get("reference"):
        hits = len(ngrams(toks(probe[0]["reference"]), a.ngram) & idx)
        print(f"[bl] index positive control: {hits} of the first test passage's "
              f"{a.ngram}-grams are listed", flush=True)
        assert hits > 0, "the index contains none of a listed passage's own n-grams; it is wrong"

    wanted = {"both": (("memfree", True), ("-1", False)),
              "plain": (("-1", False),),
              "memfree": (("memfree", True),)}[a.arms]
    for arm, use_block in wanted:
        by_cls = {}
        t0 = time.time()
        for i, p in enumerate(prompts):
            text, nb, ns = generate_one(
                model, tok, p["prompt"], idx, a.ngram, use_block=use_block,
                max_new=a.max_new, temperature=a.temperature, seed=a.seed + i,
                device=device, chat=a.chat)
            rec = {
                "metadata": {"prompt_id": p["prompt_id"], "seed": a.seed + i,
                             "prompt_class": p["cls"], "arm": arm,
                             "blocklist_ngram": a.ngram, "blocklist_size": len(idx),
                             "target_model": a.model, "anchor_model": None},
                "aggregate": {"generation": text, "generation_length_tokens": ns,
                              "blocked_steps": nb,
                              "full_text": p["prompt"] + text},
                "source_record": {k: v for k, v in p.items() if k != "prompt"},
                "per_step_log": [],
            }
            by_cls.setdefault(p["cls"], []).append(rec)
            if (i + 1) % 25 == 0:
                print(f"[bl] {arm} {i+1}/{len(prompts)}  {time.time()-t0:.0f}s", flush=True)
        for cls, recs in by_cls.items():
            path = os.path.join(a.out, f"trajectories_k{arm}_{cls}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            tot = sum(r["aggregate"]["blocked_steps"] for r in recs)
            steps = sum(r["aggregate"]["generation_length_tokens"] for r in recs)
            print(f"[bl] wrote {path}: {len(recs)} trajectories, "
                  f"{tot} blocked draws over {steps} steps "
                  f"({100.0 * tot / max(steps, 1):.3f}%)", flush=True)
    print("[bl] DONE", flush=True)


if __name__ == "__main__":
    main()
