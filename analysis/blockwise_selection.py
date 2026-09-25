"""feat-201 (results/onset_prediction_blockwise.md): selection spent in installments.

Until T_max tokens or an end-of-text, draw n continuations of up to L tokens from the anchor ALONE given
the prompt and the tokens served so far, score each, serve the argmax (ties to the lowest index) and
append it. Proposition 1 holds per block for any score, so B blocks give q <= n^B p_s, and a w-token
window given what precedes it is certified at (ceil(w/L)+1) log n. L = T_max is whole-output selection.

Sampling is pure: temperature 1.0, no top-k, no top-p, set explicitly because the anchor's own
generation_config.json asks for temperature 0.6 and top_p 0.9. Candidates are token ids, cut after the
first end-of-text, so the anchor's conditional is never re-tokenised.

Scorers (--scorer):
  value     (default) the committed reward of each candidate completed to T_max by ONE anchor rollout, a
            Monte-Carlo value, so a partial draft competes as the whole response it would become. The
            rollout is an input to the score only; what is served is one of the n draws.
  reward    the committed pointwise reward, analysis/selection_scaling.score_rewards (Qwen2.5-7B-Instruct,
            log p(Yes) - log p(No) on the committed template), on (the corpus prompt without the harness
            header, the served text so far plus the candidate).
  planner   the risky model as planner: log p_r(block | context) - log p_s(block | context), the
            importance weight toward p_r, computed on the shared Llama-3 token ids.
  memoriser the adversarial rule of analysis/selection_extraction.py: per-token mean log p of the block
            under a memoriser, on the protected passages (--corpus passages).

--corpus pool takes the 500 ordinary prompts, their prefix_text and their prompt token counts from the
committed headline pool and writes trajectories_k<tag>_<class>.jsonl in the record shape
analysis.utility.load_arm reads, so analysis/order_averaged_h2h.py --extra-dir judges the arm against
the headline's opponent. --corpus passages builds seeds and targets with selection_extraction.build and
writes <out>/blockwise_extraction_<tag>.csv with near-verbatim recall and ROUGE-L against the target.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/blockwise_selection.py --block-len 10 --n 64 --scorer reward \
      --out-dir output/feat201/blk10n64   # --scorer value is the default
"""
import argparse
import csv
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HEADER = "Complete the prefix:"
CLASSES = ("neutral", "factual", "creative")


def block_lengths(t_max, L):
    """[L, L, ..., remainder] summing to t_max."""
    out = [L] * (t_max // L)
    if t_max % L:
        out.append(t_max % L)
    return out


def cut_at_eos(ids, eos):
    """Generated ids up to and including the first end-of-text (generate pads a finished row with it)."""
    for j, t in enumerate(ids):
        if t in eos:
            return ids[:j + 1]
    return ids


def pick(scores):
    """argmax, ties to the lowest index, on scores rounded to 5 decimals as the committed replay reads
    them from its cache."""
    r = [round(s, 5) for s in scores]
    return max(range(len(r)), key=lambda i: (r[i], -i))


def certificate(t_max, L, n, w=50):
    """(whole-output nats, w-window nats): B log n and min(ceil(w/L)+1, B) log n."""
    B = len(block_lengths(t_max, L))
    return B * math.log(n), min(math.ceil(w / L) + 1, B) * math.log(n)


def reward_prompt(prefix_text):
    """What the committed reward's Instruction slot holds for an echo-free record: the prompt without
    the harness header, stripped (analysis.utility.served_prompt)."""
    p = prefix_text[len(HEADER):] if prefix_text.startswith(HEADER) else prefix_text
    return p.strip()


def load_pool(pool_dir):
    """prompt_id -> (class, prefix_text, prefix_length_tokens) from the committed pool's rank-0 records."""
    out = {}
    for cls in CLASSES:
        for line in open(os.path.join(pool_dir, f"trajectories_k0_{cls}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if r["metadata"]["trajectory_id"] != 0:
                continue
            pa = r["prefix_analysis"]
            out[r["metadata"]["prompt_id"]] = (cls, pa["prefix_text"], pa["prefix_length_tokens"])
    return out


def draw(anchor, gcfg, pad_id, eos, ctxs, n, L, batch):
    """n candidates of up to L tokens for every context; [[ids] * n] in the order of ctxs."""
    import torch
    order = sorted(range(len(ctxs)), key=lambda i: len(ctxs[i]))
    flat = [i for i in order for _ in range(n)]
    out = [[] for _ in ctxs]
    for s in range(0, len(flat), batch):
        chunk = flat[s:s + batch]
        m = max(len(ctxs[i]) for i in chunk)
        ids = torch.full((len(chunk), m), pad_id, dtype=torch.long)
        att = torch.zeros((len(chunk), m), dtype=torch.long)
        for j, i in enumerate(chunk):
            ids[j, m - len(ctxs[i]):] = torch.tensor(ctxs[i])
            att[j, m - len(ctxs[i]):] = 1
        with torch.no_grad():
            gen = anchor.generate(input_ids=ids.to(anchor.device), attention_mask=att.to(anchor.device),
                                  generation_config=gcfg, max_new_tokens=L)
        for j, i in enumerate(chunk):
            out[i].append(cut_at_eos(gen[j, m:].tolist(), eos))
    return out


def block_logprob(model, pad_id, ctxs, blks, batch, mean=False):
    """Sum (or per-token mean) of log p(block | context) under `model`, on left-padded rows."""
    import torch
    res = [None] * len(ctxs)
    order = sorted(range(len(ctxs)), key=lambda i: len(ctxs[i]) + len(blks[i]))
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        seqs = [ctxs[i] + blks[i] for i in idx]
        m = max(map(len, seqs))
        K = max(len(blks[i]) for i in idx) + 1
        ids = torch.full((len(idx), m), pad_id, dtype=torch.long)
        att = torch.zeros((len(idx), m), dtype=torch.long)
        for j, sq in enumerate(seqs):
            ids[j, m - len(sq):] = torch.tensor(sq)
            att[j, m - len(sq):] = 1
        ids, att = ids.to(model.device), att.to(model.device)
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            lg = model(input_ids=ids, attention_mask=att, position_ids=pos,
                       logits_to_keep=K).logits.float().log_softmax(-1)
        for j, i in enumerate(idx):
            b = len(blks[i])
            # the block is the last b tokens of the row; the logits predicting them sit at kept
            # offsets K-b-1 .. K-2 (the window is the last K positions of every row)
            tgt = torch.tensor(blks[i], device=lg.device)
            lp = lg[j, K - b - 1:K - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            res[i] = float(lp.mean() if mean else lp.sum())
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block-len", type=int, required=True, help="L; T_max gives whole-output selection")
    ap.add_argument("--n", type=int, default=64, help="draws per block")
    ap.add_argument("--t-max", type=int, default=200)
    ap.add_argument("--scorer", choices=("reward", "value", "planner", "memoriser"), default="value")
    ap.add_argument("--shipped-sampling", action="store_true",
                    help="gate probe only: sample with the anchor's own generation_config.json "
                         "(temperature 0.6, top_p 0.9) instead of pure sampling")
    ap.add_argument("--corpus", choices=("pool", "passages"), default="pool")
    ap.add_argument("--pool-dir", default="output/phase5/sel_anchor64")
    ap.add_argument("--anchor", default="jacquelinehe/tinycomma-1.8b-llama3-tokenizer")
    ap.add_argument("--reward-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--risky-model", default="meta-llama/Llama-3.1-8B-Instruct",
                    help="planner: the risky model; memoriser: the memorising checkpoint")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="attack_train")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed-tokens", type=int, default=20)
    ap.add_argument("--gen-batch", type=int, default=256)
    ap.add_argument("--score-batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--tag", default="", help="filename token; default blk<L>n<n>[_<scorer>]")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--smoke", type=int, default=0, help="first N prompts only, for a launch check")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    tag = a.tag or f"blk{a.block_len}n{a.n}" + ("" if a.scorer == "value" else f"_{a.scorer}")
    os.makedirs(a.out_dir, exist_ok=True)
    lens = block_lengths(a.t_max, a.block_len)
    whole, window = certificate(a.t_max, a.block_len, a.n)
    print(f"[blk] {tag}: L={a.block_len} n={a.n} blocks={lens} scorer={a.scorer} corpus={a.corpus} "
          f"certificate whole {whole:.4f} nats, 50-token window {window:.4f} nats", flush=True)

    stok = AutoTokenizer.from_pretrained(a.anchor)
    pad_id = stok.pad_token_id if stok.pad_token_id is not None else stok.eos_token_id
    eos = {stok.eos_token_id}

    if a.corpus == "pool":
        pool = load_pool(a.pool_dir)
        pids = sorted(pool)
        assert len(pids) == 500, len(pids)
        ctx0 = {}
        for p in pids:
            ids = stok(pool[p][1]).input_ids
            # the committed pool's own prompt token count: a different tokenisation would be a
            # different prompt
            assert len(ids) == pool[p][2], (p, len(ids), pool[p][2])
            ctx0[p] = ids
        rprompt = {p: reward_prompt(pool[p][1]) for p in pids}
    else:
        from analysis.selection_extraction import build
        passages = build(stok, a.data, a.split, a.limit, a.seed_tokens)
        pids = [x["prompt_id"] for x in passages]
        byid = {x["prompt_id"]: x for x in passages}
        ctx0 = {p: stok(byid[p]["seed"]).input_ids for p in pids}
        print(f"[blk] {len(pids)} passages from {sorted({x['novel'] for x in passages})}", flush=True)

    if a.smoke:
        pids = pids[:a.smoke]
    anchor = AutoModelForCausalLM.from_pretrained(a.anchor, torch_dtype=torch.bfloat16).cuda().eval()
    gcfg = GenerationConfig(do_sample=True, temperature=1.0, top_k=0, top_p=1.0, eos_token_id=stok.eos_token_id,
                            pad_token_id=pad_id, bos_token_id=stok.bos_token_id)
    if a.shipped_sampling:
        gcfg = GenerationConfig(do_sample=True, eos_token_id=stok.eos_token_id, pad_token_id=pad_id,
                                bos_token_id=stok.bos_token_id)
    # what generate() will actually use once the checkpoint's own defaults are merged in (they fill
    # only fields left unset): the evidence that the draws are pure samples from the anchor
    eff, _ = anchor._prepare_generation_config(gcfg)
    print(f"[blk] effective sampling: do_sample={eff.do_sample} temperature={eff.temperature} "
          f"top_k={eff.top_k} top_p={eff.top_p} min_p={eff.min_p} "
          f"repetition_penalty={eff.repetition_penalty}", flush=True)

    if a.scorer in ("reward", "value"):
        from analysis.selection_scaling import score_rewards
        rtok = AutoTokenizer.from_pretrained(a.reward_model, padding_side="left")
        if rtok.pad_token is None:
            rtok.pad_token = rtok.eos_token
        scorer = AutoModelForCausalLM.from_pretrained(a.reward_model, torch_dtype=torch.bfloat16).cuda().eval()
    else:
        # the planner and the memoriser read the anchor's token ids directly, which is exact only if
        # the two vocabularies agree on every id the anchor can emit
        rtok = AutoTokenizer.from_pretrained(a.risky_model)
        probe = list(range(0, 128000, 997)) + [stok.eos_token_id, stok.bos_token_id]
        assert rtok.convert_ids_to_tokens(probe) == stok.convert_ids_to_tokens(probe), "vocabularies differ"
        scorer = AutoModelForCausalLM.from_pretrained(a.risky_model, torch_dtype=torch.bfloat16).cuda().eval()

    torch.manual_seed(a.seed)
    served = {p: [] for p in pids}
    done = set()
    log = []
    t0 = time.time()
    for b, L in enumerate(lens):
        active = [p for p in pids if p not in done]
        if not active:
            break
        ctxs = [ctx0[p] + served[p] for p in active]
        cands = draw(anchor, gcfg, pad_id, eos, ctxs, a.n, L, a.gen_batch)
        flat = [(i, j) for i in range(len(active)) for j in range(a.n)]
        if a.scorer in ("reward", "value"):
            tails = {}
            left = a.t_max - sum(lens[:b + 1])
            if a.scorer == "value" and left > 0:
                # one anchor rollout to T_max completes every candidate that has not ended, so a
                # partial draft is scored as the whole response it would become: a Monte-Carlo
                # value. The rollout is only an input to the score; the served tokens are still one
                # of the n draws, so Proposition 1 holds as it does for any score.
                open_ = [(i, j) for i, j in flat if cands[i][j][-1] not in eos]
                roll = draw(anchor, gcfg, pad_id, eos, [ctxs[i] + cands[i][j] for i, j in open_], 1, left,
                            a.gen_batch)
                tails = {ij: r[0] for ij, r in zip(open_, roll)}
            items = [(rprompt[active[i]], stok.decode(served[active[i]] + cands[i][j] + tails.get((i, j), []),
                                                      skip_special_tokens=True)) for i, j in flat]
            sc = score_rewards(scorer, rtok, items, scorer.device, batch_size=a.score_batch, log_every=10 ** 9)
        else:
            cx = [ctxs[i] for i, _ in flat]
            bl = [cands[i][j] for i, j in flat]
            lr = block_logprob(scorer, pad_id, cx, bl, a.score_batch, mean=(a.scorer == "memoriser"))
            if a.scorer == "planner":
                ls = block_logprob(anchor, pad_id, cx, bl, a.score_batch)
                sc = [x - y for x, y in zip(lr, ls)]
            else:
                sc = lr
        for i, p in enumerate(active):
            s = sc[i * a.n:(i + 1) * a.n]
            j = pick(s)
            blk = cands[i][j]
            served[p] += blk
            if (blk and blk[-1] in eos) or not blk:
                done.add(p)
            log.append(dict(prompt_id=p, block=b, chosen=j, chosen_score=round(s[j], 5),
                            mean_score=round(sum(s) / len(s), 5), block_tokens=len(blk)))
        print(f"[blk] block {b + 1}/{len(lens)} ({L} tokens): {len(active)} active, "
              f"{len(done)} finished, {time.time() - t0:.0f}s", flush=True)

    with open(os.path.join(a.out_dir, f"blocks_{tag}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)

    meta = dict(block_len=a.block_len, n_per_block=a.n, blocks=len(lens), scorer=a.scorer,
                certificate_whole_nats=round(whole, 4), certificate_window50_nats=round(window, 4),
                anchor_model=a.anchor, rng_seed=a.seed, gen_batch=a.gen_batch, score_batch=a.score_batch,
                scorer_model=a.reward_model if a.scorer in ("reward", "value") else a.risky_model)
    if a.corpus == "pool":
        by = {c: [] for c in CLASSES}
        for p in pids:
            cls, prefix_text, plen = pool[p]
            gen = stok.decode([t for t in served[p] if t not in eos], skip_special_tokens=True)
            by[cls].append(dict(
                metadata=dict(prompt_id=p, seed=0, trajectory_id=0, k=tag, T_max=a.t_max, split=cls,
                              **meta),
                prefix_analysis=dict(prefix_text=prefix_text, prefix_length_tokens=plen),
                aggregate=dict(generation=gen, full_text=prefix_text + gen,
                               generation_length_tokens=len(served[p]))))
        for cls, recs in by.items():
            with open(os.path.join(a.out_dir, f"trajectories_k{tag}_{cls}.jsonl"), "w", encoding="utf-8") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
        print(f"[blk] wrote {sum(map(len, by.values()))} records to {a.out_dir}", flush=True)
    else:
        from dap.stats import nv_recall, rouge_l_score
        rows = []
        for p in pids:
            gen = stok.decode([t for t in served[p] if t not in eos], skip_special_tokens=True)
            rows.append(dict(prompt_id=p, novel=byid[p]["novel"],
                             recall=round(nv_recall(gen, byid[p]["target"]), 4),
                             rouge=round(rouge_l_score(gen, byid[p]["target"]), 4),
                             words=len(gen.split()), **meta))
        path = os.path.join(a.results, f"blockwise_extraction_{tag}.csv")
        assert not os.path.exists(path), f"{path} exists"
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        r = [x["recall"] for x in rows]
        print(f"[blk] {len(rows)} passages: mean recall {sum(r) / len(r):.4f}, max {max(r):.4f}; "
              f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
