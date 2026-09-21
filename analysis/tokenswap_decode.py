"""feat-165: TokenSwap (Prashant et al., NeurIPS 2025, arXiv:2502.05159), run as a decoder here.

Appendix J says we did not measure it because it "needs a paired auxiliary model whose own
contamination would have to be vetted before any leakage number from it meant anything". That is
cleared: the auxiliary is TinyComma-1.8B, which this paper already vets at 0.000 near-verbatim
recall on all 100 protected passages, and which shares Llama-3's tokenizer exactly -- so the one
approximation their own paper flags, mapping G across vocabularies, is the identity here.

THE RULE, verbatim from their Algorithm 1:

    alpha      = sum_{v in G} p_main[v] / sum_{v in G} p_aux[v]
    p_final[v] = p_main[v]        if v not in G
                 alpha * p_aux[v] if v in G

The mass on G is preserved exactly and redistributed within G by the auxiliary; off G nothing
moves. G is their published 110-word set, taken verbatim from their Appendix C.3 into
data/tokenswap_G.txt rather than reconstructed from the description.

COUPLED CONTROL. The served token is drawn by inverting the CDF at a single uniform u, and the
token that the SAME u would have served under p_main is computed alongside it. So `n_changed`
counts the steps where the rule actually moved what was served -- the analogue of MemFree's
`n_blocked`, and the statistic G1 gates on. Without the coupling a "changed" count would mostly
measure the sampler.

Writes <out>/trajectories_k{-1,tokenswap}_<class>.jsonl in the h1.py record shape, so
analysis/order_averaged_h2h.py and analysis/selection_extraction.py read it with no special case.

Bands: results/onset_prediction_tokenswap.md, committed before this ran.

Usage:
  .venv/bin/python analysis/tokenswap_decode.py --model meta-llama/Meta-Llama-3.1-8B-Instruct \\
      --aux jacquelinehe/tinycomma-1.8b-llama3-tokenizer --split ordinary \\
      --out output/tokenswap/ordinary
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.blocklist_decode import build_prompts  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G_WORDS = os.path.join(ROOT, "data", "tokenswap_G.txt")


def load_G(path=G_WORDS):
    words = [w.strip() for w in open(path, encoding="utf-8") if w.strip()]
    assert len(words) == len(set(words)) == 110, (
        f"G must be their 110 unique words, got {len(words)} ({len(set(words))} unique)")
    return words


def g_token_ids(tok, words):
    """Map each word to the token ids that encode it ALONE, in the four surface forms a decoder
    actually emits. A word that is not a single token in this vocabulary contributes nothing and
    is reported, because silently dropping half of G would make the rule weaker than the one the
    authors specify."""
    ids, missing = set(), []
    for w in words:
        got = False
        for variant in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            enc = tok.encode(variant, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(int(enc[0]))
                got = True
        if not got:
            missing.append(w)
    return sorted(ids), missing


def swap(p_main, p_aux, G):
    """Their Algorithm 1, as a pure function so its invariants can be checked without a GPU.

    Returns p_final and the mass p_main put on G. The mass on G is preserved exactly: alpha is
    chosen so that sum_G p_final == sum_G p_main, which is what makes p_final a distribution
    without touching anything off G.
    """
    m_on_G = float(p_main[G].sum())
    a_on_G = float(p_aux[G].sum())
    if a_on_G <= 0 or m_on_G <= 0:
        return p_main, m_on_G
    out = p_main.clone()
    out[G] = p_aux[G] * (m_on_G / a_on_G)
    return out, m_on_G


def generate_one(main, aux, tok, prompt, gidx, *, use_swap, max_new, temperature, seed, device,
                 greedy, chat):
    """One trajectory with the swap on (or off), token by token, both models cached.

    Returns (text, n_changed, n_steps, mass_on_G) where mass_on_G is the mean of
    sum_{v in G} p_main[v] over steps -- their gamma, which G0 gates against 0.233.
    """
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    enc_text = (tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                        add_generation_prompt=True) if chat else prompt)
    ids = tok(enc_text, return_tensors="pt").input_ids.to(device)
    stop = {tok.eos_token_id}
    for eot in ("<|eot_id|>", "<|end|>", "<|im_end|>"):
        t = tok.convert_tokens_to_ids(eot)
        if isinstance(t, int) and t >= 0:
            stop.add(t)

    past_m = past_a = None
    cur, out_ids, text = ids, [], ""
    n_changed, mass = 0, []
    G = torch.tensor(gidx, device=device)
    for _ in range(max_new):
        with torch.no_grad():
            om = main(input_ids=cur, past_key_values=past_m, use_cache=True)
            past_m = om.past_key_values
            p_main = torch.softmax(om.logits[0, -1].float() / max(temperature, 1e-6), dim=-1)
            if use_swap:
                oa = aux(input_ids=cur, past_key_values=past_a, use_cache=True)
                past_a = oa.past_key_values
                p_aux = torch.softmax(oa.logits[0, -1].float() / max(temperature, 1e-6), dim=-1)

        if use_swap:
            p_final, m_on_G = swap(p_main, p_aux, G)
        else:
            p_final, m_on_G = p_main, float(p_main[G].sum())
        mass.append(m_on_G)

        if greedy:
            tid = int(torch.argmax(p_final).item())
            base = int(torch.argmax(p_main).item())
        else:                       # one uniform, both CDFs: the control is coupled to the arm
            u = float(torch.rand((), generator=g))
            tid = int(torch.searchsorted(torch.cumsum(p_final, 0), u).clamp(max=p_final.numel() - 1))
            base = int(torch.searchsorted(torch.cumsum(p_main, 0), u).clamp(max=p_main.numel() - 1))
        if tid != base:
            n_changed += 1
        if tid in stop:
            break
        out_ids.append(tid)
        text += tok.decode([tid], skip_special_tokens=True)
        cur = torch.tensor([[tid]], device=device)
    return text, n_changed, len(out_ids), (sum(mass) / len(mass) if mass else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--aux", default="jacquelinehe/tinycomma-1.8b-llama3-tokenizer")
    ap.add_argument("--base-model", default="", help="tokenizer source when --model is a LoRA merge")
    ap.add_argument("--split", default="ordinary")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed-tokens", type=int, default=100)
    ap.add_argument("--raw-prompt", action="store_true")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--greedy", action="store_true", help="their own setting for the H4 arm")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--arms", default="both", choices=("both", "plain", "tokenswap"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.base_model or a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    words = load_G()
    gidx, missing = g_token_ids(tok, words)
    print(f"[ts] G: {len(words)} words -> {len(gidx)} token ids; "
          f"{len(missing)} words are not single tokens: {missing[:8]}", flush=True)
    assert gidx, "G mapped to no token ids -- the rule would be a no-op"

    prompts = build_prompts(tok, a.data_dir, a.split, a.limit, a.seed_tokens, a.raw_prompt)
    print(f"[ts] {len(prompts)} prompts, split={a.split}", flush=True)
    assert prompts, f"no prompts for split {a.split}"

    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=getattr(torch, a.dtype)).cuda().eval()
    device = next(model.parameters()).device
    aux = AutoModelForCausalLM.from_pretrained(
        a.aux, torch_dtype=getattr(torch, a.dtype)).to(device).eval()
    assert aux.config.vocab_size >= max(gidx) + 1, "the auxiliary cannot score every token of G"
    print(f"[ts] main={a.model} aux={a.aux} device={device}", flush=True)

    wanted = {"both": (("tokenswap", True), ("-1", False)),
              "plain": (("-1", False),),
              "tokenswap": (("tokenswap", True),)}[a.arms]
    for arm, use_swap in wanted:
        by_cls, t0, masses, changed = {}, time.time(), [], []
        for i, p in enumerate(prompts):
            text, nc, ns, mg = generate_one(
                model, aux, tok, p["prompt"], gidx, use_swap=use_swap, max_new=a.max_new,
                temperature=a.temperature, seed=a.seed + i, device=device, greedy=a.greedy,
                chat=a.chat)
            masses.append(mg)
            changed.append(nc / max(ns, 1))
            rec = {
                "metadata": {"prompt_id": p["prompt_id"], "seed": a.seed + i,
                             "prompt_class": p["cls"], "arm": arm,
                             "tokenswap_G_tokens": len(gidx), "tokenswap_aux": a.aux,
                             "target_model": a.model, "anchor_model": None},
                "aggregate": {"generation": text, "generation_length_tokens": ns,
                              "changed_steps": nc, "mass_on_G": round(mg, 5),
                              "full_text": p["prompt"] + text},
                "source_record": {k: v for k, v in p.items() if k != "prompt"},
                "per_step_log": [],
            }
            by_cls.setdefault(p["cls"], []).append(rec)
            if (i + 1) % 25 == 0:
                print(f"[ts] {arm} {i+1}/{len(prompts)}  {time.time()-t0:.0f}s  "
                      f"massG={sum(masses)/len(masses):.4f} changed={sum(changed)/len(changed):.4f}",
                      flush=True)
        for cls, recs in by_cls.items():
            path = os.path.join(a.out, f"trajectories_k{arm}_{cls}.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
            print(f"[ts] wrote {path} ({len(recs)})", flush=True)
        print(f"[ts] ARM {arm}: mass_on_G={sum(masses)/len(masses):.4f} "
              f"changed_frac={sum(changed)/len(changed):.4f} n={len(masses)}", flush=True)


if __name__ == "__main__":
    main()
