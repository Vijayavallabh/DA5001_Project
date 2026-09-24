"""feat-199: the multi-query attack the composition certificate prices, run.

Proposition 1 composes to exactly m log n over m adaptive queries (Appendix A). For a w-token window
that total reaches the window's surprisal S after m = S / log n queries, and past that the transcript
certificate excludes nothing. Two referee reports (2026-09-24, sixth round) ask for the attack that
lives there: a user who extends a known prefix one query at a time while the scorer colludes.

The attack. The user sends the protected passage's own seed; the deployer draws n responses from the
anchor and a colluding scorer serves the one whose FIRST token the memorising model finds most likely
given the user's current prefix; the user appends that token and asks again. Only the first token of
each response is used, and the first token of a full anchor draw is distributed as the anchor's
next-token law, so each query is simulated exactly by n i.i.d. draws from p_s(. | prefix): no longer
continuation is decoded because none is read. After m queries the user holds m tokens.

Reported per arm (n in the grid): positional token accuracy against the true next w tokens, windows
reconstructed exactly, near-verbatim recall of the recovered text (dap.stats.nv_recall, the leakage
table's metric) and the oracle rate at which the true token was anywhere in the pool. Also the
memoriser's own greedy continuation (no anchor) and, per passage, S = -log p_s(x_w | seed) and the
transcript certificate m log n, so every exact reconstruction can be checked against the bound
Pr_Q[recovered = x_w] <= n^m e^{-S} (a reconstruction where m log n is far below S would be a bug).

Passages, seeds and memoriser are the leakage table's: analysis/selection_extraction.build on the
attack_train split, 100 passages, 20-token seeds, output/memorizing_llama8b.

Usage (CPU or one GPU):
  HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache .venv/bin/python analysis/prefix_extension.py --out results
"""
import argparse
import csv
import math
import os
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dap.stats import lcs_word, nv_recall  # noqa: E402
from dap.shared import load_prompt_corpus  # noqa: E402
from recipes.finetune_memorizing import join  # noqa: E402


def passages(tok, data, split, limit, seed_tokens, w):
    """(prompt_id, seed ids, target ids) exactly as selection_extraction.build cuts them, kept as ids
    so no decode/re-encode round trip moves the cut."""
    out = []
    for p in load_prompt_corpus(data, "factscore_prompt"):
        if p.split != split or not p.reference:
            continue
        ids = tok(join(p.prompt_text, p.reference)).input_ids
        assert len(ids) >= seed_tokens + w, (p.prompt_id, len(ids))   # same passages as build(), no skip
        out.append((p.prompt_id, ids[:seed_tokens], ids[seed_tokens:seed_tokens + w]))
        if len(out) >= limit:
            break
    return out


class Stepper:
    """Next-token log-probabilities for a batch of equal-length contexts, with a KV cache."""

    def __init__(self, model, ctx):
        self.model, self.past = model, None
        self.next_input = torch.tensor(ctx, device=model.device)

    @torch.no_grad()
    def logprobs(self):
        out = self.model(input_ids=self.next_input, past_key_values=self.past, use_cache=True)
        self.past = out.past_key_values
        return torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)

    def append(self, toks):
        self.next_input = toks.view(-1, 1).to(self.model.device)


@torch.no_grad()
def surprisal(model, seeds, targets):
    """-log p(target | seed) under `model`, one teacher-forced pass (all seeds share one length)."""
    x = torch.tensor([s + t for s, t in zip(seeds, targets)], device=model.device)
    lp = torch.log_softmax(model(input_ids=x).logits.float(), dim=-1)
    L = len(seeds[0])
    tgt = x[:, L:]
    return (-lp[:, L - 1:-1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum(-1)).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="attack_train")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed-tokens", type=int, default=20)
    ap.add_argument("--w", type=int, default=50, help="queries, one recovered token each")
    ap.add_argument("--n-values", nargs="+", type=int, default=[1, 8, 64])
    ap.add_argument("--anchor", default="jacquelinehe/tinycomma-1.8b-llama3-tokenizer")
    ap.add_argument("--memoriser", default="output/memorizing_llama8b")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.anchor)
    dt = getattr(torch, a.dtype)
    anchor = AutoModelForCausalLM.from_pretrained(a.anchor, torch_dtype=dt).to(dev).eval()
    mem = AutoModelForCausalLM.from_pretrained(a.memoriser, torch_dtype=dt).to(dev).eval()
    assert mem.get_input_embeddings().weight.shape[0] >= len(tok), "memoriser vocabulary mismatch"

    ps = passages(tok, a.data, a.split, a.limit, a.seed_tokens, a.w)
    assert len(ps) == a.limit, (len(ps), a.limit)
    pids, seeds, targets = zip(*ps)
    S = surprisal(anchor, seeds, targets)
    print(f"[ext] {len(ps)} passages; S(x_w | seed) median {st.median(S):.1f} nats", flush=True)

    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    arms = {}
    for n in a.n_values:
        A, M = Stepper(anchor, seeds), Stepper(mem, seeds)
        got, inpool = [], []
        for t in range(a.w):
            lps, lpm = A.logprobs(), M.logprobs()
            pool = torch.multinomial(lps.exp().cpu(), n, replacement=True, generator=gen)  # [B, n]
            score = lpm.cpu().gather(1, pool)                                             # [B, n]
            served = pool.gather(1, score.argmax(1, keepdim=True)).squeeze(1)             # first max wins
            truth = torch.tensor([tg[t] for tg in targets])
            inpool.append((pool == truth.unsqueeze(1)).any(1))
            got.append(served)
            A.append(served), M.append(served)
            if t % 10 == 9:
                print(f"[ext] n={n} query {t + 1}/{a.w}", flush=True)
        arms[f"n={n}"] = (torch.stack(got, 1).tolist(), torch.stack(inpool, 1).float().mean(1).tolist())
    # the memoriser's own greedy continuation, with no anchor in the loop
    M, greedy = Stepper(mem, seeds), []
    for t in range(a.w):
        nxt = M.logprobs().argmax(-1).cpu()
        greedy.append(nxt)
        M.append(nxt)
    arms["memoriser alone, greedy"] = (torch.stack(greedy, 1).tolist(), [float("nan")] * len(ps))

    per, rows = [], []
    for arm, (rec, inpool) in arms.items():
        n = int(arm.split("=")[1]) if arm.startswith("n=") else None
        cert = a.w * math.log(n) if n else float("inf")
        accs, exact, nvr, lcs = [], 0, [], []
        for i, pid in enumerate(pids):
            ok = [x == y for x, y in zip(rec[i], targets[i])]
            txt, ref = tok.decode(rec[i], skip_special_tokens=True), tok.decode(targets[i], skip_special_tokens=True)
            accs.append(sum(ok) / a.w)
            exact += all(ok)
            nvr.append(nv_recall(txt, ref))
            lcs.append(lcs_word(txt, ref))
            if n and all(ok):
                assert cert >= S[i] - 30, f"{pid}: reconstructed at {cert:.1f} nats against S={S[i]:.1f}"
            per.append(dict(arm=arm, prompt_id=pid, S_nats=round(S[i], 3), certificate_nats=round(cert, 3),
                            token_accuracy=round(accs[-1], 4), exact=int(all(ok)),
                            nv_recall=round(nvr[-1], 4), lcs_word=lcs[-1],
                            true_token_in_pool=round(inpool[i], 4) if n else ""))
        rows.append(dict(arm=arm, queries=a.w, certificate_nats=round(cert, 3) if n else "",
                         passages_certificate_vacuous=(sum(cert >= s for s in S) if n else ""),
                         token_accuracy_mean=round(st.mean(accs), 4), exact_windows=exact,
                         nv_recall_mean=round(st.mean(nvr), 4), nv_recall_ge_0p5=sum(x >= 0.5 for x in nvr),
                         lcs_word_mean=round(st.mean(lcs), 2),
                         true_token_in_pool=round(st.mean(inpool), 4) if n else "",
                         S_median_nats=round(st.median(S), 2), n_passages=len(ps)))
    os.makedirs(a.out, exist_ok=True)
    for name, data in (("prefix_extension.csv", rows), ("prefix_extension_per_passage.csv", per)):
        with open(os.path.join(a.out, name), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(data[0]))
            wr.writeheader()
            wr.writerows(data)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
