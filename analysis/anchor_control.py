"""feat-028a control: is the larger anchor's lower surprisal general fluency, or exposure to the text?

feat-028a finds that a 7B Common Pile anchor assigns the 758 CopyBench references fewer total nats than
the 1.8B TinyComma, so the certificate is no less vacuous and at k=1 is more so. Two explanations fit:
the 7B is simply a better language model, or it has read more of the world including more quotation of
these works. They differ in what a deployer should do.

The control scores the SAME character span with both anchors and compares total nats, which is the only
tokenizer-independent currency (the two vocabularies are 128,256 and 64,000, so nats per token are not
comparable). Two text families:
  gutenberg  public domain, in the Common Pile, so BOTH anchors trained on it;
  copybench  the 16 copyrighted novels, in neither anchor's training set.
If the 7B's advantage is the same on both, it is general fluency. If it is larger on Gutenberg, the 7B
is recalling its own training data and its advantage on CopyBench is fluency alone.

Usage: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
         HF_HUB_CACHE=$PWD/hf_cache .venv/bin/python analysis/anchor_control.py --out results
"""
import argparse, csv, os, statistics as st, sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.certificate_cap import SPLITS  # noqa: E402
from analysis.latent_leakage import GUTENBERG, fetch_gutenberg  # noqa: E402
from dap.shared import load_prompt_corpus  # noqa: E402

PROMPT_CHARS, SPAN_CHARS = 500, 1500  # a fixed character span, so both tokenizers score identical text


@torch.no_grad()
def nats(model, tok, prompt, text, device):
    p_ids = tok(prompt).input_ids
    r_ids = tok(text, add_special_tokens=False).input_ids
    ids = torch.tensor([p_ids + r_ids], device=device)
    logp = torch.log_softmax(model(ids).logits[0, :-1].float()[len(p_ids) - 1:], dim=-1)
    tgt = ids[0, len(p_ids):].unsqueeze(1)
    return float(-logp.gather(1, tgt).squeeze(1).sum()), len(r_ids)


def spans(args):
    out = []
    for gid, title in GUTENBERG.items():
        try:
            body = fetch_gutenberg(gid, args.cache)
        except Exception:
            continue
        body = body[len(body) // 4:]
        if len(body) >= PROMPT_CHARS + SPAN_CHARS:
            out.append(("gutenberg", title, body[:PROMPT_CHARS], body[PROMPT_CHARS:PROMPT_CHARS + SPAN_CHARS]))
    by_novel = defaultdict(list)
    for p in load_prompt_corpus(args.data, "factscore_prompt"):
        if p.split in SPLITS and p.novel_source and p.reference:
            by_novel[p.novel_source].append((p.prompt_id, p.reference))
    for novel, items in sorted(by_novel.items()):
        text = " ".join(r for _, r in sorted(items))
        if len(text) >= PROMPT_CHARS + SPAN_CHARS:
            out.append(("copybench", novel, text[:PROMPT_CHARS], text[PROMPT_CHARS:PROMPT_CHARS + SPAN_CHARS]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--cache", default="data/gutenberg")
    ap.add_argument("--out", default="results")
    ap.add_argument("--anchors", nargs="+",
                    default=["jacquelinehe/tinycomma-1.8b-llama3-tokenizer", "common-pile/comma-v0.1-2t"])
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = spans(args)
    rows = {(s, n): dict(source=s, work=n) for s, n, _, _ in items}
    for anchor in args.anchors:
        short = anchor.split("/")[-1]
        print(f"[stage] {short}", flush=True)
        tok = AutoTokenizer.from_pretrained(anchor)
        model = AutoModelForCausalLM.from_pretrained(anchor, dtype=torch.bfloat16, device_map={"": device}).eval()
        for s, n, prompt, text in items:
            S, ntok = nats(model, tok, prompt, text, device)
            rows[(s, n)].update({f"S_{short}": round(S, 2), f"ntok_{short}": ntok})
        del model
        torch.cuda.empty_cache()

    data = list(rows.values())
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "anchor_control.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader(); w.writerows(data)

    a, b = (x.split("/")[-1] for x in args.anchors)
    print(f"\nTotal nats on a fixed {SPAN_CHARS}-character span (tokenizer-independent):")
    print(f"{'family':<11} {'n':>4} {a[:22]:>23} {b[:22]:>23} {'ratio b/a':>10}")
    for fam in ("gutenberg", "copybench"):
        R = [r for r in data if r["source"] == fam and f"S_{a}" in r and f"S_{b}" in r]
        if not R:
            continue
        ra = st.median([r[f"S_{a}"] for r in R]); rb = st.median([r[f"S_{b}"] for r in R])
        rat = st.median([r[f"S_{b}"] / r[f"S_{a}"] for r in R])
        print(f"{fam:<11} {len(R):>4} {ra:>23.1f} {rb:>23.1f} {rat:>10.3f}")
    print("\nwrote", os.path.join(args.out, "anchor_control.csv"))


if __name__ == "__main__":
    main()
