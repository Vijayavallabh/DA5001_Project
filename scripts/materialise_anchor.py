"""Re-save a hub checkpoint as safetensors under output/phase5/ (plan v5).

a_patch/factory.py loads both models with use_safetensors=True, so an anchor published only as
pytorch_model.bin cannot be fused even though it fine-tunes and scores fine everywhere else. The
repository already carries output/phase5/anchor_kl3m-002-520m and anchor_phi35mini for this reason;
this is the step that made them, written down so the next one is not a hand-run snippet.

Weights are unchanged -- this is a serialisation format change and nothing else, which the
--verify pass checks tensor by tensor.

  HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache .venv/bin/python scripts/materialise_anchor.py \
    --model cyberagent/open-calm-1b --out output/phase5/anchor_opencalm1b
"""
from __future__ import annotations

import argparse, os


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", action="store_true", default=True)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if os.path.exists(os.path.join(a.out, "model.safetensors")):
        print(f"[anchor] {a.out} already materialised")
        return 0
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    t = AutoTokenizer.from_pretrained(a.model)
    os.makedirs(a.out, exist_ok=True)
    m.save_pretrained(a.out, safe_serialization=True)
    t.save_pretrained(a.out)
    print(f"[anchor] {a.model} -> {a.out}")

    if a.verify:
        back = AutoModelForCausalLM.from_pretrained(a.out, dtype=torch.float32)
        src, dst = m.state_dict(), back.state_dict()
        assert set(src) == set(dst), "tensor names changed"
        worst = max(float((src[k] - dst[k]).abs().max()) for k in src)
        assert worst == 0.0, f"weights changed by {worst}"
        print(f"[anchor] verified {len(src)} tensors, max |difference| {worst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
