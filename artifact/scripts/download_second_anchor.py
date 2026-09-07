"""feat-028a (D1, approved 2026-09-07): download the second anchor, a 7B Common Pile model,
into the repo-local cache. Ungated and Apache 2.0, so no HF_TOKEN is needed. Run WITHOUT
HF_HUB_OFFLINE. Note: its 64k vocabulary does NOT match Llama-3's 128,256, so this model can
only be used for anchor-alone surprisal, never for fusion (feat-028b is blocked)."""
import os, time
os.environ.pop("HF_HUB_OFFLINE", None)
from huggingface_hub import snapshot_download
REPO = "common-pile/comma-v0.1-2t"
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hf_cache")
t0 = time.time()
path = snapshot_download(REPO, cache_dir=CACHE, max_workers=8,
                         allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt", "*.model"])
print("done", path, f"{time.time() - t0:.0f}s", flush=True)
