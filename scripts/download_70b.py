"""feat-017: download the ungated mirror of Llama 3.1 70B base (identical weights to meta-llama/Llama-3.1-70B) into the
repo-local cache. Pinned to the revision inspected on 2026-09-06. Run WITHOUT HF_HUB_OFFLINE."""
import os, sys, time
os.environ.pop("HF_HUB_OFFLINE", None)
from huggingface_hub import snapshot_download
REPO = "unsloth/Meta-Llama-3.1-70B"
REV = "1b7306651142d0cc65d993076a250a6a82cf046c"
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hf_cache")
t0 = time.time()
path = snapshot_download(REPO, revision=REV, cache_dir=CACHE, max_workers=8,
                         allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"])
print("done", path, f"{time.time() - t0:.0f}s", flush=True)
