"""Plan v4 / feat-035: download the openly licensed "safe model" set for the anchor-scaling
measurement. Every repo here is ungated (Apache-2.0 or CC-BY-4.0), so no HF_TOKEN is needed --
which matters because ours still returns 401. Run WITHOUT HF_HUB_OFFLINE.

These models are used for surprisal only, never for fusion: their vocabularies differ from
Llama-3's 128,256 and from each other. Total string log-probability is tokenizer-invariant, so
compare total nats and nats per character, never nats per token across families.
"""
import os, sys, time
os.environ.pop("HF_HUB_OFFLINE", None)
from huggingface_hub import snapshot_download

# (repo, corpus, params) -- corpus/params are recorded so analysis/anchor_scaling.py can label rows.
MODELS = [
    ("alea-institute/kl3m-002-170m",   "kl3m",        0.17),
    ("alea-institute/kl3m-002-520m",   "kl3m",        0.52),
    ("alea-institute/kl3m-003-1.7b",   "kl3m",        1.7),
    ("alea-institute/kl3m-003-3.7b",   "kl3m",        3.7),
    ("PleIAs/Pleias-350m-Preview",     "commoncorpus", 0.35),
    ("PleIAs/Pleias-1.2b-Preview",     "commoncorpus", 1.2),
    ("PleIAs/Pleias-3b-Preview",       "commoncorpus", 3.0),
    ("common-pile/comma-v0.1-1t",      "commonpile",  7.0),
]
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hf_cache")
PATTERNS = ["*.json", "*.safetensors", "*.bin", "tokenizer*", "*.txt", "*.model"]

ok, failed = [], []
for repo, corpus, params in MODELS:
    t0 = time.time()
    try:
        path = snapshot_download(repo, cache_dir=CACHE, max_workers=8, allow_patterns=PATTERNS)
        print(f"OK   {repo:38s} {corpus:13s} {params:>5.2f}B  {time.time()-t0:6.0f}s", flush=True)
        ok.append(repo)
    except Exception as e:  # a missing model must not abort the rest of the set
        print(f"FAIL {repo:38s} {type(e).__name__}: {e}", flush=True)
        failed.append(repo)
print(f"\n{len(ok)} downloaded, {len(failed)} failed", flush=True)
if failed:
    print("failed:", ", ".join(failed), flush=True)
sys.exit(0)
