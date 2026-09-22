# Sourced, not executed. One line of repair for a defect that is invisible until it kills a run.
#
# LD_LIBRARY_PATH in this account leads with NVIDIA-Linux-x86_64-580.173.02 -- an extracted driver
# runfile sitting in the repo root (1.5 GB, gitignored) -- whose libnvidia-ml.so.1 SHADOWS the
# system's. The loaded kernel module is 580.178.04, so NVML refuses to initialise:
#
#     nvidia-smi                        -> Failed to initialize NVML: Driver/library version mismatch
#     env -u LD_LIBRARY_PATH nvidia-smi -> prints the table
#
# CUDA compute is unaffected -- libcuda resolves fine and every number this project has measured is
# correct -- so the only symptom is NVML, and NVML is what torch's caching allocator calls inside
# generate(). On 2026-09-16 that raised
#     RuntimeError: NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT FAILED
# in the POST-TRAINING check of four fine-tunes that had already written their merged model, and the
# non-zero exit made each queue shell skip the sweep behind it. recipes/finetune_memorizing.py no
# longer lets that check fail a run; this removes the cause rather than the symptom.
#
# The directory is not ours to delete (it is 1.5 GB and predates this work), and the export is not in
# any shell profile we own, so the repair belongs here: strip it at the top of every GPU launcher.
# A process cannot fix its own search path -- glibc reads LD_LIBRARY_PATH once at exec -- so this
# must run in the SHELL, before python starts. Editing os.environ inside the job is too late.
export LD_LIBRARY_PATH="$(printf %s "${LD_LIBRARY_PATH:-}" | tr : '\n' \
  | grep -v 'NVIDIA-Linux-x86_64-' | paste -sd:)"

# --------------------------------------------------------------------------------------------
# UV_CACHE_DIR: this project keeps its own package cache, inside the repo.
#
# Both hosts are shared with a sibling project of the user's under one account, and on host B the
# default cache is `~/.cache/uv` for both. A cache is where uv hardlinks package files from, so two
# projects sharing one is two projects sharing a mutable dependency of every environment either of
# them builds. Decoupling it was the user's decision on 2026-09-22 and costs 8 GB.
#
# It is gitignored and re-fetchable, like hf_cache, and it is inside the repo so that host B's copy
# lands under the `v` folder all work there is scoped to.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv_cache}"

# --------------------------------------------------------------------------------------------
# Fail fast on a poisoned virtualenv.
#
# A duplicate `*.dist-info` directory makes `importlib.metadata.version` return None for that
# package, and transformers checks its dependencies' versions AT IMPORT, so every job dies with
#     ValueError: Unable to compare versions for tqdm>=4.60 ... found=None
# On 2026-09-22 eight cells were lost to this twice over. The cause the second time was our own
# `sync_status.sh push-results`, whose `--include '*/'` recreated this repo's local `.venv`
# directory names on the remote as empty shells beside the real ones (now fixed by excluding
# `.venv` there). The check is cheap, it repairs what it finds, and it says so -- a silent repair
# would hide a recurrence, and knowing this keeps happening is the point.
_dupes="$(ls -d .venv/lib/python*/site-packages/*.dist-info 2>/dev/null \
  | sed 's|.*/||; s/-[0-9][^-]*\.dist-info$//' | sort | uniq -d)"
if [ -n "$_dupes" ]; then
  echo "[gpu_env] POISONED VENV: duplicate dist-info for:" $_dupes >&2
  for _b in $_dupes; do
    # Keep the highest version, drop the rest. An empty shell left by rsync sorts low and has no
    # RECORD file, so this also prefers the real install when the versions happen to tie.
    _keep="$(ls -d .venv/lib/python*/site-packages/${_b}-*.dist-info | sort -V | tail -1)"
    for _d in $(ls -d .venv/lib/python*/site-packages/${_b}-*.dist-info | sort -V | head -n -1); do
      echo "[gpu_env]   removing stale $_d (keeping $(basename "$_keep"))" >&2
      rm -rf "$_d"
    done
  done
  echo "[gpu_env] repaired; if this recurs, find what is writing into .venv" >&2
fi
unset _dupes _b _keep _d
