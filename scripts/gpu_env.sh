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
