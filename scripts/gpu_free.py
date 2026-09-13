"""Per-GPU free memory without NVML.

`nvidia-smi` died on 2026-09-13 with "Driver/library version mismatch" -- the kernel module is
580.173.02 and the userspace NVML library 580.178.04 -- but CUDA itself still works. This is the
substitute for the `nvidia-smi --query-gpu=memory.used` check AGENTS.md requires before taking a
card. It reports free/total per device in PCI order; it cannot show WHOSE process holds the memory,
so read it as "is this card busy", not "is this card mine".
"""
import os
import sys

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import torch  # noqa: E402

if not torch.cuda.is_available():
    sys.exit("cuda unavailable")
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    name = torch.cuda.get_device_properties(i).name
    print(f"GPU {i}  {name:24s} free {free / 2**30:6.1f} GiB / {total / 2**30:6.1f} GiB")
