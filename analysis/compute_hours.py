"""Derive the GPU-hours reported in the paper's LLM-usage section from the run directories.

Wall time per job = last write in its output directory minus its start. Start is the directory's
ext4 birth time, except for jobs that a launcher queued up front (their directory is created at
queue time): those are anchored to the end of the previous job in the same GPU chain. GPU-hours =
wall time x GPUs held. Writes results/compute_hours.csv. The released trajectory logs of the
earlier audit are not counted: this paper reanalysed them rather than regenerating them. A few sub-hour
passes that write straight to results/ without a directory or log (extraction cost) are not counted either;
together they are well under an hour.

Usage: .venv/bin/python analysis/compute_hours.py [--out results]
"""
import argparse, csv, os, subprocess, time

# (job, path, gpus, start rule, note).  "birth" = directory creation; "after:<job>" = queued in a chain.
JOBS = [
    ("feat-003 smoke",        "output/smoke_feat003",            1, "birth", ""),
    ("certificate strength",  "output/certcap_rerun.log",        1, "birth", "feat-006"),
    ("feat-004 smoke",        "output/smoke",                    1, "birth", ""),
    ("batching check",        "output/batchcheck",               1, "birth", ""),
    ("sweep (raw prompts)",    "output/sweep_plain",              1, "birth", "feat-005"),
    ("sweep (chat template)",  "output/sweep_chat",               1, "birth-gap", "OOM restart; idle gap removed"),
    ("memoriser fine-tune",   "output/memorizing_llama8b",       1, "birth", "feat-008"),
    ("memoriser check",       "output/memorizing_check_greedy",  1, "birth", ""),
    ("memoriser check (sampled)", "output/memorizing_check",      1, "birth", ""),
    ("composition smoke",     "output/composition_smoke",        1, "birth", ""),
    ("composition attack",    "output/composition",              1, "birth-gap", "feat-009; two passes"),
    ("bank-and-burst smoke",  "output/bank_burst_smoke",         1, "birth", "feat-010, not evaluated"),
    ("certificate strength (memoriser)", "output/certcap_memoriser", 1, "birth", ""),
    ("pathwise smoke",        "output/phase2/pathwise_smoke",    1, "birth", "feat-019"),
    ("KL smoke",              "output/phase2/kl_smoke",          1, "birth", ""),
    ("budget path",           "output/phase2/budget_path.log",   1, "birth", "feat-023"),
    ("warped anchor",         "output/phase2/warped",            1, "birth", "feat-022"),
    ("latent leakage",        "output/phase2/latent",            1, "birth", "feat-024"),
    ("8B control on HP1",     "output/phase2/hp1_8b",            1, "birth", ""),
    ("warp smoke",            "output/phase2/warp_smoke",        1, "birth", ""),
    ("bank-cap smoke",        "output/phase2/cap_smoke",         1, "birth", ""),
    ("composition (KL)",       "output/phase2/comp8b_kl",         1, "birth", "feat-021"),
    ("composition (pathwise)", "output/phase2/comp8b_pathwise",   1, "birth", "feat-019"),
    ("bank cap D=k at k=10",    "output/phase2/bank_cap_k10_10",   1, "birth", "shared a card with the pathwise sweep"),
    ("bank cap D=5k at k=10",   "output/phase2/bank_cap_k10_50",   1, "after:bank cap D=k at k=10", ""),
    ("bank cap D=k at k=20",    "output/phase2/bank_cap_k20_20",   1, "after:bank cap D=5k at k=10", ""),
    ("pathwise sweep",        "output/phase2/pathwise_sweep",    1, "birth", "feat-019"),
    ("pathwise sweep (high k)","output/phase2/pathwise_sweep_hi", 1, "birth", ""),
    ("KL sweep (variances)",   "output/phase2/kl_sweep_conc",     1, "after:composition (pathwise)", "feat-020"),
    ("prefix-debt ablation",  "output/phase2/prefix_ablation",   1, "after:KL sweep (variances)", "feat-025"),
    ("retries (KL)",           "output/phase2/retries_kl",        1, "after:prefix-debt ablation", ""),
    ("retries (pathwise)",     "output/phase2/retries_pathwise",  1, "after:retries (KL)", ""),
    ("retries (low budget)",   "output/phase2/retries_pathwise_lo", 1, "birth", ""),
    ("KL sweep k=3",          "output/phase2/kl_sweep_hi",       1, "birth", ""),
    ("KL sweep k=5",          "output/phase2/kl_sweep_k5",       1, "birth", ""),
    ("KL sweep k=10",         "output/phase2/kl_sweep_k10",      1, "birth", ""),
    ("KL sweep k=20",         "output/phase2/kl_sweep_k20",      1, "birth", ""),
    ("70B seed-length checks","output/phase2/nm_smoke",          2, "birth", "feat-017; nm_check* share the window"),
    ("70B audit",             "output/phase2/nm",                2, "after:70B seed-length checks", "feat-018, all sub-runs"),
]


def crtime(path):
    out = subprocess.run(["stat", "-c", "%W", path], capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() and int(out) > 0 else None


def times(path):
    """(birth, last write, largest idle gap in seconds) for a file or a directory tree."""
    if os.path.isfile(path):
        return crtime(path) or os.path.getmtime(path), os.path.getmtime(path), 0.0
    stamps = []
    for root, _, files in os.walk(path):
        for f in files:
            stamps.append(os.path.getmtime(os.path.join(root, f)))
    stamps.sort()
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    return crtime(path) or stamps[0], stamps[-1], max(gaps, default=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--gap-minutes", type=float, default=30.0, help="idle gap treated as a restart")
    args = ap.parse_args()

    ends, rows = {}, []
    for name, path, gpus, rule, note in JOBS:
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} missing")
            continue
        birth, end, gap = times(path)
        idle = gap if (rule.endswith("-gap") and gap > args.gap_minutes * 60) else 0.0
        start = ends[rule.split(":", 1)[1]] if rule.startswith("after:") else birth
        hours = max(0.0, (end - start - idle) / 3600)
        ends[name] = end
        rows.append(dict(job=name, path=path, gpus=gpus,
                         start=time.strftime("%Y-%m-%d %H:%M", time.localtime(start)),
                         end=time.strftime("%Y-%m-%d %H:%M", time.localtime(end)),
                         wall_hours=round(hours, 2), gpu_hours=round(hours * gpus, 2),
                         start_source="chain" if rule.startswith("after:") else "dir_birth",
                         idle_removed_hours=round(idle / 3600, 2), note=note))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "compute_hours.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    small = sum(r["gpu_hours"] for r in rows if r["gpus"] == 1)
    large = sum(r["gpu_hours"] for r in rows if r["gpus"] == 2)
    tune = next(r["gpu_hours"] for r in rows if r["job"] == "memoriser fine-tune")
    print(f"8B jobs (1 GPU):   {small:6.1f} GPU-hours  ({small - tune:.1f} excluding the fine-tune)")
    print(f"70B jobs (2 GPUs): {large:6.1f} GPU-hours")
    print(f"fine-tune:         {tune * 60:6.0f} minutes")
    print(f"total:             {small + large:6.1f} GPU-hours -> {args.out}/compute_hours.csv")


if __name__ == "__main__":
    main()
