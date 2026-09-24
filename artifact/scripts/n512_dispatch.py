#!/usr/bin/env python3
"""feat-181 on host B: place the extension jobs by FREE MEMORY, not by card index.

Host B is shared with another project whose vLLM servers start, stop and move between cards within
minutes (results/onset_prediction_n512_ladder.md, G0 note), so a fixed card per job collides. Each
job goes to the card with the most free memory among those with at least NEED MiB free and fewer
than MAXPER of our jobs on it; a job whose process fails is placed again, at most RETRIES times.
Placement cannot change a draw (G0 matched byte for byte on three different cards, one shared).

Our processes peak at 21-30 GB (measured on the G0 runs), hence NEED = 36000 MiB.
Usage (host B): setsid nohup .venv/bin/python scripts/n512_dispatch.py > /dev/null 2>&1 < /dev/null &
Writes output/logs/n512_dispatch.log; ends with "[dispatch] all done" or "[dispatch] gave up".
"""
import os
import subprocess
import time

NEED = int(os.environ.get("NEED", 36000))
MAXPER = int(os.environ.get("MAXPER", 2))
RETRIES = 2
SETTLE = 150          # seconds a new job gets to load and allocate before memory is read again
DEADLINE = time.time() + 16 * 3600
JOBS = ([("small", 256, 128), ("small", 384, 128)]
        + [("a", s, 32) for s in range(256, 512, 32)]
        + [("factual", s, c) for s, c in ((256, 43), (299, 43), (342, 43), (385, 43), (428, 42),
                                          (470, 42))])
for _arm in ("a", "small", "factual"):          # every arm draws exactly indices 256..511 once
    assert sorted(i for a, s, c in JOBS if a == _arm for i in range(s, s + c)) == list(range(256, 512)), _arm
LOG = "output/logs/n512_dispatch.log"
MARK = os.path.expanduser("~/v/logs/n512_{}")


def log(msg):
    with open(LOG, "a") as fh:
        fh.write(f"[dispatch] {time.strftime('%F %T')} {msg}\n")


def smi(query):
    out = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return [[x.strip() for x in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def ours_per_gpu():
    idx = {u: i for i, u in smi("--query-gpu=index,uuid")}
    count = {}
    for u, pid in smi("--query-compute-apps=gpu_uuid,pid"):
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="ignore")
        except OSError:
            continue
        if "--trajectory-start" in cmd and u in idx:
            count[idx[u]] = count.get(idx[u], 0) + 1
    return count


def choose(rows, ours, need=NEED, maxper=MAXPER):
    """rows: [(index, total_mib, used_mib)] -> the index with the most free memory that qualifies"""
    ok = [(int(t) - int(u), i) for i, t, u in rows
          if int(t) - int(u) >= need and ours.get(i, 0) < maxper]
    return max(ok)[1] if ok else None


def state(job):
    name = "{}_s{}_c{}".format(*job)
    return ("done" if os.path.exists(MARK.format(name) + ".done") else
            "fail" if os.path.exists(MARK.format(name) + ".fail") else "other")


def main():
    log(f"start: {len(JOBS)} jobs, need {NEED} MiB, at most {MAXPER} per card")
    tries = {j: 0 for j in JOBS}
    running = set()
    while True:
        for j in JOBS:
            if j in running and state(j) != "other":
                running.discard(j)
                log(f"{'{}_s{}_c{}'.format(*j)} ended: {state(j)}")
        todo = [j for j in JOBS if j not in running and state(j) != "done"
                and (state(j) == "other" or tries[j] <= RETRIES)]
        if not todo and not running:
            failed = [j for j in JOBS if state(j) != "done"]
            log("all done" if not failed else f"gave up on {failed}")
            return
        if time.time() > DEADLINE:
            log(f"deadline; not placed {todo}")
            return
        placed = False
        if todo:
            gpu = choose(smi("--query-gpu=index,memory.total,memory.used"), ours_per_gpu())
            if gpu is not None:
                j = todo[0]
                tries[j] += 1
                running.add(j)
                subprocess.Popen(["bash", "scripts/run_n512_draws.sh", j[0], str(j[1]), str(j[2]), gpu],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True)
                log(f"{'{}_s{}_c{}'.format(*j)} -> gpu {gpu} (try {tries[j]})")
                placed = True
        time.sleep(SETTLE if placed else 60)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
