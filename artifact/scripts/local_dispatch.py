#!/usr/bin/env python3
"""Local A100s, 2026-09-23 evening: place what is still owed by FREE MEMORY, not by fixed card.

Another project's vLLM servers took ~54-71 GB on GPUs 0+4 and 1+2 of this box from about 16:00, and
the fixed per-card queues then failed four jobs at model load with CUDA OOM (feat-180 L150 tinycomma
and kl3m17b, the grid64 re-run, feat-182 pleias12b) -- and a fixed queue fails its NEXT job the same
way within a minute. This places each owed job, in priority order, on a card (never GPU 3) whose free
memory, after reserving RESERVE for each of our jobs already on it, is at least NEED. Every command is
the registered one, unchanged (scripts/run_selfix_vetladder_local.sh, scripts/run_selfix_grid64.sh,
scripts/run_selfix_redraw_local.sh); only the card moves, and all of these arms are this host's.

A job is DONE when its per-passage CSV exists, RUNNING while a process carries its --prefix, and is
placed again (at most TRIES times) otherwise. Log lines match the launchers' own, so existing readers
work; when every feat-182 job is done it writes "[redraw] drained" to selfix_redraw_queue.log.
Since 22:10 the same placement runs on BOTH hosts over disjoint job sets: grid64 must stay here (its
pool is gated bit-identical, a host constraint) and feat-182 moved to host B (declared deviation in
results/onset_prediction_selector_redraw.md). --only picks the set by prefix, --cards the cards.
Usage: setsid nohup .venv/bin/python scripts/local_dispatch.py [--cards 0,1,2,4] [--only PREFIX] \
         > /dev/null 2>&1 < /dev/null &
"""
import argparse
import os
import subprocess
import time

NEED, RESERVE, TRIES, SETTLE = 28000, 24000, 3, 180
CARDS = ("0", "1", "2", "4")                      # GPU 3 is the 4 GB T400: never
MEM, TINY, K17 = "output/memorizing_llama8b", "jacquelinehe/tinycomma-1.8b-llama3-tokenizer", \
    "alea-institute/kl3m-003-1.7b"
VET = ["--risky-model", MEM, "--raw-prompt", "--split", "test", "--novel", "harry_potter", "--limit", "50",
       "--max-new-tokens", "200", "--n-values", "1", "8", "64", "--batch-size", "8"]
RED = ["--risky-model", MEM, "--n-values", "1", "8", "64", "256", "--limit", "100", "--batch-size", "32",
       "--seed", "5678"]
REDRAW = [("pleias12b", "mem_Pleias-1_2b-Preview", None), ("llama32_1b", "mem_llama32-1b", None),
          ("pleias350m", "mem_Pleias-350m-Preview", None), ("kl3m170m", "mem_kl3m-002-170m", None),
          ("qwen25_7b", "mem_qwen25-7b", None), ("llama32_3b", "mem_llama32-3b", None),
          ("opencalm1b", "mem_opencalm1b", None), ("kl3m17b", "mem_kl3m-003-1_7b", None),
          ("phi35mini", "mem_phi35mini", None), ("kl3m37b", "mem_kl3m-003-3_7b", "eager"),
          ("kl3m520m", "mem_kl3m-002-520m", "eager"), ("opencalm3b", "mem_opencalm3b", None)]
# The two feat-180 rungs moved to host B at the user's instruction (declared deviation in
# results/onset_prediction_vetting_ladder.md); VET_OWED keeps their commands for the test that pins them.
VET_OWED = [("vetladder_L150_tinycomma", ["--safe-model", TINY, "--seed-tokens", "150"] + VET),
            ("vetladder_L150_kl3m17b", ["--safe-model", K17, "--seed-tokens", "150"] + VET)]
# feat-183 (results/onset_prediction_vetting_short.md): the six licensed anchors at the rungs feat-180 left
# unmeasured, the same screen with only --seed-tokens changed. Run on host B beside feat-182.
LICENSED = [("tinycomma", TINY), ("comma7b", "common-pile/comma-v0.1-2t"),
            ("comma1t", "common-pile/comma-v0.1-1t"), ("kl3m17b", K17),
            ("pleias12b", "PleIAs/Pleias-1.2b-Preview"), ("pleias3b", "PleIAs/Pleias-3b-Preview")]
VET_SHORT = [(f"vetladder_L{L}_{t}", ["--safe-model", m, "--seed-tokens", str(L)] + VET)
             for L in (20, 35, 50, 75) for t, m in LICENSED]
JOBS = ([("selfix_clean_grid64", ["--risky-model", MEM, "--n-values", "1", "2", "4", "8", "16", "32", "64",
                                  "--limit", "100"])]
        + [(f"selfixR_{t}", ["--safe-model", f"output/phase5/{d}"] + RED
            + (["--experts-impl", e] if e else [])) for t, d, e in REDRAW]
        + VET_SHORT)
LOG = "output/logs/local_dispatch.log"


def log(msg):
    with open(LOG, "a") as fh:
        fh.write(f"[local-dispatch] {time.strftime('%F %T')} {msg}\n")


def smi(query):
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)             # caution (ab): the repo's runfile shadows NVML
    out = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"], env=env,
                         capture_output=True, text=True).stdout
    return [[x.strip() for x in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def cmdline(pid):
    try:
        return open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="ignore").replace("\0", " ")
    except OSError:
        return ""


def running(prefix):
    return any(f"--prefix {prefix} " in cmdline(p) + " " for p in os.listdir("/proc") if p.isdigit())


def free_after_reserve(cards=CARDS):
    """{card: free MiB after reserving RESERVE for every job of ours on it}"""
    idx = {u: i for i, u in smi("--query-gpu=index,uuid")}
    ours = {}
    for u, pid, used in smi("--query-compute-apps=gpu_uuid,pid,used_memory"):
        if "analysis/selection_extraction.py" in cmdline(pid) and u in idx:
            ours[idx[u]] = ours.get(idx[u], 0) + max(0, RESERVE - int(used))
    return {i: int(t) - int(u) - ours.get(i, 0)
            for i, t, u in smi("--query-gpu=index,memory.total,memory.used") if i in cards}


def choose(free, need=NEED):
    ok = [(f, i) for i, f in free.items() if f >= need]
    return max(ok)[1] if ok else None


def done(prefix):
    p = f"results/{prefix}_per_passage.csv"
    return os.path.exists(p) and os.path.getsize(p) > 0


def launch(prefix, args, gpu):
    lg = f"output/logs/{prefix}.log"
    with open(lg, "a") as fh:
        fh.write(f"[{prefix}] start {time.strftime('%F %T')} on GPU {gpu}\n")
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE="1",
               HF_HUB_CACHE=os.path.abspath("hf_cache"))
    cmd = (f".venv/bin/python analysis/selection_extraction.py {' '.join(args)} --prefix {prefix} "
           f"--out results >> {lg} 2>&1; echo \"[{prefix}] exit=$? at $(date '+%F %T')\" >> {lg}")
    subprocess.Popen(["bash", "-c", f". scripts/gpu_env.sh; set -a; . ./.env; set +a; {cmd}"], env=env,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)


def main(cards=CARDS, only=""):
    jobs = [(p, a) for p, a in JOBS if any(p.startswith(o) for o in only.split(","))]
    assert jobs, f"no job starts with any of {only!r}"
    log(f"start: {len(jobs)} jobs on cards {','.join(cards)}, need {NEED} MiB after reserving "
        f"{RESERVE} per job of ours")
    tries = {p: 0 for p, _ in jobs}
    redraw_logged = not any(p.startswith("selfixR_") for p, _ in jobs)   # not this host's to report
    while True:
        todo = [(p, a) for p, a in jobs if not done(p) and not running(p) and tries[p] < TRIES]
        live = [p for p, _ in jobs if running(p)]
        if not redraw_logged and all(done(p) for p, _ in jobs if p.startswith("selfixR_")):
            with open("output/logs/selfix_redraw_queue.log", "a") as fh:
                fh.write(f"[redraw] drained {time.strftime('%F %T')}\n")
            redraw_logged = True
        if not todo and not live:
            left = [p for p, _ in jobs if not done(p)]
            log("all done" if not left else f"gave up on {left}")
            if not redraw_logged:
                with open("output/logs/selfix_redraw_queue.log", "a") as fh:
                    fh.write(f"[redraw] deadline: gave up on {[p for p in left if 'selfixR' in p]}\n")
            return
        placed = False
        if todo:
            gpu = choose(free_after_reserve(cards))
            if gpu is not None:
                p, a = todo[0]
                tries[p] += 1
                launch(p, a, gpu)
                log(f"{p} -> gpu {gpu} (try {tries[p]})")
                placed = True
        time.sleep(SETTLE if placed else 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default=",".join(CARDS))
    ap.add_argument("--only", default="",
                    help="place only the jobs whose prefix starts with one of these (comma-separated)")
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main(tuple(a.cards.split(",")), a.only)
