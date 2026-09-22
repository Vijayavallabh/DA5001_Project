#!/usr/bin/env bash
# Keep the two hosts in sync and print what is running on both. Safe to run repeatedly.
#
# DIRECTIONS ARE DELIBERATE AND ASYMMETRIC:
#   code    local -> host B   (local is authoritative; edits are made here)
#   results host B -> local   (--update, so a newer local file is never clobbered)
# output/ is NOT synced: it is tens of GB, gitignored and host-specific. hf_cache/ never moves.
#
# Process detection matches on the EXECUTABLE FIELD and never on the absence of a pattern
# (caution (c)): `pgrep -f X` matches the shell running it, and a waiter keyed on `! pgrep` can
# never exit. Staleness is read off the filesystem, which is the evidence that works.
set -u
cd "$(dirname "$0")/.."
H=PrakashDGX_H2
# The remote path is the one thing here that cannot be derived from $0. It is read from the
# environment so the repository name is not written into a file the artifact ships.
R="${DGX_REMOTE_DIR:-\~/v/$(basename "$PWD")}"
EX="--exclude .git --exclude __pycache__ --exclude hf_cache --exclude output --exclude .venv
    --exclude .uv_cache --exclude data/bench/cotaeval_raw --exclude *.pyc --exclude .pt_now.txt"

case "${1:-status}" in
  push)   # code out
    # data/tokenswap_G.txt is METHOD data, not a corpus: TokenSwap's published 110-word set, which
    # analysis/tokenswap_decode.py refuses to run without. It is neither code nor a bench corpus,
    # so the include list below missed it and feat-165's first launch died on FileNotFoundError --
    # caution (aw), third kind: a dependency that is a committed FILE but not a matched PATTERN.
    rsync -az $EX --include 'analysis/***' --include 'scripts/***' --include 'tests/***' \
          --include 'figures/***' --include 'data/tokenswap_G.txt' \
          --include '*.py' --include '*.sh' --include '*.md' \
          --include '*.json' --include '*/' --exclude '*' ./ "$H:$R/" && echo "[sync] code -> host B ok"
    # data/bench/<corpus>/ is a directory of symlinks with ABSOLUTE targets, so a tree copied from
    # this host points at this host's paths and dangles silently on the other one -- invisible
    # until h1.py dies on a missing file, which is how feat-159's first launch failed with six of
    # twelve corpora broken. Re-anchor them on every push; it is idempotent and only ever touches a
    # link that is already broken.
    ssh "$H" "cd $R && ./scripts/fix_bench_symlinks.sh" 2>/dev/null | sed 's/^/[sync] /' ;;
  pull)   # results back, never clobbering a newer local file
    rsync -az --update "$H:$R/results/" results/ && echo "[sync] results <- host B ok"
    # CODE CREATED ON HOST B MUST COME BACK TOO. The first version of this script pushed code one
    # way and pulled only results, so eight launchers written directly on host B lived nowhere
    # else -- including run_memfree.sh, run_cpfuse.sh and run_frontier_judge.sh, which PRODUCED
    # COMMITTED NUMBERS. A producing command that exists on one host is not reproducible and would
    # not have shipped in the artifact. Report anything remote-only rather than silently ignoring.
    # ANCHOR THE PATTERNS. An rsync include with no leading slash matches at ANY depth, so
    # 'analysis/***' also matched .uv_cache/archive-v0/*/torch/_inductor/analysis/, and the
    # remote-only report filled with a package cache the moment UV_CACHE_DIR moved into the repo.
    # A report that cries wolf is a report nobody reads, which is the whole point of this check.
    NEW=$(rsync -rn --ignore-existing --out-format='%n' $EX \
          --include '/analysis/***' --include '/scripts/***' --include '*/' --exclude '*' \
          "$H:$R/" ./ 2>/dev/null | grep -Ev '/$|__pycache__' || true)
    if [ -n "$NEW" ]; then
      echo "[sync] REMOTE-ONLY CODE (not in the repo):"; echo "$NEW" | sed 's/^/          /'
    else
      echo "[sync] no remote-only code"
    fi ;;
  both)   "$0" push && "$0" pull ;;
  push-results)  # a results file CORRECTED here must reach host B, and nothing else pushes it
    # `push` sends code patterns only and `pull` runs B -> local, so a correction made locally to a
    # committed CSV never crossed. results/scorer_scale_6rung{,_bands}.csv carried the stale
    # 14.7701 on host B for a day after 14.7700 was measured and fixed here (caution (ax)): a
    # number typed from knowledge, corrected in one place only. --update means a newer file on B
    # is never clobbered, so a fresh arm scored there still wins.
    # $EX IS NOT OPTIONAL HERE. Without it `--include '*/'` tells rsync to create every local
    # directory on the remote, and that includes all of `.venv`: it recreated this host's
    # `tqdm-4.70.0.dist-info` and thirty others as EMPTY directories beside host B's real
    # `tqdm-4.70.1.dist-info`, which makes `importlib.metadata.version` return None and kills every
    # job with "Unable to compare versions for tqdm>=4.60: found=None". Eight cells died of it on
    # 2026-09-22 before the cause was found. rsync without `--delete` cannot REMOVE a file, which
    # is what the first diagnosis checked; it can freely ADD a directory, which is what broke it.
    rsync -az --update $EX --include 'results/***' --include '*/' --exclude '*' \
          ./ "$H:$R/" && echo "[sync] corrected results -> host B ok" ;;
  verify) # content equality over code + results, IGNORING line endings
    # A raw md5 comparison reports five permanent phantom diffs: `csv.writer` with `newline=""`
    # uses the csv module's default lineterminator, which is CRLF, so a file freshly written on
    # host B carries \r\n while its committed twin here is LF (git tracks them `i/lf w/lf`).
    # Row-for-row the data are identical. Comparing raw bytes would have this check crying drift
    # forever and so training us to ignore it, which is worse than not having it.
    python3 - "$H" "$R" <<'PYEND'
import hashlib, os, subprocess, sys
host, remote = sys.argv[1], sys.argv[2]
EXT = (".py", ".sh", ".md", ".csv", ".json")
DIRS = ("analysis", "scripts", "tests", "results")

def norm(b):
    return hashlib.md5(b.replace(b"\r\n", b"\n")).hexdigest()

local = {}
for d in DIRS:
    for root, _sub, files in os.walk(d):
        if "__pycache__" in root:
            continue
        for f in files:
            if f.endswith(EXT):
                fp = os.path.join(root, f)
                local[fp] = norm(open(fp, "rb").read())

cmd = (f"cd {remote} && find " + " ".join(DIRS) + " -type f "
       + r"\( " + " -o ".join(f"-name '*{e}'" for e in EXT) + r" \) "
       + "-not -path '*__pycache__*' -print0 "
       + "| xargs -0 -n1 sh -c 'printf \"%s \" \"$0\"; tr -d \"\\r\" < \"$0\" | md5sum | cut -d\" \" -f1'")
out = subprocess.run(["ssh", host, cmd], capture_output=True, text=True).stdout
rem = {}
for line in out.splitlines():
    path, _, h = line.rpartition(" ")
    if path.strip():
        rem[path.strip()] = h.strip()

both = set(local) & set(rem)
diff = sorted(p for p in both if local[p] != rem[p])
only_l, only_r = sorted(set(local) - set(rem)), sorted(set(rem) - set(local))
print(f"  verify  {len(both)} files on both, {len(diff)} differ in CONTENT")
for p in diff[:15]:
    print("    DIFF", p)
print(f"  verify  local-only {len(only_l)}, hostB-only {len(only_r)}")
for p in only_r[:10]:
    print("    hostB-only (never pulled!)", p)
print("  verify  IN SYNC" if not diff and not only_r else "  verify  ACTION NEEDED")
PYEND
    ;;
  status) : ;;
  # AN UNKNOWN SUBCOMMAND MUST NOT BE A SILENT NO-OP. `case` with no default falls straight through
  # to the status block, so `pull-results` -- a name that does not exist -- printed the GPU tables
  # and the .done markers and looked exactly like a successful sync. Three files scored on host B
  # were believed pulled and were not (2026-09-22). A typo in a sync direction is the one place
  # where doing nothing must never look like doing the thing.
  *) echo "usage: $0 {push|pull|both|push-results|verify|status}" >&2; exit 2 ;;
esac

echo
echo "=== LOCAL $(date +%H:%M:%S) ==="
ps -eo pid,etime,args --no-headers \
  | awk '$3 ~ /python$/ && (/h1\.py/ || /h2\.py/ || /analysis\//) {printf "  job  %-8s %-12s %s\n", $1, $2, substr($0, index($0,$3), 70)}'
ps -eo pid,args --no-headers \
  | awk '$2 ~ /bash$/ && /scripts\/run_/ {printf "  shell %-8s %s\n", $1, substr($0, index($0,$2), 60)}'
env -u LD_LIBRARY_PATH nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
  --format=csv,noheader 2>/dev/null | sed 's/^/  gpu   /' | head -5
for L in output/logs/comma7b128_card1.log output/logs/comma7b128_merge.log; do
  [ -f "$L" ] && printf "  log   %-42s %4d min since last write\n" \
    "$(basename "$L")" "$(( ($(date +%s) - $(stat -c %Y "$L")) / 60 ))"
done

echo
echo "=== HOST B ==="
ssh -o ConnectTimeout=15 "$H" 'nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader | sed "s/^/  gpu   /"; echo "  --- sessions ---"; tmux -S ~/v/tmux.sock ls 2>/dev/null | sed "s/^/  tmux  /" || echo "  tmux  (none)"; echo "  --- markers ---"; ls -t ~/v/logs/*.done ~/v/logs/*.fail 2>/dev/null | head -4 | sed "s|.*/|  mark  |"'
