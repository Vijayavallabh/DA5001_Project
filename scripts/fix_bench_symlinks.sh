#!/usr/bin/env bash
# Re-point dangling corpus symlinks at the repo they actually live in.
#
# data/bench/<corpus>/ is a directory of symlinks to the committed prompt sets, which is what lets
# a run on a new corpus take the same code path as every run on record (h1.py --data-dir, and
# tests/test_bench_corpora.py asserts every entry is a symlink). build_bench_corpora.py writes
# those links with ABSOLUTE targets, so rsyncing the tree to a second host copies links that point
# at the first host's paths. They are silently dangling until h1.py dies on a missing file --
# which is how feat-159's smoke test failed on host B, with six of twelve corpora affected.
#
# Idempotent, and it only ever touches a symlink that is already broken: a resolvable link is left
# exactly as it is, and nothing under data/*.jsonl (read-only, committed) is written.
set -u
cd "$(dirname "$0")/.."
ROOT="$PWD"
PROJ=$(basename "$PWD")
fixed=0; dangling=0
for link in data/bench/*/*; do
  [ -L "$link" ] || continue
  [ -e "$link" ] && continue
  dangling=$((dangling + 1))
  target=$(readlink "$link")
  # Re-anchor any absolute target that ends in .../<this repo>/<rest> onto THIS repo.
  # The name is taken from the checkout rather than written in, so the artifact does
  # not ship the repository name (scripts/build_artifact.sh refuses a tree with it).
  rest="${target##*/$PROJ/}"
  if [ "$rest" != "$target" ] && [ -e "$ROOT/$rest" ]; then
    ln -sfn "$ROOT/$rest" "$link"
    fixed=$((fixed + 1))
  else
    echo "[bench] UNFIXABLE $link -> $target" >&2
  fi
done
echo "[bench] $dangling dangling, $fixed re-pointed at $ROOT"
[ "$dangling" -eq "$fixed" ]
