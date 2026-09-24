#!/usr/bin/env bash
# feat-015: build the anonymised artifact directory + zip with a verified SHA-256 manifest.
# Contents: code snapshot (no .git, .venv, output/, secrets), results/*.csv, figures/*.pdf, data/ prompt sets,
# but NOT data/gutenberg/ or data/bench/ -- re-fetchable corpora that their own scripts rebuild
# (analysis/build_gutenberg_excerpts.py, analysis/build_bench_corpora.py) and that
# README_artifact.md says are rebuilt rather than shipped,
# recipes/ (memorising-model recipe; weights are NOT included), tests/, README with reproduction commands.
# Usage: scripts/build_artifact.sh [artifact_dir]   (default: artifact/)
set -e
cd "$(dirname "$0")/.."
ART=${1:-artifact}
rm -rf "$ART"; mkdir -p "$ART"
rsync -a --exclude '.git' --exclude '.venv' --exclude 'output' --exclude 'output.zip' --exclude 'hf_cache' --exclude '.env' --exclude '__pycache__' --exclude '.pytest_cache' \
      --exclude '.claude' --exclude '.claude-private' --exclude 'claude-me' --exclude "$ART" --exclude 'artifact*' \
      --exclude 'data/gutenberg' --exclude 'data/bench' --exclude 'NVIDIA-Linux-*' --exclude 'torchinductor_*' --exclude 'GOAL.md' --exclude 'AGENTS.md' --exclude 'CLAUDE.md' --exclude 'progress.md' --exclude 'session-handoff.md' \
      --exclude 'feature_list.json' --exclude 'init.sh' --exclude 'figures/legacy' --exclude 'manuscript_snapshot' --exclude 'scripts/build_artifact.sh' --exclude 'scripts/sync_status.sh' --exclude 'README_artifact.md' \
      ./ "$ART/"
cp README_artifact.md "$ART/README.md"
# anonymity: no author names, emails, institutions, hostnames or absolute paths inside the artifact.
# Runs after the README is copied in, because that file is the likeliest place for a leak and the
# earlier version of this check ran before the copy and so never saw it.
if grep -rIliE --exclude-dir=data -e "vijayavallabh" -e "be23b041" -e "smail\.iitm" -e "iit ?madras" \
     -e "da5001" -e "cessa-g242" -e "/home/sports" -e "/mnt/md0" -e "prakashdgx" -e "prachh" \
     -e "@[a-z0-9.-]+\.(ac\.in|edu)" "$ART" ; then
  echo "identifying strings found in the files above; fix before release" >&2; exit 1
fi
# The scan above reads file CONTENTS. A path can leak on its own: torch's compile cache is named
# `torchinductor_$USER` and shipped for as long as the artifact has existed, carrying the account
# name in a directory name that no content grep would ever see.
if find "$ART" -iname '*sports*' -o -iname '*vijayavallabh*' -o -iname '*iitm*' | grep -q . ; then
  echo "identifying strings in PATH names:" >&2
  find "$ART" -iname '*sports*' -o -iname '*vijayavallabh*' -o -iname '*iitm*' >&2
  exit 1
fi
( cd "$ART" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256 )
( cd "$ART" && sha256sum -c --quiet MANIFEST.sha256 && echo "manifest verified: $(wc -l < MANIFEST.sha256) files" )
MAX_MB=${ARTIFACT_MAX_MB:-80}
SIZE_MB=$(du -sm "$ART" | cut -f1)
if [ "$SIZE_MB" -gt "$MAX_MB" ]; then
  echo "artifact is ${SIZE_MB} MB, over the ${MAX_MB} MB guard. Largest entries:" >&2
  du -sm "$ART"/* | sort -rn | head -5 >&2
  echo "Ship code, results and figures; re-fetchable data is rebuilt by its own script." >&2
  echo "Raise ARTIFACT_MAX_MB only if the growth is genuinely content." >&2
  exit 1
fi
rm -f "$ART.zip"; zip -qr "$ART.zip" "$ART"
echo "artifact: $ART/ and $ART.zip ($(du -sh "$ART.zip" | cut -f1), dir ${SIZE_MB} MB)"
