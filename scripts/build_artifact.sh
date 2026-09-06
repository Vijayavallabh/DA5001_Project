#!/usr/bin/env bash
# feat-015: build the anonymised artifact directory + zip with a verified SHA-256 manifest.
# Contents: code snapshot (no .git, .venv, output/, secrets), results/*.csv, figures/*.pdf, data/ prompt sets,
# recipes/ (memorising-model recipe; weights are NOT included), tests/, README with reproduction commands.
# Usage: scripts/build_artifact.sh [artifact_dir]   (default: artifact/)
set -e
cd "$(dirname "$0")/.."
ART=${1:-artifact}
rm -rf "$ART"; mkdir -p "$ART"
rsync -a --exclude '.git' --exclude '.venv' --exclude 'output' --exclude 'output.zip' --exclude 'hf_cache' --exclude '.env' --exclude '__pycache__' \
      --exclude '.claude' --exclude '.claude-private' --exclude 'claude-me' --exclude "$ART" --exclude 'artifact*' \
      --exclude 'GOAL.md' --exclude 'AGENTS.md' --exclude 'CLAUDE.md' --exclude 'progress.md' --exclude 'session-handoff.md' \
      --exclude 'feature_list.json' --exclude 'init.sh' --exclude 'figures/legacy' --exclude 'scripts/build_artifact.sh' --exclude 'README_artifact.md' \
      ./ "$ART/"
# anonymity: no author names, emails, or home paths inside the artifact
if grep -rIl --exclude-dir=data -e "vijayavallabh" -e "@gmail" -e "/home/sports" "$ART" | grep -v "^$ART/README" ; then
  echo "identifying strings found above; fix before release" >&2; exit 1
fi
cp README_artifact.md "$ART/README.md"
( cd "$ART" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256 )
( cd "$ART" && sha256sum -c --quiet MANIFEST.sha256 && echo "manifest verified: $(wc -l < MANIFEST.sha256) files" )
rm -f "$ART.zip"; zip -qr "$ART.zip" "$ART"
echo "artifact: $ART/ and $ART.zip ($(du -sh "$ART.zip" | cut -f1))"
