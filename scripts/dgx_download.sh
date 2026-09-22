#!/usr/bin/env bash
# Download one repo into hf_cache and write a DONE marker. Never wait on pgrep (caution (c)):
# the marker file is the completion condition.
set -uo pipefail
source "$HOME/v/env.sh"
# Derived, not hardcoded: a literal `$HOME/v/<project>` puts the repository name inside
# every launcher, and scripts/build_artifact.sh refuses to ship a tree containing it.
cd "$(dirname "$0")/.."
REPO="$1"; TAG="$2"
MARK="$HOME/v/logs/dl_${TAG}"
rm -f "${MARK}.done" "${MARK}.fail"
set -a; source .env 2>/dev/null; set +a
.venv/bin/python - <<PY >"${MARK}.log" 2>&1
from huggingface_hub import snapshot_download
p = snapshot_download("$REPO", max_workers=8,
                      ignore_patterns=["*.pth", "original/*", "*.gguf"])
print("PATH", p)
PY
if [ $? -eq 0 ]; then touch "${MARK}.done"; else touch "${MARK}.fail"; fi
