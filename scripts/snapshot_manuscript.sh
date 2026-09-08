#!/usr/bin/env bash
# Copy the ICLR manuscript sources into this repo so they have version history.
# ~/sub/satml stays authoritative; this is a backup, not a working copy.
set -e
cd "$(dirname "$0")/.."
S=/mnt/md0/IITM/BackUp/Home/vijayavallabh/sub/satml
mkdir -p manuscript_snapshot/sections
cp "$S/iclr_2027.tex" "$S/references.bib" manuscript_snapshot/
for f in $(grep -o 'sections/[a-z_0-9]*' "$S/iclr_2027.tex" | sort -u); do
  [ -f "$S/$f.tex" ] && cp "$S/$f.tex" manuscript_snapshot/sections/
done
echo "snapshot: $(ls manuscript_snapshot/sections | wc -l) section files + iclr_2027.tex + references.bib"
