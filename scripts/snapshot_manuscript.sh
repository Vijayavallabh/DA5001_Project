#!/usr/bin/env bash
# Copy the ICLR manuscript sources into this repo so they have version history.
# ~/sub/satml stays authoritative; this is a backup, not a working copy.
set -e
cd "$(dirname "$0")/.."
# manuscript dir; override with SATML_DIR. Relative default keeps this file portable.
S=${SATML_DIR:-../sub/satml}
mkdir -p manuscript_snapshot/sections
cp "$S/iclr_2027.tex" "$S/references.bib" manuscript_snapshot/
# Follow \input from the top-level file AND from the section files it pulls in: since the v8
# restructure, onset.tex is \input by orders.tex rather than by iclr_2027.tex, and scanning only
# the top level silently left a stale copy in the snapshot.
for f in $(cat "$S/iclr_2027.tex" "$S"/sections/*.tex 2>/dev/null \
             | grep -o 'sections/[a-z_0-9]*' | sort -u); do
  [ -f "$S/$f.tex" ] && cp "$S/$f.tex" manuscript_snapshot/sections/
done
echo "snapshot: $(ls manuscript_snapshot/sections | wc -l) section files + iclr_2027.tex + references.bib"
