#!/usr/bin/env bash
# Build the CrystEngComm (RSC) manuscript and SI from the markdown sources.
# pandoc + citeproc with the RSC style (superscript Vancouver numbers), then ref_plain.lua so that
# "ref. N" stays in running text. Figures are the exact PNGs of the RSOS-262915 submission (figs/Fig1-6.png).
# Run from the repo root:  bash paper/cec/build.sh [OUTDIR]     (default OUTDIR = paper/cec/build)
set -euo pipefail
cd "$(dirname "$0")"
out="${1:-build}"
mkdir -p "$out"
for i in 1 2 3 4 5 6; do [ -s "figs/Fig$i.png" ] || { echo "missing figs/Fig$i.png" >&2; exit 1; }; done
grep -q '{{' manuscript.md && { echo "unfilled placeholder in manuscript.md" >&2; exit 1; }
pandoc manuscript.md --citeproc --lua-filter=ref_plain.lua --resource-path=. -o "$out/manuscript.docx"
pandoc esi.md -o "$out/esi.docx"
echo "built $out/manuscript.docx and $out/esi.docx"
