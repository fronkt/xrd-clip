#!/usr/bin/env bash
# Build manuscript.docx and esi.docx from the markdown sources (pandoc + citeproc, RSC numbered style).
# Word cannot embed PDF figures, so a temporary copy points at the 300 dpi PNGs.
# Run from the repo root:  bash paper/dd/build.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
python scripts/verify_claims.py paper/dd/manuscript.md
cd paper/dd
sed 's/\.pdf)/.png)/g' manuscript.md > _manuscript_png.md
pandoc _manuscript_png.md --citeproc --bibliography refs_dd.bib --csl royal-society-of-chemistry.csl \
       --resource-path=. -o manuscript.docx
pandoc esi.md -o esi.docx
rm -f _manuscript_png.md
echo "built paper/dd/manuscript.docx and paper/dd/esi.docx"
