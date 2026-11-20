#!/usr/bin/env bash
# Build manuscript.docx and esi.docx for Royal Society Open Science from the markdown sources
# (pandoc + citeproc, Vancouver numbering from royal-society-open-science.csl).
# Word cannot embed PDF figures, so a temporary copy points at the 600 dpi PNGs.
# Run from the repo root:  bash paper/rsos/build.sh            (refuses while any {{placeholder}} is left)
#                          bash paper/rsos/build.sh --draft    (allows only {{zenodo_doi}}, for review copies)
set -euo pipefail
cd "$(dirname "$0")/../.."
draft=0; [ "${1:-}" = "--draft" ] && draft=1
left=$(sed -n '/^## Abstract/,$p' paper/rsos/manuscript.md | grep -o '{{[A-Za-z0-9_]*}}' | sort -u || true)
if [ "$draft" = 1 ]; then left=$(echo "$left" | grep -v '^{{zenodo_doi}}$' || true); fi
if [ -n "$left" ]; then echo "unfilled placeholders: $left" >&2; exit 1; fi
PYTHONIOENCODING=utf-8 python scripts/verify_claims_rsos.py --md paper/rsos/manuscript.md
cd paper/rsos
for f in fig1_benchmark fig_model fig3_simulated_sweep fig4_real_benchmark fig5_case_zzzuwk fig6_effect_ladder; do
  [ -s "figs/$f.png" ] || { echo "missing figs/$f.png" >&2; exit 1; }
done
sed 's/\.pdf)/.png)/g' manuscript.md > _manuscript_png.md
pandoc _manuscript_png.md --citeproc --bibliography refs_rsos.bib --csl royal-society-open-science.csl \
       --resource-path=. -o manuscript.docx
pandoc esi.md -o esi.docx
rm -f _manuscript_png.md
echo "built paper/rsos/manuscript.docx and paper/rsos/esi.docx"
