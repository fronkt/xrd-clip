#!/usr/bin/env bash
# Build + format + audit the CrystEngComm package into PKG (default D:/xrd-clip/crystengcomm_submission).
# Needs the journal-submission skill (apply_format.py, audit.py) and PKG/requirements.md (the approved spec).
# The Word->PDF render is a separate step (skill scripts/export_pdf.ps1).
set -euo pipefail
cd "$(dirname "$0")"
PKG="${1:-D:/xrd-clip/crystengcomm_submission}"
SK="${JS_SKILL:-$HOME/.claude/skills/journal-submission/scripts}"
MS="$PKG/Cai_PXRD-retrieval_CEC_manuscript.docx"
bash build.sh "$PKG/_build"
cp "$PKG/_build/manuscript.docx" "$MS"
cp "$PKG/_build/esi.docx" "$PKG/Cai_PXRD-retrieval_CEC_SI.docx"
python "$SK/apply_format.py" "$MS" --spec "$PKG/requirements.md" --out "$MS"
python postfix.py "$MS"
python "$SK/audit.py" "$MS" --spec "$PKG/requirements.md" --out "$PKG/compliance_report.md"
tail -n 1 "$PKG/compliance_report.md"
