#!/usr/bin/env bash
# One-command eval battery for a trained XRD-CLIP checkpoint.
# Runs the headline robustness sweep, the two-phase mixture experiment, and prints the week-1
# gate verdict. Point it at a checkpoint + the matching sticks/splits.
#
#   bash scripts/run_eval.sh [STICKS] [TEST_REFS] [CKPT] [EMBED_DIM]
#
# Defaults are the pilot. For the full GPU run, pass cache/sticks, refcodes/scaffold_test.txt,
# runs/xrdclip/best.pt, 256.
set -euo pipefail

STICKS="${1:-cache/pilot_sticks}"
TEST_REFS="${2:-refcodes_pilot/scaffold_test.txt}"
CKPT="${3:-runs/pilot_clip/best.pt}"
EMBED="${4:-128}"

export PYTHONPATH=src
echo "== robustness sweep (headline figure) =="
python scripts/05_robustness.py --sticks "$STICKS" --test_refs "$TEST_REFS" --ckpt "$CKPT" \
    --embed_dim "$EMBED" --out results/robustness.json --fig results/robustness.png

echo "== two-phase mixture experiment =="
python scripts/07_mixtures.py --sticks "$STICKS" --test_refs "$TEST_REFS" --ckpt "$CKPT" \
    --embed_dim "$EMBED" --out results/mixtures.json

echo "== done. figures + json under results/ =="
