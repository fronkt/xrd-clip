#!/usr/bin/env bash
# X2 + X3 repair (10-14): pattern_594 is in both the X2 and the X3 pattern sets, and the two concurrent runs of
# critic2_x23_box.sh shared its cmp_<cand>.cri/.cro files (29_critic2_rerank.py compare, now per-process names):
# 23 X3 and 2 X2 comparisons came back empty and the rest may have read the other run's peak list. Reruns its 50
# shortlist pairs under each peak list into diffs_zs594 / diffs_x3594 (merged locally over the 594 rows), then
# out_fix.tar + FIX_DONE. Runs beside critic2_x23_box.sh, whose guards (6 h after RUN_DONE, onstart hard stop) stop
# the box; pull out_fix.tar before touching PULLED.
#   cd /workspace/xrd; setsid nohup bash scripts/box/critic2_x23_fix594.sh > /workspace/fix.log 2>&1 < /dev/null &
set -u
cd /workspace/xrd || exit 1
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CRITIC2=$(find /workspace/critic2 -type f -name critic2 -perm -u+x | head -1)
echo "[fix] start $(date -u +%FT%TZ), critic2 at $CRITIC2"
python scripts/29_critic2_rerank.py run --k 50 --timeout 14400 --procs 50 --peaks fit_zs.peaks --suffix _zs594 \
  --pids pattern_594 > logs/x2_fix594.log 2>&1 &
P1=$!
python scripts/29_critic2_rerank.py run --k 50 --timeout 14400 --procs 50 --peaks fit_x3.peaks --suffix _x3594 \
  --pids pattern_594 > logs/x3_fix594.log 2>&1 &
P2=$!
for p in $P1 $P2; do wait $p; echo "[fix] worker $p exit $?"; done
echo "[fix] X2 $(wc -l < cache/critic2/diffs_zs594.jsonl), X3 $(wc -l < cache/critic2/diffs_x3594.jsonl) rows"
tar -cf /workspace/out_fix.tar logs/x2_fix594.log logs/x3_fix594.log cache/critic2/diffs_zs594.jsonl \
  cache/critic2/diffs_x3594.jsonl
md5sum /workspace/out_fix.tar > /workspace/out_fix.tar.md5
touch /workspace/FIX_DONE
echo "[fix] FIX_DONE $(date -u +%FT%TZ) $(du -h /workspace/out_fix.tar | cut -f1)"
