#!/usr/bin/env bash
# R1 CSLS ln q v2 score, alone on the GPU (10-13 follow-up: on box A it ran out of memory beside the R4c retrain,
# and the retry never ran because the watchdog stopped the box first). Same command as rev_gpu_box.sh step 2.
#   setsid nohup bash scripts/box/lnq2_box.sh > /workspace/lnq2.log 2>&1 < /dev/null &
# A local loop pulls out_lnq2.tar on LNQ2_DONE and touches PULLED. Self-stop: 6 h after LNQ2_DONE without PULLED,
# or the watchdog after 20 min with no 21_hubness process. The score itself is capped at 3 h.
set -u
cd /workspace/xrd || exit 1

selfstop() {
  local id key
  id=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^CONTAINER_ID=//p')
  key=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^CONTAINER_API_KEY=//p')
  echo "[lnq2] SELF-STOP $(date -u +%FT%TZ) ($1)"
  curl -s -X PUT -H "Authorization: Bearer $key" -d '{"state":"stopped"}' "https://console.vast.ai/api/v0/instances/$id/"
}

(
  idle=0
  while true; do
    sleep 60
    if ps -eo args | grep -qE "[2]1_hubness"; then idle=0; else idle=$((idle + 1)); fi
    [ -f /workspace/LNQ2_DONE ] && exit 0
    if [ $idle -ge 20 ]; then selfstop "watchdog: no 21_hubness for 20 min"; exit 0; fi
  done
) &

echo "[lnq2] start $(date -u +%FT%TZ)"
rm -f /workspace/PULLED
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
python -c "import torch; assert torch.cuda.is_available()" || { selfstop "no cuda"; exit 1; }
pip install -q scipy pymatgen > /dev/null 2>&1 || true
ulimit -n 65536
export PYTHONPATH=src NUMPY_MADVISE_HUGEPAGE=0 MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=68719476736
mkdir -p logs cache/real_bench_lnq2r
cp cache/real_bench/hub_ctx.npz cache/real_bench_lnq2r/
OMP_NUM_THREADS=1 LNQ_VERSION=2 timeout 3h python scripts/21_hubness.py score --device cuda --workers 8 \
  --work cache/real_bench_lnq2r < /dev/null > logs/hub_lnq2_retry.log 2>&1
echo "EXIT $?" >> logs/hub_lnq2_retry.log
tar -cf /workspace/out_lnq2.tar logs/hub_lnq2_retry.log cache/real_bench_lnq2r/full
md5sum /workspace/out_lnq2.tar > /workspace/out_lnq2.tar.md5
touch /workspace/LNQ2_DONE
echo "[lnq2] LNQ2_DONE $(du -h /workspace/out_lnq2.tar | cut -f1) $(date -u +%FT%TZ)"
for i in $(seq 1 360); do
  [ -f /workspace/PULLED ] && { echo "[lnq2] pulled"; exit 0; }
  sleep 60
done
selfstop "6 h after LNQ2_DONE"
