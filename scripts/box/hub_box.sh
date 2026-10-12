#!/usr/bin/env bash
# Rented-box stage of scripts/21_hubness.py (score on the GPU), with self-stop.
#   bash scripts/box/hub_box.sh > /workspace/box.log 2>&1   (detached: setsid nohup ... &)
# Ends in a self-stop either way: 2 h after RUN_DONE (time to pull full.tar), or after 30 min with no
# scoring process (watchdog). Stop, not destroy, so the disk survives for a late pull.
set -u
cd /workspace/xrd || exit 1

selfstop() {
  local id key
  id=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^CONTAINER_ID=//p')
  key=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^CONTAINER_API_KEY=//p')
  echo "[box] SELF-STOP $(date -u +%FT%TZ) ($1)"
  curl -s -X PUT -H "Authorization: Bearer $key" -d '{"state":"stopped"}' \
    "https://console.vast.ai/api/v0/instances/$id/"
}

# independent watchdog: no scoring process for 30 min -> stop
(
  idle=0
  while true; do
    sleep 60
    if ps -eo args | grep -q "[2]1_hubness"; then idle=0; else idle=$((idle + 1)); fi
    if [ -f /workspace/RUN_DONE ]; then exit 0; fi
    if [ $idle -ge 30 ]; then selfstop "watchdog: no scoring process for 30 min"; exit 0; fi
  done
) &

echo "[box] start $(date -u +%FT%TZ)"
md5sum -c MANIFEST.md5 && md5sum -c HUB_MANIFEST.md5 || { echo "[box] MD5 MISMATCH"; selfstop "md5 mismatch"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch, numpy; print('[box] torch', torch.__version__, 'numpy', numpy.__version__); assert torch.cuda.is_available()" \
  || { echo "[box] NO CUDA"; selfstop "no cuda"; exit 1; }

quota=$(awk '{ if ($1 == "max") print 0; else printf "%d", $1 / $2 }' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 0)
[ "${quota:-0}" -gt 0 ] || quota=$(nproc)
# rendering only has to keep one GPU fed (~4 ms per pattern per core); more workers just queue chunks
W=$(( quota - 2 )); [ $W -gt 16 ] && W=16; [ $W -lt 1 ] && W=1
echo "[box] cgroup quota ${quota} cores -> ${W} render workers"

ulimit -n 65536
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMPY_MADVISE_HUGEPAGE=0 MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=68719476736   # lesson 22
python scripts/21_hubness.py score --device cuda --workers "$W" < /dev/null
rc=$?
echo "[box] python exit $rc $(date -u +%FT%TZ)"
tar -cf /workspace/full.tar -C cache/real_bench full
md5sum /workspace/full.tar > /workspace/full.tar.md5
touch /workspace/RUN_DONE
echo "[box] RUN_DONE rc=$rc $(du -h /workspace/full.tar | cut -f1)"

# keep the box up 2 h for the pull, then stop
for i in $(seq 1 120); do
  [ -f /workspace/PULLED ] && { echo "[box] pulled; idling until destroyed"; exit 0; }
  sleep 60
done
selfstop "2 h after RUN_DONE"
