#!/usr/bin/env bash
# Rented-box stage of scripts/19_real_bench.py (classical chunks only), with self-stop.
#   bash scripts/box/real_bench_box.sh <chunk> > /workspace/box.log 2>&1   (inside tmux)
# Ends in a self-stop either way: 2 h after RUN_DONE (time to pull chunks.tar), or after 30 min with no
# python worker alive (watchdog). Stop, not destroy, so the disk survives for a late pull.
set -u
CHUNK=${1:-2500}
cd /workspace/xrd || exit 1

selfstop() {
  local id key
  id=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^CONTAINER_ID=//p')
  key=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^CONTAINER_API_KEY=//p')
  echo "[box] SELF-STOP $(date -u +%FT%TZ) ($1)"
  curl -s -X PUT -H "Authorization: Bearer $key" -d '{"state":"stopped"}' \
    "https://console.vast.ai/api/v0/instances/$id/"
}

# independent watchdog: no python worker for 30 min -> stop
(
  idle=0
  while true; do
    sleep 60
    if ps -eo args | grep -q "[1]9_real_bench"; then idle=0; else idle=$((idle + 1)); fi
    if [ -f /workspace/RUN_DONE ]; then exit 0; fi
    if [ $idle -ge 30 ]; then selfstop "watchdog: no worker for 30 min"; exit 0; fi
  done
) &

echo "[box] start $(date -u +%FT%TZ)"
md5sum -c MANIFEST.md5 || { echo "[box] MD5 MISMATCH"; selfstop "md5 mismatch"; exit 1; }

quota=$(awk '{ if ($1 == "max") print 0; else printf "%d", $1 / $2 }' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 0)
[ "${quota:-0}" -gt 0 ] || quota=$(nproc)
W=$(( quota - 4 )); [ $W -gt 120 ] && W=120; [ $W -lt 1 ] && W=1
echo "[box] cgroup quota ${quota} cores -> ${W} workers, chunk ${CHUNK}"

ulimit -n 65536
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/19_real_bench.py --box_ctx cache/real_bench/box_ctx.npz --work cache/real_bench \
  --chunk "$CHUNK" --workers "$W" < /dev/null
rc=$?
echo "[box] python exit $rc $(date -u +%FT%TZ)"
tar -cf /workspace/chunks.tar -C cache/real_bench chunks
( cd cache/real_bench/chunks && ls | wc -l ) > /workspace/N_CHUNKS
md5sum /workspace/chunks.tar > /workspace/chunks.tar.md5
touch /workspace/RUN_DONE
echo "[box] RUN_DONE rc=$rc chunks=$(cat /workspace/N_CHUNKS)"

# keep the box up 2 h for the pull, then stop
for i in $(seq 1 120); do
  [ -f /workspace/PULLED ] && { echo "[box] pulled; idling until destroyed"; exit 0; }
  sleep 60
done
selfstop "2 h after RUN_DONE"
