#!/usr/bin/env bash
# R3 critic2 re-rank, ranks 11-50 (K = 50 with a 4-h cap per comparison), on the box that ran K = 10
# (critic2_k10_box.sh): the K = 10 run measured 76 CPU-h for 589 comparisons, ~6x below the estimate that had cut
# K to 10, so Frank chose to restore K = 50 (10-13). Uploaded on top of the K = 10 tree: this script + the extra
# candidate CIFs (+ MANIFEST.md5 of those). critic2 is already installed; the K = 10 runner's shell was stopped by
# PID, its last comparison still runs (its row is a duplicate of one this run lists; eval keeps the last row).
#   cd /workspace/xrd; setsid nohup bash scripts/box/critic2_k50_box.sh > /workspace/box50.log 2>&1 < /dev/null &
# Ends in a self-stop either way: right after the pull (PULLED), 6 h after RUN_DONE with no pull, or a watchdog
# stop after 30 min with no worker process.
set -u
cd /workspace/xrd || exit 1

selfstop() {
  local id key
  id=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^CONTAINER_ID=//p')
  key=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^CONTAINER_API_KEY=//p')
  echo "[box] SELF-STOP $(date -u +%FT%TZ) ($1)"
  curl -s -X PUT -H "Authorization: Bearer $key" -d '{"state":"stopped"}' "https://console.vast.ai/api/v0/instances/$id/"
}

(
  idle=0
  while true; do
    sleep 60
    # process NAME critic2 (this script's own name contains "critic2", so an args match would never go idle)
    if ps -eo comm= | grep -qx critic2 || ps -eo args= | grep -q "[2]9_critic2"; then idle=0; else idle=$((idle + 1)); fi
    [ -f /workspace/RUN_DONE ] && exit 0
    if [ $idle -ge 30 ]; then selfstop "watchdog: no worker for 30 min"; exit 0; fi
  done
) &

echo "[box] K = 50 start $(date -u +%FT%TZ)"
md5sum -c --quiet MANIFEST.md5 || { echo "[box] MD5 MISMATCH"; selfstop "md5 mismatch"; exit 1; }
quota=$(awk '{ if ($1 == "max") print 0; else printf "%d", $1 / $2 }' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 0)
[ "${quota:-0}" -gt 0 ] || quota=$(nproc)
export CRITIC2=$(find /workspace/critic2 -type f -name critic2 -perm -u+x | head -1)
echo "[box] quota ${quota}; critic2 at $CRITIC2; $(wc -l < cache/critic2/diffs.jsonl) rows before"

export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
C2=$(( quota - 3 )); [ $C2 -lt 4 ] && C2=4      # one core stays with the K = 10 run's last comparison
mkdir -p logs
python scripts/29_critic2_rerank.py run --k 50 --timeout 14400 --procs "$C2" > logs/c2_run50.log 2>&1
echo "EXIT $?" >> logs/c2_run50.log
while ps -eo args= | grep -q "[2]9_critic2_rerank.py run --k 10"; do sleep 60; done   # the K = 10 straggler
echo "[box] all workers done $(date -u +%FT%TZ); $(wc -l < cache/critic2/diffs.jsonl) rows in diffs.jsonl"
tar -cf /workspace/out.tar logs cache/critic2/diffs.jsonl
md5sum /workspace/out.tar > /workspace/out.tar.md5
touch /workspace/RUN_DONE
echo "[box] RUN_DONE $(du -h /workspace/out.tar | cut -f1)"
for i in $(seq 1 360); do
  [ -f /workspace/PULLED ] && { selfstop "pulled"; exit 0; }
  sleep 60
done
selfstop "6 h after RUN_DONE"
