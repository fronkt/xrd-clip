#!/usr/bin/env bash
# Revision CPU box (tasks/todo.md Step 3): R3 critic2 VC-GPWDF fits and comparisons (the sweep runs on the GPU box).
#   setsid nohup bash scripts/box/rev_cpu_box.sh > /workspace/box.log 2>&1 &
# CSD-derived inputs (candidate CIFs, stick shards, checkpoint) arrive only by a manual-mode upload and never leave
# the box except as aggregate outputs (DIFF values per refcode pair, peak lists of the open opXRD patterns). Ends in a self-stop either way:
# 2 h after RUN_DONE (time to pull), or a watchdog stop after 30 min with no worker process.
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
    if ps -eo args | grep -qE "[c]ritic2|[2]9_critic2"; then idle=0; else idle=$((idle + 1)); fi
    [ -f /workspace/RUN_DONE ] && exit 0
    if [ $idle -ge 30 ]; then selfstop "watchdog: no worker for 30 min"; exit 0; fi
  done
) &

echo "[box] start $(date -u +%FT%TZ)"
md5sum -c MANIFEST.md5 || { echo "[box] MD5 MISMATCH"; selfstop "md5 mismatch"; exit 1; }
quota=$(awk '{ if ($1 == "max") print 0; else printf "%d", $1 / $2 }' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 0)
[ "${quota:-0}" -gt 0 ] || quota=$(nproc)
echo "[box] cgroup quota ${quota} cores"

# critic2 1.4 (the version validated locally on Windows), checksum from the release's SHA256SUMS
mkdir -p /workspace/critic2 && cd /workspace/critic2
curl -sSLO https://github.com/aoterodelaroza/critic2/releases/download/1.4/critic2-linux-x86_64.tar.gz
echo "2020d490ffba8d3d340f0a47f26eebf722391b959265d5c5c29de17c29a754f9  critic2-linux-x86_64.tar.gz" | sha256sum -c \
  || { echo "[box] critic2 checksum FAIL"; selfstop "critic2 checksum"; exit 1; }
tar -xzf critic2-linux-x86_64.tar.gz
export CRITIC2=$(find /workspace/critic2 -type f -name critic2 -perm -u+x | head -1)
echo "[box] critic2 at $CRITIC2"; printf "end\n" > t.cri; "$CRITIC2" t.cri | grep -E "version|features"
cd /workspace/xrd

export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
C2=$(( quota - 2 )); [ $C2 -lt 4 ] && C2=4
mkdir -p logs results
python scripts/29_critic2_rerank.py fit --procs "$C2" > logs/c2_fit.log 2>&1
python scripts/29_critic2_rerank.py run --procs "$C2" > logs/c2_run.log 2>&1
echo "EXIT $?" >> logs/c2_run.log
wait
echo "[box] all workers done $(date -u +%FT%TZ)"
tar -cf /workspace/out.tar logs cache/critic2/diffs.jsonl $(ls cache/critic2/*/fit.peaks 2>/dev/null)
md5sum /workspace/out.tar > /workspace/out.tar.md5
touch /workspace/RUN_DONE
echo "[box] RUN_DONE $(du -h /workspace/out.tar | cut -f1)"
for i in $(seq 1 120); do
  [ -f /workspace/PULLED ] && { echo "[box] pulled"; exit 0; }
  sleep 60
done
selfstop "2 h after RUN_DONE"
